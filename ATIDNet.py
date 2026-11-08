# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from losses import bgr2hsv
from wtconv import WTConv2d

T_MAX = 256 * 256

from torch.utils.cpp_extension import load

wkv_cuda = load(name="wkv", sources=["./cuda/wkv_op.cpp",
                                     "./cuda/wkv_cuda.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '-O3', '-Xptxas -O3',
                                   f'-DTmax={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class OmniShift(nn.Module):
    def __init__(self, dim, wt_levels=2):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False)
        self.act1 = nn.ReLU()
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.act2 = nn.ReLU()
        self.wtconv = WTConv2d(in_channels=dim, out_channels=dim, kernel_size=5, wt_levels=wt_levels)
        self.conv1x1_2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False)

    def forward(self, x):
        out1x1 = self.conv1x1(x)
        out1x1 = self.act1(out1x1)
        out3x3 = self.conv3x3(out1x1)
        out3x3 = self.act1(out3x3)
        out5x5 = self.wtconv(out3x3)
        out5x5 = self.conv1x1_2(out5x5)
        out = x + out5x5
        return out


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy',
                 key_norm=False, wt_levels=3, shift_mode='q_shift'):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        # self.device = None
        attn_sz = n_embd
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)
        self.recurrence = 2
        self.omni_shift = OmniShift(dim=n_embd, wt_levels=wt_levels)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.v_fusion = nn.Linear(n_embd * 2, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
        self.df = nn.Linear(n_embd, attn_sz, bias=False)

    def jit_func(self, x, resolution, y):
        # Mix x with the previous timestep to produce xk, xv, xr
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        y1 = rearrange(y, 'b c h w -> b (h w) c')
        k = self.key(x)
        v = self.value(x)
        fusion_r_df = x * self.df(y1)
        r = self.receptance(fusion_r_df)
        sr = torch.sigmoid(r)
        return sr, k, v, y

    def forward(self, x, resolution, y):
        B, T, C = x.size()
        h, w = resolution
        self.device = x.device
        sr, k, v, y = self.jit_func(x, resolution, y)
        for j in range(self.recurrence):
            v_vertical = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
            v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
            v_horizontal = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w)
            v_horizontal = rearrange(v_horizontal, 'b (w h) c -> b (h w) c', h=h, w=w)
            v = self.v_fusion(torch.cat([v_vertical, v_horizontal], dim=-1))
        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x, y


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy',
                 key_norm=False, wt_levels=3, shift_mode='q_shift'):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.omni_shift = OmniShift(dim=n_embd, wt_levels=wt_levels)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.df = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, resolution, y):
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        y1 = rearrange(y, 'b c h w -> b (h w) c')

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x * self.df(y1))) * kv
        return x, y


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False, wt_levels=3):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, init_mode,
                                    key_norm=key_norm, wt_levels=wt_levels)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, hidden_rate,
                                    init_mode, key_norm=key_norm, wt_levels=wt_levels)

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, mix_input):
        x, y = mix_input
        b, c, h, w = x.shape
        resolution = (h, w)
        # x = self.dwconv1(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')
        spa_out, y1 = self.att(self.ln1(x), resolution, y)
        spa_out = self.gamma1 * spa_out
        x = x + spa_out
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # x = self.dwconv2(x) + x
        # x = rearrange(x, 'b c h w -> b (h w) c')
        channel_out, y2 = self.ffn(self.ln2(x), resolution, y)
        channel_out = self.gamma2 * channel_out
        x = x + channel_out
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x, y


class Depth_Illumination_Estimator(nn.Module):
    def __init__(self, in_channel=3, dim=8, out_dim=32):
        super(Depth_Illumination_Estimator, self).__init__()
        self.enc1 = self._encoder_block(in_channel, dim)
        self.enc2 = self._encoder_block(dim, dim * 2)
        self.enc3 = self._encoder_block(dim * 2, dim * 4)
        self.enc4 = nn.Sequential(
            nn.Conv2d(1, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=5, stride=1, padding=2, groups=dim),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )
        # Global pooling and linear layers
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(dim * 4, 5)
        self.sigmoid = nn.Sigmoid()

    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x3 = self.gpool(x3)
        x3 = x3.view(b, -1)
        x_out = self.linear(x3)
        w = self.sigmoid(x_out[::, 0]).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        alpha = x_out[::, 1].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        beta = x_out[::, 2].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        gamma = x_out[::, 3].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        epsilon = x_out[::, 4].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        x = x * w + (1 - w)
        hsv = bgr2hsv(x)
        S = hsv[:, 1, :, :].unsqueeze(1)
        V = hsv[:, 2, :, :].unsqueeze(1)
        coarse_depth = alpha + beta * V + gamma * S + epsilon
        out_coarse_depth = self.enc4(coarse_depth)
        return out_coarse_depth, coarse_depth


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Restore_RWKV(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=24,
                 num_blocks=[2, 4, 4, 8],
                 num_refinement_blocks=2,
                 wt_levels=[3, 3, 2, 2]
                 ):
        super(Restore_RWKV, self).__init__()

        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.encoder_level1 = nn.Sequential(
            *[Block(n_embd=dim, n_layer=num_blocks[0], layer_id=i, wt_levels=wt_levels[0]) for i in
              range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 1), n_layer=num_blocks[1], layer_id=i, wt_levels=wt_levels[1]) for i in
              range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 2), n_layer=num_blocks[2], layer_id=i, wt_levels=wt_levels[2]) for i in
              range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 3), n_layer=num_blocks[3], layer_id=i, wt_levels=wt_levels[3]) for i in
              range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=True)
        self.decoder_level3 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 2), n_layer=num_blocks[2], layer_id=i, wt_levels=wt_levels[2]) for i in
              range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=True)
        self.decoder_level2 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 1), n_layer=num_blocks[1], layer_id=i, wt_levels=wt_levels[1]) for i in
              range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 1), n_layer=num_blocks[0], layer_id=i, wt_levels=wt_levels[0]) for i in
              range(num_blocks[0])])

        self.refinement = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 1), n_layer=num_refinement_blocks, layer_id=i, wt_levels=wt_levels[0]) for i
              in
              range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.ill_depth = Depth_Illumination_Estimator(out_dim=dim)
        self.depth_conv = nn.Conv2d(int(dim), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)

        ill_depth_fea, corase_depth = self.ill_depth(inp_img)
        out_enc_level1, out_dep_fea_level1 = self.encoder_level1((inp_enc_level1, ill_depth_fea))

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_dep_fea_level2 = self.down1_2(out_dep_fea_level1)
        out_enc_level2, out_dep_fea_level2 = self.encoder_level2((inp_enc_level2, out_dep_fea_level2))

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_dep_fea_level3 = self.down2_3(out_dep_fea_level2)
        out_enc_level3, out_dep_fea_level3 = self.encoder_level3((inp_enc_level3, out_dep_fea_level3))

        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_dep_fea_level4 = self.down3_4(out_dep_fea_level3)
        latent, out_dep_fea_level4 = self.latent((inp_enc_level4, out_dep_fea_level4))

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_dep_fea_level3 = self.up4_3(out_dep_fea_level4)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, inp_dec_dep_fea_level3 = self.decoder_level3((inp_dec_level3, inp_dec_dep_fea_level3))

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_dep_fea_level2 = self.up3_2(out_dep_fea_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, inp_dec_dep_fea_level2 = self.decoder_level2((inp_dec_level2, inp_dec_dep_fea_level2))

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_dep_fea_level1 = self.depth_conv(self.up2_1(inp_dec_dep_fea_level2))
        # print(inp_dec_dep_fea_level1.shape, inp_dec_level1.shape)

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, inp_dec_dep_fea_level1 = self.decoder_level1((inp_dec_level1, inp_dec_dep_fea_level1))

        out_dec_level1, inp_dec_dep_fea_level1 = self.refinement((out_dec_level1, inp_dec_dep_fea_level1))
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1, corase_depth


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


x = torch.zeros((1, 3, 256, 256)).type(torch.FloatTensor).cuda()

model_s = ATIDNet(
        inp_channels=3,
        out_channels=3,
        dim=32,
        num_blocks=[2, 4, 4, 8],
        num_refinement_blocks=2,
        wt_levels=[3, 3, 2, 2]
    )
