import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def bgr2hsv(bgr_image):
    bgr_image = bgr_image.float() / 255.0
    b, g, r = bgr_image[:, 0, :, :], bgr_image[:, 1, :, :], bgr_image[:, 2, :, :]
    v = torch.max(b, torch.max(g, r))
    min_val = torch.min(b, torch.min(g, r))
    s = (v - min_val) / (v + 1e-6)
    h = torch.zeros_like(v)
    mask = s > 0
    h[mask & (v == r)] = 60 * (
            (g[mask & (v == r)] - b[mask & (v == r)]) / (v[mask & (v == r)] - min_val[mask & (v == r)])) % 360
    h[mask & (v == g)] = 60 * (
            (b[mask & (v == g)] - r[mask & (v == g)]) / (v[mask & (v == g)] - min_val[mask & (v == g)])) + 120
    h[mask & (v == b)] = 60 * (
            (r[mask & (v == b)] - g[mask & (v == b)]) / (v[mask & (v == b)] - min_val[mask & (v == b)])) + 240
    hsv_image = torch.stack((h, s, v), dim=1)
    return hsv_image

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        return loss


class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation
        self.mlpA = nn.Sequential(nn.Conv2d(512, 512, 1),
                                  nn.GELU(),
                                  nn.Conv2d(512, 512, 1)).cuda()
        self.mlpP = nn.Sequential(nn.Conv2d(512, 512, 1),
                                  nn.GELU(),
                                  nn.Conv2d(512, 512, 1)).cuda()
        self.mlpN = nn.Sequential(nn.Conv2d(512, 512, 1),
                                  nn.GELU(),
                                  nn.Conv2d(512, 512, 1)).cuda()

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            if i != 4:
                a, p, n = a_vgg[i], p_vgg[i], n_vgg[i]
            else:
                a, p, n = self.mlpA(a_vgg[i]), self.mlpP(p_vgg[i]), self.mlpN(n_vgg[i])
            d_ap = self.l1(a, p.detach())
            if not self.ab:
                d_an = self.l1(a, n.detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss
