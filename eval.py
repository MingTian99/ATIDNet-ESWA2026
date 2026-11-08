import os
import math
import numpy as np
import cv2
import glob
import lpips
import argparse
from concurrent.futures import ThreadPoolExecutor
import scipy.ndimage

# Argument parsing
parser = argparse.ArgumentParser(description='PSNR SSIM LPIPS', add_help=False)
parser.add_argument('--generated_images_path', default=r'',
                    help='Generated / Restored / Recovered images -path')
parser.add_argument('--target_path', default=r'',
                    help='Ground-truth -path')
parser.add_argument('--Test_Y', type=bool, default=True, help='True: test Y channel only; False: test RGB channels')
parser.add_argument('--Score_save_path', default='Score', help='save the psnr, ssim and lpips score by txt')
parser.add_argument('--init_model', default='alex', help='Initializing the model, alex or vgg')
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('--crop_border', type=int, default=4, help='same with scale')
parser.add_argument('--max_workers', type=int, default=4, help='Max number of workers for processing images')
args = parser.parse_args()

# Initialize LPIPS model once globally
loss_fn = lpips.LPIPS(net=args.init_model, version=args.version)


def main():
    test_Y = args.Test_Y
    PSNR_all = []
    SSIM_all = []
    LPIPS_all = []
    VIF_all = []
    RMSE_all = []

    img_list = sorted(glob.glob(os.path.join(args.target_path, '*')))

    print('Testing Y channel.' if test_Y else 'Testing RGB channels.')

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(process_image, img_list))

    for base_name, metrics in results:
        PSNR_all.append(metrics['PSNR'])
        SSIM_all.append(metrics['SSIM'])
        LPIPS_all.append(metrics['LPIPS'])
        VIF_all.append(metrics['VIF'])
        RMSE_all.append(metrics['RMSE'])
        print(f'{base_name}.png \tPSNR: {metrics["PSNR"]:.4f} dB, '
              f'\tSSIM: {metrics["SSIM"]:.4f},\tVIF: {metrics["VIF"]:.4f},'
              f'\tLPIPS: {metrics["LPIPS"]:.4f},\tRMSE: {metrics["RMSE"]:.4f}')

    print_mean_metrics(PSNR_all, SSIM_all, LPIPS_all, VIF_all, RMSE_all)


def process_image(img_path):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    im_GT = cv2.imread(img_path) / 255.0
    im_Gen = cv2.imread(os.path.join(args.generated_images_path, base_name + '.png')) / 255.0

    if args.Test_Y and im_GT.shape[2] == 3:
        im_GT_in = bgr2ycbcr(im_GT)
        im_Gen_in = bgr2ycbcr(im_Gen)
    else:
        im_GT_in, im_Gen_in = im_GT, im_Gen

    cropped_GT, cropped_Gen = crop_images(im_GT_in, im_Gen_in)

    PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)
    SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
    LPIPS = loss_fn.forward(lpips.im2tensor(im_Gen), lpips.im2tensor(im_GT)).item()
    VIF = vifp_mscale(cropped_GT * 255, cropped_Gen * 255)
    RMSE = rmse(cropped_GT * 255, cropped_Gen * 255)

    print(f'{base_name}.png: \n'
          f'\tPSNR: {PSNR:.4f}, \tSSIM: {SSIM:.4f}, \tVIF: {VIF:.4f}, '
          f'\tLPIPS: {LPIPS:.4f}, \tRMSE: {RMSE:.4f}')

    return base_name, {'PSNR': PSNR, 'SSIM': SSIM, 'LPIPS': LPIPS, 'VIF': VIF, 'RMSE': RMSE}


def crop_images(im_GT, im_Gen):
    if args.crop_border == 0:
        return im_GT, im_Gen
    return im_GT[args.crop_border:-args.crop_border, args.crop_border:-args.crop_border], \
           im_Gen[args.crop_border:-args.crop_border, args.crop_border:-args.crop_border]


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        return np.mean([ssim(img1[..., i], img2[..., i]) for i in range(img1.shape[2])])
    raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1.astype(np.float64), -1, window)
    mu2 = cv2.filter2D(img2.astype(np.float64), -1, window)
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1 ** 2
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2 ** 2
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1 * mu2

    return np.mean(((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) /
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)))


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def vifp_mscale(ref, dist):
    sigma_nsq = 2
    eps = 1e-10
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        sd = (2 ** (4 - scale + 1) + 1) / 5.0
        ref = scipy.ndimage.gaussian_filter(ref, sd)[::2, ::2] if scale > 1 else ref
        dist = scipy.ndimage.gaussian_filter(dist, sd)[::2, ::2] if scale > 1 else dist
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        sigma1_sq = scipy.ndimage.gaussian_filter(ref ** 2, sd) - mu1 ** 2
        sigma2_sq = scipy.ndimage.gaussian_filter(dist ** 2, sd) - mu2 ** 2
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1 * mu2

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    return num / den if den > 0 else 1.0


def rmse(org_img: np.ndarray, pred_img: np.ndarray) -> float:
    return np.sqrt(np.mean((org_img - pred_img) ** 2))


def print_mean_metrics(PSNR_all, SSIM_all, LPIPS_all, VIF_all, RMSE_all):
    print(f'Mean_PSNR: {np.mean(PSNR_all):.4f}, Mean_SSIM: {np.mean(SSIM_all):.4f}, '
          f'Mean_VIF: {np.mean(VIF_all):.4f}, Mean_LPIPS: {np.mean(LPIPS_all):.4f}, '
          f'Mean_RMSE: {np.mean(RMSE_all):.4f}')


if __name__ == '__main__':
    main()

