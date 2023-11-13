import random
import warnings

import kornia
import numpy as np
import torch
from einops import repeat
from torch import nn, Tensor
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def strong_transform(param, data):
    data = color_jitter(
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    if param['blur'] == 1:
        data = gaussian_blur(data)
    return data


def denorm(img, mean, std):
    return img.mul(std).add(mean)


def renorm(img, mean, std):
    return img.sub(mean).div(std)


def color_jitter(mean, std, data, s=.25, p=.2):
    # s is the strength of colorjitter
    color_jitter = random.uniform(0, 1)
    if color_jitter > (1-p):
        mean = torch.as_tensor(mean, device=data.device)
        mean = repeat(mean, 'C -> B C 1 1', B=data.shape[0], C=3)
        std = torch.as_tensor(std, device=data.device)
        std = repeat(std, 'C -> B C 1 1', B=data.shape[0], C=3)
        if isinstance(s, dict):
            seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
        else:
            seq = nn.Sequential(
                kornia.augmentation.ColorJitter(
                    brightness=s, contrast=s, saturation=s, hue=s))
        data = denorm(data, mean, std)
        data = seq(data)
        data = renorm(data, mean, std)
    return data


def gaussian_blur(data):
    sigma = np.random.uniform(0.15, 1.15)
    kernel_size_y = int(
        np.floor(
            np.ceil(0.1 * data.shape[2]) - 0.5 +
            np.ceil(0.1 * data.shape[2]) % 2))
    kernel_size_x = int(
        np.floor(
            np.ceil(0.1 * data.shape[3]) - 0.5 +
            np.ceil(0.1 * data.shape[3]) % 2))
    kernel_size = (kernel_size_y, kernel_size_x)
    seq = nn.Sequential(
        kornia.filters.GaussianBlur2d(
            kernel_size=kernel_size, sigma=(sigma, sigma)))
    data = seq(data)
    return data

class StrongAugmentation(object):
    """
    强数据增强, 和mask保持一致
    """

    def __init__(self, color_jitter_s, color_jitter_p, blur, mean, std):
        super().__init__()
        if color_jitter_p > 0:
            print('[Masking] Use color augmentation.')
        self.augmentation_params = {
            # 'color_jitter': random.uniform(0, 1),
            'color_jitter_s': color_jitter_s,
            'color_jitter_p': color_jitter_p,
            # 'blur': random.uniform(0, 1) if blur else 0,
            'blur': 1 if blur else 0,
            'mean': mean,
            'std': std
        }

    def __call__(self, img):
        with_batch = True
        if len(img.shape) == 3:
            img = img.clone().unsqueeze(0)
            with_batch = False
        if self.augmentation_params is not None:
            # strong_transform是按batch操作，而transform是按单个图像操作，维度要变一下
            img = strong_transform(self.augmentation_params, data=img.clone())
        if not with_batch:
            img = img[0]
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    {}'.format(self.augmentation_params)
        format_string += '\n)'
        return format_string

class Masking(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean, std):
        super(Masking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.strong_transform = StrongAugmentation(color_jitter_s, color_jitter_p, blur, mean, std)

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        img = self.strong_transform(img.clone())

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        masked_img = img * input_mask

        return masked_img
    
    def update_ratio(self, new_ratio):
        self.ratio = new_ratio
