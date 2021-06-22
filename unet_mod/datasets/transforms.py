import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms as TF
from torchvision.transforms import functional as F

__all__ = ['Compose', 'RandomAffine', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'ColorJitter', 'LabelMap', 'GaussianBlur']

class Compose(TF.Compose):
    def __call__(self, **kwargs):
        img = kwargs['img']
        mask = kwargs.get('mask', None)
        if mask is not None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        else:
            for t in self.transforms:
                img = t(img)
            return img


class RandomAffine(TF.RandomAffine):
    def __init__(
        self, degrees, translate, scale, shear=None,
        fillcolor=None, resample=Image.BILINEAR
        ):
        super().__init__(degrees, translate=translate, scale=scale, shear=shear,
                         fillcolor=fillcolor, resample=resample)

    def __call__(self, img, mask):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        img = F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
        mask = F.affine(mask, *ret, resample=Image.NEAREST, fillcolor=255)
        return img, mask


class RandomHorizontalFlip(TF.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)
    
    def forward(self, img, mask):
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask


class RandomVerticalFlip(TF.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)
    
    def forward(self, img, mask):
        if torch.rand(1) < self.p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask


class ColorJitter(TF.ColorJitter):
    def __init__(self, brightness, contrast, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, img):
        return super().forward(img)


class LabelMap(object):
    """Given a label map dict, map the label."""
    def __init__(self, label_map:dict):
        self.label_map = label_map
    
    def __call__(self, mask):
        old_mask = np.array(mask)
        if len(old_mask.shape)==3:
            old_mask = old_mask[:,:,0]
        new_mask = np.zeros_like(old_mask)
        for k,v in self.label_map.items():
            new_mask[old_mask==k] = v
        return Image.fromarray(new_mask)


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img