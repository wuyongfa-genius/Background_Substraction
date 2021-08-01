import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms as TF
from torchvision.transforms import functional as F

__all__ = ['Compose', 'RandomAffine', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'ColorJitter', 'LabelMap', 'GaussianBlur', 'VideoToTensor']

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
            img (PIL.Image | Tensor | List[PIL.Image|Tensor]): Image(s) to be transformed.

        Returns:
            Affine transformed image(s) and mask(s).
        """
        assert (isinstance(img, Image.Image) and isinstance(mask, Image.Image))\
            or (isinstance(img, list) and isinstance(mask, list) and len(img)==len(mask))
        imgsize = img.size if isinstance(img, Image.Image) else img[0].size
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, imgsize)
        if isinstance(img, list):
            imgs = []
            masks = []
            for i, m in zip(img, mask):
                imgs.append(F.affine(i, *ret, resample=self.resample, fillcolor=self.fillcolor))
                masks.append(F.affine(m, *ret, resample=Image.NEAREST, fillcolor=255))
            return imgs, masks
        else:
            img = F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
            mask = F.affine(mask, *ret, resample=Image.NEAREST, fillcolor=255)
            return img, mask


class RandomHorizontalFlip(TF.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)
    
    def forward(self, img, mask):
        assert (isinstance(img, Image.Image) and isinstance(mask, Image.Image))\
            or (isinstance(img, list) and isinstance(mask, list) and len(img)==len(mask))
        if torch.rand(1) < self.p:
            if isinstance(img, list):
                img = [F.hflip(i) for i in img]
                mask = [F.hflip(m) for m in mask]
            else:
                img = F.hflip(img)
                mask = F.hflip(mask)
        return img, mask


class RandomVerticalFlip(TF.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)
    
    def forward(self, img, mask):
        assert (isinstance(img, Image.Image) and isinstance(mask, Image.Image))\
            or (isinstance(img, list) and isinstance(mask, list) and len(img)==len(mask))
        if torch.rand(1) < self.p:
            if isinstance(img, list):
                img = [F.vflip(i) for i in img]
                mask = [F.vflip(m) for m in mask]
            else:
                img = F.vflip(img)
                mask = F.vflip(mask)
        return img, mask


class ColorJitter(TF.ColorJitter):
    def __init__(self, brightness, contrast, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, img):
        if isinstance(img, list):
            return [super(ColorJitter, self).forward(i) for i in img]
        return super(ColorJitter, self).forward(img)


class LabelMap(object):
    """Given a label map dict, map the label."""
    def __init__(self, label_map:dict):
        self.label_map = label_map
    
    def map_single_mask(self, mask):
        old_mask = np.array(mask)
        if len(old_mask.shape)==3:
            old_mask = old_mask[:,:,0]
        new_mask = np.zeros_like(old_mask)
        for k,v in self.label_map.items():
            new_mask[old_mask==k] = v
        return Image.fromarray(new_mask)
    
    def __call__(self, mask):
        if isinstance(mask, list):
            return [self.map_single_mask(m) for m in mask]
        return self.map_single_mask(mask)


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def blur_single_img(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __call__(self, img):
        if isinstance(img, list):
            return [self.blur_single_img(i) for i in img]
        return self.blur_single_img(img)


class VideoToTensor(TF.ToTensor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, img, mask=None):
        if isinstance(img , (list, np.ndarray)):
            img = [super(VideoToTensor, self).__call__(i) for i in img]
            if mask is not None:
                mask = [torch.from_numpy(np.array(m)).long() for m in mask]
        else:
            img = super(VideoToTensor, self).__call__(img)
            if mask is not None:
                mask = torch.from_numpy(np.array(mask)).long()
        if mask is None:
            return img
        return img, mask