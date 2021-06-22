import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomApply
from torchvision.transforms.transforms import Normalize
from unet_mod.datasets.transforms import (Compose, RandomAffine, RandomHorizontalFlip,
                         RandomVerticalFlip, ColorJitter, LabelMap, GaussianBlur)


class MOD(Dataset):
    def __init__(self, root=None, label_map={0:0, 255:1}, fg_imgs=None, annfile=None, val=False, test_mode=False):
        self.root = root
        self.label_map = label_map
        self.num_classes = len(list(label_map.keys()))
        self.test_mode = test_mode
        self.val = val
        if not test_mode:
            assert fg_imgs is None
            assert annfile is not None
            with open(annfile, 'r') as f:
                lines = f.readlines()
            self.imgpaths = [l.rstrip('\n') for l in lines]
            # transforms
            self.label_map_transform = LabelMap(label_map)
            if not val:
                self.spatial_transforms = Compose([
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10, fillcolor=103)
                ])
                self.color_transforms = Compose([
                    RandomApply([ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),
                    RandomApply([GaussianBlur(sigma=[.1, 1.])], p=0.2)
                ])
        ## test 
        else:
            assert fg_imgs is not None
            assert annfile is None
            self.imgs = fg_imgs # the results of robust PCA
        ## mutual transform
        self.to_tensor = Compose([
            ToTensor(),
            # Normalize(mean=[0.4026], std=[0.4025])
        ])
    
    def __getitem__(self, index):
        if not self.test_mode:
            imgpath = os.path.join(self.root, self.imgpaths[index])
            img = Image.open(imgpath)
            label = Image.open(imgpath.replace('fg', 'gt').replace('.png', '_mask.png').replace('gt_', 'fg_'))
            # map label in order that it can be transformed
            mask = self.label_map_transform(label)
            if not self.val:
                img, mask = self.spatial_transforms(img=img, mask=mask)
                img = self.color_transforms(img=img)
            return self.to_tensor(img=img), torch.from_numpy(np.array(mask)).long()
        else:
            return self.to_tensor(img=self.imgs[index])
    
    def __len__(self):
        if not self.test_mode:
            return len(self.imgpaths)
        else:
            return len(self.imgs)


# if __name__=="__main__":
#     import numpy as np
#     import random
#     dataset = MOD(root='/data/datasets/mod_dataset', annfile='/data/datasets/mod_dataset/train_list.txt', val=False)
#     index = random.randint(0, len(dataset)-1)
#     img, mask = dataset[index]
#     img = (img*255).numpy().astype(np.uint8)
#     img = Image.fromarray(img.squeeze())
#     mask[mask==255] = 128
#     mask[mask==1] = 255
#     mask = mask.numpy().astype(np.uint8)
#     mask = Image.fromarray(mask)
#     img.show()
#     mask.show()
