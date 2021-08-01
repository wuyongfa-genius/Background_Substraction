import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomApply
from torchvision.transforms.transforms import Normalize
from unet_mod.datasets.transforms import (Compose, RandomAffine, RandomHorizontalFlip,
                         RandomVerticalFlip, ColorJitter, LabelMap, GaussianBlur, VideoToTensor)


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


class MOD_3d(MOD):
    """The dataset to load video as clips. Note when testing, the 
        relationship of dataset length N and video_length M is 
        1+(N-1)*stride+(clip_length-1)*interval=M.
    Args:
        root(Path): the root dir of dataset
        label_map(dict): the label map between given labels and 
            the actual training labels
        clip_length(int): number of frames in one clip.
        fg_imgs(np.ndarray): only not None when test_mode is 
            True. The result of background substraction, shape of THW.
        annfile(Path): the annatation file used when training or validation
        val(bool): whether validating or not
        test_mode(bool): whether testing or not.
    """
    def __init__(self, root=None, label_map={0:0, 255:1}, clip_length=4, fg_imgs=None, annfile=None, val=False, test_mode=False, **kwargs):
        super().__init__(root=root, label_map=label_map, fg_imgs=fg_imgs, annfile=annfile, val=val, test_mode=test_mode)
        self.clip_length = clip_length
        if not self.test_mode:
            self.imgpaths = self.generate_clip_paths(annfile=annfile, val=val)
        else:
            assert fg_imgs is not None
            assert annfile is None
            self.imgs = fg_imgs # the results of robust PCA, shape of THW
            self.stride = kwargs.get('stride', 1)
            self.interval = kwargs.get('interval', 1)
        self.to_tensor = VideoToTensor()
    
    def generate_clip_paths(self, annfile, val=False):
        with open(annfile, 'r') as f:
                lines = f.readlines()
        imgpaths = [l.rstrip('\n') for l in lines]
        # divide imgpaths into num_seqs
        unique_seqs = {}
        for imgpath in imgpaths:
            seqname = imgpath.split('_')[-2]
            if  seqname not in unique_seqs:
                unique_seqs[seqname] = [imgpath]
            else:
                unique_seqs[seqname].append(imgpath)
        # get all valid clip paths
        all_clip_paths = []
        for seq in unique_seqs:
            # sort frames in each seq
            unique_seqs[seq].sort()
            frames = unique_seqs[seq]
            if not val:
                for frame_index in range(len(frames)-self.clip_length):
                    # choose clip_length from every clip_length+1
                    chosen_indices = np.random.choice(np.arange(frame_index, frame_index+self.clip_length+1), self.clip_length, replace=False)
                    chosen_indices.sort()
                    this_clip = [frames[i] for i in chosen_indices]
                    all_clip_paths.append(tuple(this_clip))
            else:
                for frame_index in range(len(frames)-self.clip_length+1):
                    this_clip = frames[frame_index:frame_index+self.clip_length]
                    all_clip_paths.append(tuple(this_clip))
        return all_clip_paths

    def __getitem__(self, index):
        if not self.test_mode:
            imgpaths = [os.path.join(self.root, self.imgpaths[index][i]) for i in range(self.clip_length)]
            imgs = [Image.open(i) for i in imgpaths]
            labels = [Image.open(i.replace('fg', 'gt').replace('.png', '_mask.png').replace('gt_', 'fg_')) for i in imgpaths]
            # map label in order that it can be transformed
            masks = self.label_map_transform(labels)
            if not self.val:
                imgs, masks = self.spatial_transforms(img=imgs, mask=masks)
                imgs = self.color_transforms(img=imgs)
            imgs, masks = self.to_tensor(img=imgs, mask=masks)
            imgs = torch.stack(imgs, dim=1) # CTHW
            masks = torch.stack(masks) # THW
            return imgs, masks
        else:
            this_clip = self.imgs[index*self.stride:index*self.stride+(self.clip_length-1)*self.interval+1:self.interval] # clip*H*W
            this_clip = self.to_tensor(this_clip)
            return torch.stack(this_clip, dim=1) # CTHW
    
    def __len__(self):
        if not self.test_mode:
            return len(self.imgpaths)
        else:
            num_sample = self.imgs.shape[0]
            return int((num_sample-1-(self.clip_length-1)*self.interval)/self.stride)+1

# if __name__=="__main__":
#     import numpy as np
#     import random
#     import cv2
#     # dataset = MOD(root='/data/datasets/mod_dataset', annfile='/data/datasets/mod_dataset/train_list.txt', val=False)
#     # index = random.randint(0, len(dataset)-1)
#     # img, mask = dataset[index]
#     # img = (img*255).numpy().astype(np.uint8)
#     # img = Image.fromarray(img.squeeze())
#     # mask[mask==255] = 128
#     # mask[mask==1] = 255
#     # mask = mask.numpy().astype(np.uint8)
#     # mask = Image.fromarray(mask)
#     # img.show()
#     # mask.show()
    
#     ## 3d 
#     dataset = MOD_3d(root='/data/datasets/mod_dataset', annfile='/data/datasets/mod_dataset/train_list.txt', val=False)
#     index = random.randint(0, len(dataset)-1)
#     imgs, masks = dataset[index]
#     for img, mask in zip(imgs, masks):
#         img = (img*255).numpy().astype(np.uint8).squeeze()
#         mask[mask==255] = 128
#         mask[mask==1] = 255
#         mask = mask.numpy().astype(np.uint8)
#         print("img.shape", img.shape)
#         print("mask.shape", mask.shape)
#         cv2.imshow("img_mask", np.concatenate((img, mask), -1))
#         cv2.waitKey(0)
    # imgs = np.random.randint(0, 256, (320, 256, 256))
    # dataset = MOD_3d(clip_length=4, fg_imgs=imgs, test_mode=True, interval=4)
    # print(len(dataset))
    # print(dataset[0].shape)