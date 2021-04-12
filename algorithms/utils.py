"""Some helper functions"""
import numpy as np
import tifffile
import cv2
import os

IMG_EXTS = ['.jpg', '.png', '.jpeg', '.tif', '.tiff']

def read_img(imgpath):
    ext = os.path.splitext(imgpath)[-1]
    if ext in ['.tif', '.tiff']:
        return tifffile.imread(imgpath)
    else:
        return cv2.imread(imgpath)

def get_parent_dir(path_or_dir):
    return os.path.dirname(path_or_dir)
