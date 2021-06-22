"""Some helper functions"""
import os

import cv2
import numpy as np
import tifffile
import torch
from sporco import cupy

from .GoDec import GoDec
from .RTPCA import RTPCA

if cupy.have_cupy:
    from sporco.cupy.admm.rpca import RobustPCA as RPCA_sporco_gpu

from sporco.admm.rpca import RobustPCA as RPCA_sporco
from torch.nn import functional as F
from torch.utils.data import DataLoader
from unet_mod.datasets.mod import MOD
from unet_mod.models.unet import TinyUNet

IMG_EXTS = ['.jpg', '.png', '.jpeg', '.tif', '.tiff']


def read_img(imgpath):
    ext = os.path.splitext(imgpath)[-1]
    if ext in ['.tif', '.tiff']:
        return tifffile.imread(imgpath)
    else:
        return cv2.imread(imgpath)

def get_parent_dir(path_or_dir):
    return os.path.dirname(path_or_dir)

def mk_save_dir(algorithm, video_path=None, frame_dir=None, save_dir=None, exist_ok=True):
    new_save_dir = None or save_dir
    if video_path is not None:
        assert frame_dir is None
        # save_dir
        if save_dir is None:
            video_name = os.path.splitext(
                (os.path.basename(video_path)))[0]
            new_save_dir = os.path.join(get_parent_dir(video_path),
                                        f'{video_name}_{algorithm}_results')
    else:
        assert video_path is None
        assert os.path.exists(frame_dir)
        # save_dir
        if save_dir is None:
            video_name = frame_dir.rstrip('/').split('/')[-1]
            new_save_dir = os.path.join(get_parent_dir((frame_dir).rstrip('/')),
                                        f'{video_name}_{algorithm}_results')
    os.makedirs(new_save_dir, exist_ok=exist_ok)
    return new_save_dir

def capture_video_frames(video_path):
    frames = []
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print('[INFO] Unable to open: ' + video_path)
        exit(0)
    # get video info
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("[INFO] FPS: {}".format(fps))
    print("[INFO] Total Frames: {}".format(video_length))
    print("[INFO] Total time: {}s".format(video_length / fps))
    print("[INFO] Height: {0}, Width: {1}".format(
        frame_height, frame_width))
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.resahpe((-1, ))
        frames.append(frame)  # convert to grayscale image
    capture.release()
    return frames, frame_height, frame_width, video_length

def capture_frames(frame_dir, flatten_frame=True):
    frames = []
    seqs = os.listdir(frame_dir)
    seqs = [i for i in seqs if os.path.splitext(i)[-1] in IMG_EXTS]
    seqs.sort()
    start_frame_id = int(os.path.splitext(seqs[0])[0])
    example_frame = read_img(os.path.join(frame_dir, seqs[0]))
    frame_height, frame_width = example_frame.shape[:2]
    video_length = len(seqs)
    assert video_length > 1, "video length must be > 1 !!!"
    print("[INFO] Total Frames: {}".format(video_length))
    print("[INFO] Height: {0}, Width: {1}".format(
        frame_height, frame_width))
    for seq in seqs:
        seqpath = os.path.join(frame_dir, seq)
        frame = read_img(seqpath)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale image
        # ## filter
        # frame = cv2.GaussianBlur(frame, ksize=3, sigmaX=1, sigmaY=1)
        if not flatten_frame:
            frames.append(frame)
        else:
            frame = frame.reshape((-1, ))
            frames.append(frame) 
    return frames, frame_height, frame_width, start_frame_id, video_length

def background_substraction(X, algorithm, use_gpu=True, **kwargs):
    fg, bg = None, None
    if algorithm == 'RTPCA':
        X = torch.from_numpy(X)
        if torch.cuda.is_available() and use_gpu:
            X = X.cuda()
        bgs = RTPCA(reg_E=0.001, n_iter_max=100, backend='pytorch')  # pytorch
        bg, fg = bgs(X)
        bg, fg = bg.cpu().numpy(), fg.cpu().numpy()
    elif algorithm == 'RPCA_sporco':
        from sporco import cupy
        if cupy.have_cupy and use_gpu:
            opt = RPCA_sporco_gpu.Options(dict(Verbose=True, MaxMainIter=20))
            X = cupy.np2cp(X)
            bgs = RPCA_sporco_gpu(X, opt=opt)
            bg, fg = bgs.solve()
            bg = cupy.cp2np(bg)
            fg = cupy.cp2np(fg)
        else:
            opt = RPCA_sporco.Options(dict(Verbose=True, MaxMainIter=100))
            bgs = RPCA_sporco(X, opt=opt)
            bg, fg = bgs.solve()
    elif algorithm == 'GoDec':
        bgs = GoDec(X, rank=2, max_iter=2)
        bg, fg = bgs()
    else:
        NotImplementedError
    return fg, bg

def unet_seg(fg, resume, gpu=True, model_name='TinyUNet'):
    # make dataset
    test_dataset = MOD(fg_imgs=fg, test_mode=True)
    test_dataloader = DataLoader(
        test_dataset, 8, False, num_workers=4, pin_memory=True, drop_last=False)
    # define model
    print("[INFO] Start to segment foreground moving objects...")
    model = TinyUNet(n_channels=1, n_classes=2, bilinear=True)
    assert resume is not None
    ckpt = torch.load(resume, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    if gpu:
        model.cuda()
    model.eval()
    # test
    final_fgs = []
    for batch in test_dataloader:
        if gpu:
            batch = batch.cuda()
        with torch.no_grad():
            this_batch_result = model(batch)
            this_batch_result = F.softmax(this_batch_result, dim=1)
            this_batch_result = torch.argmax(this_batch_result, dim=1)
        final_fgs.append(this_batch_result.cpu().numpy())
    final_fgs = np.concatenate(final_fgs)
    final_fgs = np.uint8(final_fgs)
    final_fgs[final_fgs == 1] = 255
    return final_fgs

def normalize_rescale(x, rescale_min=0, rescale_max=255, out_type=np.uint8):
    min_x = np.min(x)
    max_x = np.max(x)
    normalized_x = (x-min_x)/(max_x-min_x)
    if rescale_min==0: ## in  most case
        rescaled_x = normalized_x*rescale_max
    else:
        rescaled_x = normalized_x*(rescale_max-rescale_min)+rescale_min
    return rescaled_x.astype(out_type)