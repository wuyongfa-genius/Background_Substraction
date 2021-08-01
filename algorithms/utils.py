"""Some helper functions"""
import os

import cv2
import numpy as np
import tifffile
import torch
from tqdm import tqdm
from sporco import cupy

from .GoDec import GoDec, godec_original
from .RTPCA import RTPCA

if cupy.have_cupy:
    from sporco.cupy.admm.rpca import RobustPCA as RPCA_sporco_gpu

from sporco.admm.rpca import RobustPCA as RPCA_sporco
from torch.nn import functional as F
from torch.utils.data import DataLoader
from unet_mod.datasets.mod import MOD, MOD_3d
from unet_mod.models.unet import TinyUNet, TinyUNet3d
from unet_mod.utils.window_utils import average_preds

IMG_EXTS = ['.jpg', '.png', '.jpeg', '.tif', '.tiff']
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.rmvb']


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


def capture_video_frames(video_path, flatten_frames=False, return_rgb_frames=False, **crop_kwargs):
    """Given a video path, capture the frames within. If given crop_kwargs, it also crop the frames.
    Args:
        video_path(Pathlike): video_path to process
        flatten_frames(bool): whether to flatten the grayscale frames into vectors, 
            only used when you want to cat all frames into one array.
        return_rgb_frames(bool): whether to return rgb_frames as well.
        crop_kwargs(key-word args): if you want to crop a roi out of the video, given 
            the bbox coordinates as (left=left_value, top=top_value, right=right_value, down=down_value)
    return:
        A dict contains info including grayscale_frames, height, width, video_length, fps, 
            and rgb_frames if return_rgb_frames
    """
    left = crop_kwargs.get('left', None)
    top = crop_kwargs.get('top', None)
    right = crop_kwargs.get('right', None)
    down = crop_kwargs.get('down', None)
    crop = False if left is None else True
    captured_frame_dict = {}
    frames = []
    rgb_frames = []
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print('[INFO] Unable to open: ' + video_path)
        exit(0)
    # get video info
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # update video info according to crop args
    if crop:
        frame_height = down-top
        frame_width = right-left
    print("[INFO] FPS: {}".format(fps))
    print("[INFO] Total Frames: {}".format(video_length))
    print("[INFO] Total time: {}s".format(video_length / fps))
    print("[INFO] Height: {0}, Width: {1}".format(
        frame_height, frame_width))
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        if crop:
            frame = frame[top:down, left:right]
        if return_rgb_frames:
            rgb_frames.append(frame)
        # convert to grayscale image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if flatten_frames:
            frame = frame.reshape((-1, ))
        frames.append(frame)
    capture.release()
    captured_frame_dict.update(grayscale_frames=frames, height=frame_height,
                               width=frame_width, video_length=video_length, fps=fps)
    if return_rgb_frames:
        captured_frame_dict.update(rgb_frames=rgb_frames)
    return captured_frame_dict


def capture_frames(frame_dir, flatten_frames=False, return_rgb_frames=False, **crop_kwargs):
    DEFAULT_FPS = 10
    left = crop_kwargs.get('left', None)
    top = crop_kwargs.get('top', None)
    right = crop_kwargs.get('right', None)
    down = crop_kwargs.get('down', None)
    crop = False if left is None else True
    captured_frame_dict = {}
    frames = []
    rgb_frames = []
    seqs = os.listdir(frame_dir)
    seqs = [i for i in seqs if os.path.splitext(i)[-1] in IMG_EXTS]
    seqs.sort()
    start_frame_id = int(os.path.splitext(seqs[0])[0])
    example_frame = read_img(os.path.join(frame_dir, seqs[0]))
    frame_height, frame_width = example_frame.shape[:2]
    video_length = len(seqs)
    assert video_length > 1, "video length must be > 1 !!!"
    # update video info according to crop args
    if crop:
        frame_height = down-top
        frame_width = right-left
    print("[INFO] Total Frames: {}".format(video_length))
    print("[INFO] Height: {0}, Width: {1}".format(
        frame_height, frame_width))
    for seq in tqdm(seqs):
        seqpath = os.path.join(frame_dir, seq)
        frame = read_img(seqpath)
        if crop:
            frame = frame[top:down, left:right]
        if return_rgb_frames:
            rgb_frames.append(frame)
        # convert to grayscale image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if flatten_frames:
            frame = frame.resahpe((-1, ))
        frames.append(frame)
    captured_frame_dict.update(grayscale_frames=frames, height=frame_height,
                               width=frame_width, video_length=video_length, fps=DEFAULT_FPS, start_frame_id=start_frame_id)
    if return_rgb_frames:
        captured_frame_dict.update(rgb_frames=rgb_frames)
    return captured_frame_dict


def capture_frames_from_videofile_or_framedir(path, flatten_frames=False, return_rgb_frames=True, **crop_kwargs):
    """Given a video file path or a dir which contains frames of a video, returns
    the frames(List[np.ndarray]) and other video infos(H, W, Length, FPS).
    Args:
        path(Pathlike): a video file path or a dir which contains frames of a video
        flatten_frames(bool): whether to flatten the grayscale frames into vectors, 
            only used when you want to cat all frames into one array.
        return_rgb_frames(bool): whether to return rgb_frames as well.
        crop_kwargs: if we need to crop a ROI out of the frames, we need to give the 
        ROI's bbox as left=left_value, top=top_value, right=right_value, down=down_value.
    return:
        A dict contains info including grayscale_frames, height, width, video_length, fps, 
            and rgb_frames if return_rgb_frames, and start_frame_id if capturing from raw frames.
    """
    is_video_file = False
    for exts in VIDEO_EXTS:
        if path.endswith(exts):
            is_video_file = True
            break
    if is_video_file:
        captured_frame_dict = capture_video_frames(
            path, flatten_frames=flatten_frames, return_rgb_frames=return_rgb_frames, **crop_kwargs)
    else:
        captured_frame_dict = capture_frames(
            path, flatten_frames=flatten_frames, return_rgb_frames=return_rgb_frames, **crop_kwargs)
    return captured_frame_dict


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
        bgs = GoDec(X, rank=2, max_iter=2, **kwargs)
        bg, fg = bgs()
        # bg, fg = godec_original(X, r=2, k=int(np.prod(X.shape)/50), q=0, max_iter=20)
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
    model = TinyUNet(n_channels=1, n_classes=2)
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


def unet3d_seg(fg, resume, gpu=True, model_name='TinyUNet3d', **kwargs):
    clip_length = kwargs.get('clip_length', 1)
    stride = kwargs.get('stride', 1)
    interval = kwargs.get('interval', 1)
    percent = kwargs.get('percent', None)
    start_percent = kwargs.get('start_percent', None)
    rcode = kwargs.get('rcode', None)
    task_id = kwargs.get('task_id', None)
    update_progress = percent is not None and start_percent is not None
    if update_progress:
        try:
            import TaskUtil
        except:
            raise ModuleNotFoundError
    # dataloader definition
    dataset = MOD_3d(clip_length=4, fg_imgs=fg, test_mode=True,
                     interval=interval, stride=stride)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=4, drop_last=False, pin_memory=False)
    # model
    # define model
    model = TinyUNet3d(n_channels=1, n_classes=2)
    assert resume is not None
    ckpt = torch.load(resume, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    if gpu:
        model.cuda()
    model.eval()
    # test
    preds = []
    i = 0
    max_iter = len(dataloader)
    with torch.no_grad():
        for batch in dataloader:
            if gpu:
                batch = batch.cuda()
            this_batch_result = model(batch)  # BCTHW
            preds.append(this_batch_result)
            if update_progress:
                current_percent = int(start_percent+(i+1)*percent/max_iter)
                TaskUtil.SetPercent(rcode, task_id, current_percent, '')
    preds = torch.cat(preds)[:len(dataset)]  # NCTHW
    preds = average_preds(preds, window=clip_length,
                          stride=stride, interval=interval)  # NCHW
    preds = F.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)  # NHW
    final_fgs = np.uint8(preds.cpu().numpy())
    final_fgs[final_fgs == 1] = 255
    return final_fgs


def normalize_rescale(x, rescale_min=0, rescale_max=255, out_type=np.uint8):
    min_x = np.min(x)
    max_x = np.max(x)
    normalized_x = (x-min_x)/(max_x-min_x)
    if rescale_min == 0:  # in  most case
        rescaled_x = normalized_x*rescale_max
    else:
        rescaled_x = normalized_x*(rescale_max-rescale_min)+rescale_min
    return rescaled_x.astype(out_type)
