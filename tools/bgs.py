"""Script to perform Background substraction in videos with custom algorithms(GoDec, Robust PCA, etc).
"""
import os
from argparse import ArgumentParser

import cv2
import time
import numpy as np
import torch

from algorithms.utils import (mk_save_dir, capture_frames_from_videofile_or_framedir, background_substraction, unet_seg,
                              unet3d_seg, normalize_rescale)
from algorithms import __bgs__ as BGS_ALGORITHMS


def add_args():
    parser = ArgumentParser()
    parser.add_argument('--video_path',
                        default=None,
                        help="Path to the video to be processed.")
    parser.add_argument('--frame_dir',
                        default=None,
                        help="Dir of the raw video frames.")
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=False,
                        help="Whether to use gpu.")
    parser.add_argument('--resume', help="checkpoint used to test.")
    parser.add_argument(
        '--save_dir',
        default=None,
        help="The results will be under the original dir if save_dir is None!")
    parser.add_argument('--algorithm',
                        default='GoDec',
                        choices=BGS_ALGORITHMS,
                        help="Algorithm to perform BGS.")
    parser.add_argument('--seg',
                        default='unet3d',
                        choices=['original', 'filt_thresh', 'unet', 'unet3d'],
                        help='How to segment foreground objects.')
    parser.add_argument('--thresh',
                        default=128,
                        type=int,
                        help='Threshhold to segment foreground objects.')
    return parser.parse_args()


def main():
    args = add_args()

    # store frames in a array
    frames = []
    start_frame_id = 1
    frame_height, frame_width = 0, 0
    video_length = 1
    save_dir = mk_save_dir(args.algorithm, args.video_path,
                           args.frame_dir, args.save_dir)
    # read in video and process
    captured_frame_dict = capture_frames_from_videofile_or_framedir(
        args.frame_dir or args.video_path, return_rgb_frames=False, flatten_frames=True)
    frames = captured_frame_dict['grayscale_frames']
    frame_height = captured_frame_dict['height']
    frame_width = captured_frame_dict['width']
    start_frame_id = captured_frame_dict.get('start_frame_id', 1)
    video_length = captured_frame_dict['video_length']

    print("[INFO] All frames have been loaded !!!")
    print("[INFO] Start processing...")
    tik = time.time()
    # stack frames into a single array
    X = np.stack(frames, axis=-1)  # F,N
    # flatten frames if ther haven't been flattened.
    if len(X.shape) > 2:
        X = X.reshape((-1, X.shape[-1]))
    # normalize
    x_mean = np.mean(X)
    x_std = np.std(X)
    X = (X-x_mean)/x_std
    # background_substraction
    fg, bg = background_substraction(X, args.algorithm, args.use_gpu)
    # rearrange back to image.
    assert bg.shape == fg.shape == X.shape
    bg = np.transpose(bg)  # N,F
    bg = bg*x_std+x_mean
    fg = np.transpose(fg)  # N,F
    fg = fg*x_std+x_mean
    # min max
    fg = normalize_rescale(fg)
    bg = normalize_rescale(bg)
    bg = bg.reshape(video_length, frame_height, frame_width)
    fg = fg.reshape(video_length, frame_height, frame_width)
    # end of decomposition
    toc = time.time()
    print(f"[INFO] Time Elapsed: {(toc-tik):.3f}s")
    # segment foreground objects
    if args.seg == 'filt_thresh':
        # median blur
        fg = [cv2.medianBlur(f, ksize=3) for f in fg]
        fg = np.stack(fg)
        # thresh
        fg_mask = np.zeros_like(fg, dtype=np.uint8)
        fg_mask[fg > args.thresh] = 255
    elif args.seg == 'unet':
        assert args.resume is not None
        fg_mask = unet_seg(fg, resume=args.resume,
                           gpu=torch.cuda.is_available())
    elif args.seg == 'unet3d':
        assert args.resume is not None
        fg_mask = unet3d_seg(fg, resume=args.resume,
                             gpu=torch.cuda.is_available(), clip_length=4, stride=1, interval=4)
    elif args.seg == 'original':
        fg_mask = fg
    # create save dirs
    print("[INFO] Start to write segmentation results...")
    fg_save_dir = os.path.join(save_dir, 'fg')
    os.makedirs(fg_save_dir, exist_ok=True)
    bg_save_dir = os.path.join(save_dir, 'bg')
    os.makedirs(bg_save_dir, exist_ok=True)
    # write results
    for i in range(video_length):
        frame_id = start_frame_id+i
        new_fg_i = fg_mask[i]
        bg_i = bg[i]
        cv2.imwrite(os.path.join(
            fg_save_dir, f'fg_{frame_id:06}.png'), new_fg_i)
        cv2.imwrite(os.path.join(bg_save_dir, f'bg_{frame_id:06}.png'), bg_i)
    print(
        f"[INFO] Foreground Masks and Background estimations have been saved at {save_dir} !!!"
    )


if __name__ == "__main__":
    main()
