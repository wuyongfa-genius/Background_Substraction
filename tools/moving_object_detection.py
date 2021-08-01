"""The main script to execute moving object detction.
MIT License

Copyright (c) 2021 wuyongfa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import torch
import numpy as np
from algorithms.utils import (
    capture_frames_from_videofile_or_framedir, normalize_rescale, unet3d_seg)
from algorithms import background_substraction
from tools.visualize_video import generate_video_using_fg_and_mask
try:
    import TaskUtil
    UPDATE_PROCESEE = True
except:
    UPDATE_PROCESEE = False

CUDA_AVAILABLE = torch.cuda.is_available()


def main(args):
    ## get arguments to be used. #####################################################
    video_path = args.rasterpath
    rcode = args.rcode
    ttype = 'change'
    # get l,t,r,d according to extents
    left, top, right, down = map(lambda x: int(x), args.extents.split('_'))
    task_id = args.task_id
    ## generate roi to be detected. #####################################################
    captured_frame_dict= capture_frames_from_videofile_or_framedir(
        video_path, flatten_frames=True, return_rgb_frames=True, left=left, top=top, right=right, down=down)
    flattend_grayscale_frames = captured_frame_dict['grayscale_frames']
    rgb_frames = captured_frame_dict['rgb_frames']
    video_height = captured_frame_dict['height']
    video_width = captured_frame_dict['width']
    video_length = captured_frame_dict['video_length']
    FPS = captured_frame_dict['fps']
    print("[INFO] All frames have been loaded !!!")
    if UPDATE_PROCESEE:
        TaskUtil.SetPercent(rcode, task_id, 4, '')
    ## background substracting #######################################################
    print("[INFO] Performing background substraction...")
    # stack frames into a single array
    X = np.stack(flattend_grayscale_frames, axis=-1)  # F,N
    # flatten frames if ther haven't been flattened.
    if len(X.shape) > 2:
        X = X.reshape((-1, X.shape[-1]))
    # normalize
    x_mean = np.mean(X)
    x_std = np.std(X)
    X = (X-x_mean)/x_std
    # background_substraction (update progress inside the function)
    process_kwargs = dict()
    if UPDATE_PROCESEE:
        process_kwargs.update(start_percent=4, percent=40, rcode=rcode, task_id=task_id)
    fg, _ = background_substraction(
        X, algorithm='GoDec', use_gpu=False, **process_kwargs)
    # rearrange back to image.
    assert fg.shape == X.shape
    fg = np.transpose(fg)  # N,F
    fg = fg*x_std+x_mean
    # min max
    fg = normalize_rescale(fg)
    fg = fg.reshape(video_length, video_height, video_width)
    print("[INFO] Background substraction done !!!")
    ## segment foregraound objects ######################################################
    print("[INFO] Start segmenting foreground objects...")
    assert args.pretrained_weight is not None
    process_kwargs.clear()
    if UPDATE_PROCESEE:
        process_kwargs.update(start_percent=4, percent=40, rcode=rcode, task_id=task_id)
    fg_mask = unet3d_seg(fg, resume=args.pretrained_weight,
                         gpu=CUDA_AVAILABLE, clip_length=4, stride=1, interval=4, **process_kwargs)
    print("[INFO] Segmentation Done...")
    ## save results to a video ###########################################################
    print("[INFO] Writing results to video...")
    video_info = dict(fps=FPS, frameSize=(
        video_width, video_height), isColor=True)
    generate_video_using_fg_and_mask(
        frames=rgb_frames, fg_mask=fg_mask, save_path=f'{rcode}_{ttype}.mp4', **video_info)
    print("[INFO] ALL DONE !!!")
    if UPDATE_PROCESEE:
        TaskUtil.SetPercent(rcode, task_id, 100, '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rasterpath',
                        default='/mst/ipiudatas/original/12/zy.avi', help='the video to be processed.')
    parser.add_argument('--rcode',  default='12',
                        help='response code',)
    parser.add_argument('--extents',
                        default='168_56_200_100', help='bbox coordinates(left_top_right_down)',)
    parser.add_argument('--task_id',
                        default='target', help='the id of the task',)
    parser.add_argument('--pretrained_weight',
                        default='/home/h410/ipiu-project/Background_Substraction/exps/unet3d/epoch_35.pth',
                        help='pretrained weight of model.')
    args = parser.parse_args()
    main(args=args)
