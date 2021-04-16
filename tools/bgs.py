"""Script to perform Background substraction in videos with custom algorithms(GoDec, Robust PCA, etc).
"""
import os
from argparse import ArgumentParser

import cv2
import torch
import numpy as np
from algorithms import IMG_EXTS
from algorithms import __all__ as BGS_ALGORITHMS
from algorithms import get_parent_dir, read_img


def argument():
    parser = ArgumentParser()
    parser.add_argument('--video_path',
                        default=None,
                        help="Path to the video to be processed.")
    parser.add_argument('--frame_dir',
                        default=None,
                        help="Dir of the raw video frames.")
    parser.add_argument(
        '--save_dir',
        default=None,
        help="The results will be under the original dir if save_dir is None!")
    parser.add_argument('--algorithm',
                        default='RTPCA',
                        choices=BGS_ALGORITHMS,
                        help="Algorithm to perform BGS.")
    return parser.parse_args()


def main():
    args = argument()
    # initialize bgs alogorithm
    if args.algorithm == 'RTPCA':
        from algorithms import RTPCA
    elif args.algorithm == 'RPCA':
        from algorithms import RPCA
    elif args.algorithm == 'RPCA_gpu':
        from algorithms import RPCA_gpu
    elif args.algorithm == 'RPCA_sporco':
        from algorithms import RPCA_sporco
    elif args.algorithm == 'GoDec':
        from algorithms import GoDec
    # store frames in a array
    save_dir = None
    frames = []
    start_frame_id = 1
    frame_height, frame_width = 0, 0
    ## read in video an d process
    if args.video_path is not None:
        assert args.frame_dir is None
        # save_dir
        if args.save_dir is None:
            video_name = os.path.splitext(
                (os.path.basename(args.video_path)))[0]
            save_dir = os.path.join(get_parent_dir(args.video_path),
                                    f'{video_name}_{args.algorithm}_results')
            os.makedirs(save_dir)
        else:
            os.makedirs(args.save_dir)
        capture = cv2.VideoCapture(args.video_path)
        if not capture.isOpened():
            print('[INFO] Unable to open: ' + args.video_path)
            exit(0)
        # 获取视频fps
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_all = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("[INFO] 视频FPS: {}".format(fps))
        print("[INFO] 视频总帧数: {}".format(frame_all))
        print("[INFO] 视频时长: {}s".format(frame_all / fps))
        print("[INFO] 视频高度: {0}，宽度: {1}".format(frame_height, frame_width))
        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.resahpe((-1, ))
            frames.append(frame)  # convert to grayscale image
        capture.release()

    else:
        assert args.video_path is None
        assert os.path.exists(args.frame_dir)
        # save_dir
        if args.save_dir is None:
            video_name = args.frame_dir.rstrip('/').split('/')[-1]
            save_dir = os.path.join(get_parent_dir((args.frame_dir).rstrip('/')),
                                    f'{video_name}_{args.algorithm}_results')
            os.makedirs(save_dir)
        else:
            os.makedirs(args.save_dir)
        seqs = os.listdir(args.frame_dir)
        seqs = [i for i in seqs if os.path.splitext(i)[-1] in IMG_EXTS]
        seqs.sort()
        start_frame_id = int(os.path.splitext(seqs[0])[0])
        example_frame = read_img(os.path.join(args.frame_dir, seqs[0]))
        frame_height, frame_width = example_frame.shape[:2]
        assert len(seqs) > 1, "video length must be > 1 !!!"
        print("[INFO] 视频总帧数: {}".format(len(seqs)))
        print("[INFO] 视频高度: {0}，宽度: {1}".format(frame_height, frame_width))
        for seq in seqs:
            seqpath = os.path.join(args.frame_dir, seq)
            frame = read_img(seqpath)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.reshape((-1, ))
            frames.append(frame)  # convert to grayscale image
    print("[INFO] All frames have been loaded !!!")
    print("[INFO] Start processing...")
    # stack frames into a single array
    X = np.stack(frames, axis=-1)  ## F,N
    # normalize
    X = X/255.
    bg = None  # background
    fg = None  # foreground
    if args.algorithm == 'RTPCA':
        bgs = RTPCA(reg_E=0.001, n_iter_max=100)
        bg, fg = bgs(X)
    elif args.algorithm == 'RPCA':
        bgs = RPCA(X)
        bg, fg = bgs()
    elif args.algorithm == 'RPCA_sporco':
        bgs = RPCA_sporco(X)
        bg, fg = bgs.solve()
    elif args.algorithm == 'RPCA_gpu':
        X = torch.tensor(X)
        bgs = RPCA_gpu(X.cuda())
        bg, fg = bgs()
        bg = bg.cpu().numpy()
        fg = fg.cpu().numpy()
    elif args.algorithm == 'GoDec':
        bgs = GoDec(X, rank=2)
        bg, fg = bgs()
    else:
        NotImplementedError
    ## rearrange back to image.
    assert bg.shape == fg.shape == X.shape
    bg = np.transpose(bg) # N,F
    bg = bg*255
    fg = np.transpose(fg) # N,F
    fg = fg*255
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    fg = np.clip(fg, 0, 255).astype(np.uint8)
    ## create save dirs
    video_length = len(bg)
    save_dir = save_dir if args.save_dir is None else args.save_dir
    fg_save_dir = os.path.join(save_dir, 'fg')
    os.makedirs(fg_save_dir, exist_ok=True)
    bg_save_dir = os.path.join(save_dir, 'bg')
    os.makedirs(bg_save_dir, exist_ok=True)

    print(f"[INFO] Start to write results...")
    for i in range(video_length):
        frame_id = start_frame_id + i
        fg_i = fg[i].reshape(frame_height, frame_width)
        bg_i = bg[i].reshape(frame_height, frame_width)
        cv2.imwrite(os.path.join(fg_save_dir, f'fg_{frame_id:06}.png'), fg_i)
        cv2.imwrite(os.path.join(bg_save_dir, f'bg_{frame_id:06}.png'), bg_i)
    print(
        f"[INFO] Foreground Masks and Background estimations have been saved at {save_dir} !!!"
    )


if __name__ == "__main__":
    main()
