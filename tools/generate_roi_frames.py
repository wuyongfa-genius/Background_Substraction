"""Crop ROI raw frames from a given video or original raw frames."""
import os
from argparse import ArgumentParser

import cv2
from algorithms import IMG_EXTS, get_parent_dir, read_img
from tqdm import tqdm


def argument():
    parser = ArgumentParser()
    parser.add_argument('--video_path',
                        default=None,
                        help="Path to the video to be processed.")
    parser.add_argument('--frame_dir',
                        default=None,
                        help="Dir of the raw video frames.")
    parser.add_argument(
        '--roi',
        default='0,0,1,1',
        required=True,
        help="ROI to crop, given in format `left, top, right, down`")
    parser.add_argument(
        '--frame_range',
        default='0,1',
        required=True,
        help="frame_range to crop, given in format `start, end`")
    parser.add_argument(
        '--save_dir',
        default=None,
        help="Dir to save the results, if not specified, create one.")
    return parser.parse_args()


def main():
    args = argument()
    # roi
    roi_str = args.roi
    left, top, right, down = [int(i) for i in roi_str.split(',')]
    # frame_range
    frame_rang_str = args.frame_range
    start, end = [int(i) for i in frame_rang_str.split(',')]
    # if input is a video.
    if args.video_path is not None:
        assert args.frame_dir is None
        # save_dir
        if args.save_dir is None:
            video_name = os.path.splitext(
                (os.path.basename(args.video_path)))[0]
            save_dir = os.path.join(
                get_parent_dir(args.video_path),
                f'{video_name}_cropped_frames/roi_{left}_{top}_{right}_{down}_frame_{start}_{end}'
            )
            os.makedirs(save_dir, exist_ok=True)
        else:
            os.makedirs(args.save_dir, exist_ok=True)
        capture = cv2.VideoCapture(args.video_path)
        if not capture.isOpened():
            print('[INFO] Unable to open: ' + args.video_path)
            exit(0)
        # 获取视频信息
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_all = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        assert start >= 1 and end <= frame_all, "Invalid frame range!!!"
        assert left >= 0 and left < right and top >= 0 and top < down and\
            right <= frame_width and down <= frame_height
        print("[INFO] 视频FPS: {}".format(fps))
        print("[INFO] 视频总帧数: {}".format(frame_all))
        print("[INFO] 视频时长: {}s".format(frame_all / fps))
        print("[INFO] 视频高度: {0}，宽度: {1}".format(frame_height, frame_width))
        bar = tqdm(total=end - start + 1)
        # crop
        print("[INFO] Starting Cropping...")
        while True:
            ret, frame = capture.read()
            frame_id = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if frame is None:
                break
            if frame_id < start or frame_id > end:
                continue
            cropped_frame = frame[top:down, left:right]
            cv2.imwrite(os.path.join(save_dir, f'{frame_id:06}.jpg'),
                        cropped_frame)
            bar.update()
        bar.close()
        capture.release()
        print(f"[INFO] Done, cropped raw frames have been save at {save_dir}")
    else:
        assert args.video_path is None
        seqs = os.listdir(args.frame_dir)
        # save_dir
        if args.save_dir is None:
            video_name = args.frame_dir.rstrip('/').split('/')[-1]
            save_dir = os.path.join(
                get_parent_dir(args.frame_dir.rstrip('/')),
                f'{video_name}_cropped_frames/roi_{left}_{top}_{right}_{down}_frame_{start}_{end}'
            )
            os.makedirs(save_dir, exist_ok=True)
        else:
            os.makedirs(args.save_dir, exist_ok=True)
        seqs = [i for i in seqs if os.path.splitext(i)[-1] in IMG_EXTS]
        seqs.sort()
        # 获取视频信息
        assert len(seqs) > 1, "video length must be > 1 !!!"
        assert start >= 1 and end <= len(seqs), "Invalid frame range!!!"
        example_frame = read_img(os.path.join(args.frame_dir, seqs[0]))
        frame_height, frame_width = example_frame.shape[:2]
        assert left >= 0 and left < right and top >= 0 and top < down and\
            right <= frame_width and down <= frame_height
        print("[INFO] 视频总帧数: {}".format(len(seqs)))
        print("[INFO] 视频高度: {0}，宽度: {1}".format(frame_height, frame_width))       
        # crop
        print("[INFO] Starting Cropping...")
        bar = tqdm(total=end - start + 1)
        for seq in seqs:
            seqpath = os.path.join(args.frame_dir, seq)
            frame_id = int(os.path.splitext(seq)[0])
            if frame_id < start or frame_id > end:
                continue
            frame = read_img(seqpath)
            cropped_frame = frame[top:down, left:right]
            cv2.imwrite(os.path.join(save_dir, f'{frame_id:06}.jpg'),
                        cropped_frame)
            bar.update()
        bar.close()
        print(f"[INFO] Done, cropped raw frames have been save at {save_dir}")


if __name__ == "__main__":
    main()
