"""Script to perform Background substraction in videos with OpneCV algorithms(KNN, MOG).
"""
import os
from tqdm import tqdm
from argparse import ArgumentParser
from algorithms import read_img, IMG_EXTS

import cv2


def argument():
    parser = ArgumentParser()
    parser.add_argument('--video_path',
                        default=None,
                        help="Path to the video to be processed.")
    parser.add_argument('--frame_dir',
                        default=None,
                        help="Dir of the raw video frames.")
    parser.add_argument('--show',
                        action='store_true',
                        help="Whether to show the background when executing.")
    parser.add_argument('--save_dir',
                        default=None,
                        help="Dir to save the results!")
    parser.add_argument('--algorithm',
                        default='MOG2',
                        choices=['MOG2', 'KNN'],
                        help="Algorithms in OpenCV to perform BGS.")
    return parser.parse_args()


def main():
    args = argument()
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    # initialize bgs alogorithm
    if args.algorithm == 'MOG2':
        bgs = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=400)
    else:
        bgs = cv2.createBackgroundSubtractorKNN(history=400,
                                                dist2Threshold=400)
    ## read in video an d process
    if args.video_path is not None:
        assert args.frame_dir is None
        capture = cv2.VideoCapture(args.video_path)
        if not capture.isOpened():
            print('[INFO] Unable to open: ' + args.video_path)
            exit(0)
        # 获取视频fps
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        # 获取视频总帧数
        frame_all = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[INFO] 视频FPS: {}".format(fps))
        print("[INFO] 视频总帧数: {}".format(frame_all))
        print("[INFO] 视频时长: {}s".format(frame_all / fps))
        bar = tqdm(total=frame_all)

        while True:
            ret, frame = capture.read()
            frame_id = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if frame is None:
                break
            fgMask = bgs.apply(frame)
            if args.show:
                cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
                cv2.putText(frame, f'{frame_id}', (15, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                cv2.imshow('Frame', frame)
                cv2.imshow('FG Mask', fgMask)
            if args.save_dir is not None:
                cv2.imwrite(
                    os.path.join(args.save_dir, f'fgmask_{frame_id}.png'),
                    fgMask)

            keyboard = cv2.waitKey(0)
            if keyboard == 'q' or keyboard == 27:
                break
            bar.update()
        bar.close()
    else:
        assert args.video_path is None
        assert os.path.exists(args.frame_dir)
        seqs = os.listdir(args.frame_dir)
        seqs = [i for i in seqs if os.path.splitext(i)[-1] in IMG_EXTS]
        seqs.sort()
        assert len(seqs) > 1, "video length must be > 1 !!!"
        print("[INFO] 视频总帧数: {}".format(len(seqs)))
        for seq in tqdm(seqs):
            seqpath = os.path.join(args.frame_dir, seq)
            frame_id = int(os.path.splitext(seq)[0])
            frame = read_img(seqpath)
            fgMask = bgs.apply(frame)
            if args.show:
                cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
                cv2.putText(frame, str(frame_id), (15, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                cv2.imshow('Frame', frame)
                cv2.imshow('FG Mask', fgMask)
            if args.save_dir is not None:
                cv2.imwrite(
                    os.path.join(args.save_dir, f'fgmask_{frame_id}.png'),
                    fgMask)

            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break
    print(f'[Info] Done!!!')
    if args.save_dir is not None:
        print(f"[Info] Foreground Masks have been saved at {args.save_dir} !!!")


if __name__ == "__main__":
    main()