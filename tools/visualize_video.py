"""Using foreground mask to generate a demo video."""
import cv2
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def put_mask_on_img(img, mask, alpha=0.5):
    assert len(img.shape)==3 and len(mask.shape)==2
    bgr_mask = np.zeros_like(img, dtype=np.uint8)
    bgr_mask[:,:,-1] = mask
    img[mask!=0] = alpha*img[mask!=0]+(1-alpha)*bgr_mask[mask!=0]
    return img

def generate_video(frame_dir, fg_dir, save_path, **video_info):
    frame_names = sorted(os.listdir(frame_dir))
    fg_names = sorted(os.listdir(fg_dir))
    ## check wheter frames and fg match
    frames = [int(f.split('.')[0]) for f in frame_names]
    fgs = [int(f.split('.')[0].strip('fg_')) for f in fg_names]
    assert frames==fgs
    example_frame = cv2.imread(os.path.join(frame_dir, frame_names[0]))
    h,w = example_frame.shape[:-1]
    video_info['frameSize'] = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if save_path is None:
        savedir = os.path.dirname(frame_dir.rstrip('/'))
        video_name = fg_dir.rstrip('/').split('/')[-2]
        save_path = os.path.join(savedir, video_name+'.avi')
    writer = cv2.VideoWriter(save_path, fourcc, **video_info)
    for frame_name,fg_name in tqdm(zip(frame_names, fg_names)):
        frame = cv2.imread(os.path.join(frame_dir, frame_name))
        fg = cv2.imread(os.path.join(fg_dir, fg_name), cv2.IMREAD_GRAYSCALE)
        image = put_mask_on_img(frame, fg)
        writer.write(image)
    writer.release()
    print(f"video has been saved at {save_path}")

def generate_video_using_fg_and_mask(frames, fg_mask, save_path, **video_info):
    """Given original frames(List[np.ndarray]) and moving object masks(np.ndarray, shape of NHW),
    generate a video which visualize the moving objects.
    Args:
        frames(List[np.ndarray]): oiginal grayscale frames
        fg_mask(np.ndarray): moving object masks
        save_path(Pathlike): path to save the video
        video_info: info used to generate the video, such as 'frameSize' which is given as tuple(w, h);
            fps, given as a int; isColor, bool.
    """
    assert save_path.endswith('.mp4'), "please save as a mp4 video."
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    assert len(frames)==len(fg_mask), "please make sure fg matches mask."
    writer = cv2.VideoWriter(save_path, fourcc, **video_info)
    for i in range(len(frames)):
        image = put_mask_on_img(frames[i], fg_mask[i])
        writer.write(image)
    writer.release()
    print(f"[INFO] video has been saved at {save_path}")

##
if __name__=="__main__":
    video_info = dict(fps=10, frameSize=(0, 0), isColor=True)
    parser = ArgumentParser()
    parser.add_argument('frame_dir', help='path to the frames')
    parser.add_argument('fg_dir', help='path to the foregrounds')
    parser.add_argument('--save_path', help='path to save the demo video.')
    args = parser.parse_args()
    generate_video(args.frame_dir, args.fg_dir, args.save_path, **video_info)

