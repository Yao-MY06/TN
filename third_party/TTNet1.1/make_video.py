# -*- coding: utf-8 -*-
"""用 OpenCV 将 demo 输出的帧序列合成为 mp4 视频（替代 ffmpeg）。"""
import os
import sys
import glob

import cv2


def make_video(frame_dir, output_path, fps=30):
    frames = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
    if not frames:
        print('No frames found in {}'.format(frame_dir))
        return None
    img = cv2.imread(frames[0])
    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.imread(f))
    writer.release()
    print('Saved video: {} ({} frames, {}x{})'.format(output_path, len(frames), w, h))
    return output_path


if __name__ == '__main__':
    frame_dir = sys.argv[1]
    output_path = sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    make_video(frame_dir, output_path, fps)
