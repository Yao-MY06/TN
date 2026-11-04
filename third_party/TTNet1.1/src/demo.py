"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.10
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: This script for demonstration
"""

import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
import argparse

sys.path.append('./')

from data_process.ttnet_video_loader import TTNet_Video_Loader
from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs
from utils.post_processing import post_processing
from utils.misc import time_synchronized


def demo(configs):
    video_loader = TTNet_Video_Loader(configs.video_path, configs.input_size, configs.num_frames_sequence)
    result_filename = os.path.join(configs.save_demo_dir, 'results.txt')
    frame_rate = video_loader.video_fps
    stream_video = getattr(configs, 'stream_video', False) and configs.output_format == 'video'
    video_writer = None
    if configs.save_demo_output and not stream_video:
        configs.frame_dir = os.path.join(configs.save_demo_dir, 'frame')
        if not os.path.isdir(configs.frame_dir):
            os.makedirs(configs.frame_dir)


    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))
    # model
    model = create_model(configs)
    model.cuda()

    assert configs.pretrained_path is not None, "Need to load the pre-trained model"
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx, configs.overwrite_global_2_local)

    model.eval()
    middle_idx = int(configs.num_frames_sequence / 2)
    queue_frames = deque(maxlen=middle_idx + 1)
    frame_idx = 0
    max_frames = getattr(configs, 'max_frames', None)
    w_original, h_original = 1920, 1080
    w_resize, h_resize = 320, 128
    w_ratio = w_original / w_resize
    h_ratio = h_original / h_resize
    with torch.no_grad():
        for count, resized_imgs in video_loader:
            if max_frames is not None and frame_idx >= max_frames:
                break
            # take the middle one
            img = cv2.resize(resized_imgs[3 * middle_idx: 3 * (middle_idx + 1)].transpose(1, 2, 0), (w_original, h_original))
            # Expand the first dim
            resized_imgs = torch.from_numpy(resized_imgs).to(configs.device, non_blocking=True).float().unsqueeze(0)
            t1 = time_synchronized()
            pred_ball_global, pred_ball_local, pred_events, pred_seg = model.run_demo(resized_imgs)
            t2 = time_synchronized()
            prediction_global, prediction_local, prediction_seg, prediction_events = post_processing(
                pred_ball_global, pred_ball_local, pred_events, pred_seg, configs.input_size[0],
                configs.thresh_ball_pos_mask, configs.seg_thresh, configs.event_thresh)
            prediction_ball_final = [
                int(prediction_global[0] * w_ratio + prediction_local[0] - w_resize / 2),
                int(prediction_global[1] * h_ratio + prediction_local[1] - h_resize / 2)
            ]

            # Get infor of the (middle_idx + 1)th frame
            if len(queue_frames) == middle_idx + 1:
                frame_pred_infor = queue_frames.popleft()
                seg_img = frame_pred_infor['seg'].astype(np.uint8)
                ball_pos = frame_pred_infor['ball']
                seg_img = cv2.resize(seg_img, (w_original, h_original))
                ploted_img = plot_detection(img, ball_pos, seg_img, prediction_events)

                ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
                if configs.show_image:
                    cv2.imshow('ploted_img', ploted_img)
                    cv2.waitKey(10)
                if configs.save_demo_output:
                    if stream_video:
                        if video_writer is None:
                            height, width = ploted_img.shape[:2]
                            fps = min(frame_rate, 30)
                            output_video_path = os.path.join(configs.save_demo_dir, 'result.mp4')
                            video_writer = cv2.VideoWriter(
                                output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                        video_writer.write(ploted_img)
                    else:
                        cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx)), ploted_img)

            frame_pred_infor = {
                'seg': prediction_seg,
                'ball': prediction_ball_final
            }
            queue_frames.append(frame_pred_infor)

            frame_idx += 1
            print('Done frame_idx {} - time {:.3f}s'.format(frame_idx, t2 - t1))

    if configs.output_format == 'video':
        if stream_video:
            if video_writer is None:
                raise RuntimeError('No demo frames were produced')
            video_writer.release()
        else:
            output_video_path = os.path.join(configs.save_demo_dir, 'result.mp4')
            frame_files = sorted(os.listdir(configs.frame_dir))
            if frame_files:
                first = cv2.imread(os.path.join(configs.frame_dir, frame_files[0]))
                height, width = first.shape[:2]
                fps = min(frame_rate, 30)
                writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                for frame_file in frame_files:
                    frame = cv2.imread(os.path.join(configs.frame_dir, frame_file))
                    writer.write(frame)
                writer.release()
            else:
                raise RuntimeError('No demo frames were produced')


def plot_detection(img, ball_pos, seg_img, events):
    """Show the predicted information in the image"""
    img = cv2.addWeighted(img, 1., seg_img * 255, 0.3, 0)
    img = cv2.circle(img, tuple(ball_pos), 5, (255, 0, 255), -1)
    event_name = 'is bounce: {:.2f}, is net: {:.2f}'.format(events[0], events[1])
    img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--stream_video', action='store_true')
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    configs = parse_configs()
    configs.max_frames = extra.max_frames
    configs.stream_video = extra.stream_video
    demo(configs=configs)
