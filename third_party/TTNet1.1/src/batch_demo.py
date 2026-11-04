# -*- coding: utf-8 -*-
"""批量对 test_2~7 视频运行 TTNet 推理,直接合成结果视频(不保存中间帧)。"""
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch

sys.path.append('./')

from data_process.ttnet_video_loader import TTNet_Video_Loader
from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs
from utils.post_processing import post_processing
from utils.misc import time_synchronized


def plot_detection(img, ball_pos, seg_img, events):
    img = cv2.addWeighted(img, 1., seg_img * 255, 0.3, 0)
    img = cv2.circle(img, tuple(ball_pos), 5, (255, 0, 255), -1)
    event_name = 'is bounce: {:.2f}, is net: {:.2f}'.format(events[0], events[1])
    img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def process_video(model, configs, video_path, out_path):
    video_loader = TTNet_Video_Loader(video_path, configs.input_size, configs.num_frames_sequence)
    frame_rate = video_loader.video_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, frame_rate, (1920, 1080))

    middle_idx = int(configs.num_frames_sequence / 2)
    queue_frames = deque(maxlen=middle_idx + 1)
    w_original, h_original = 1920, 1080
    w_resize, h_resize = 320, 128
    w_ratio = w_original / w_resize
    h_ratio = h_original / h_resize
    device = configs.device

    with torch.no_grad():
        for count, resized_imgs in video_loader:
            img = cv2.resize(resized_imgs[3 * middle_idx: 3 * (middle_idx + 1)].transpose(1, 2, 0),
                             (w_original, h_original))
            resized_imgs = torch.from_numpy(resized_imgs).to(device, non_blocking=True).float().unsqueeze(0)
            pred_ball_global, pred_ball_local, pred_events, pred_seg = model.run_demo(resized_imgs)
            prediction_global, prediction_local, prediction_seg, prediction_events = post_processing(
                pred_ball_global, pred_ball_local, pred_events, pred_seg, configs.input_size[0],
                configs.thresh_ball_pos_mask, configs.seg_thresh, configs.event_thresh)
            prediction_ball_final = [
                int(prediction_global[0] * w_ratio + prediction_local[0] - w_resize / 2),
                int(prediction_global[1] * h_ratio + prediction_local[1] - h_resize / 2)
            ]

            if len(queue_frames) == middle_idx + 1:
                frame_pred_infor = queue_frames.popleft()
                seg_img = frame_pred_infor['seg'].astype(np.uint8)
                ball_pos = frame_pred_infor['ball']
                seg_img = cv2.resize(seg_img, (w_original, h_original))
                ploted_img = plot_detection(img, ball_pos, seg_img, prediction_events)
                ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
                writer.write(ploted_img)

            frame_pred_infor = {'seg': prediction_seg, 'ball': prediction_ball_final}
            queue_frames.append(frame_pred_infor)

    writer.release()
    print('done: {} -> {}'.format(os.path.basename(video_path), out_path))


if __name__ == '__main__':
    sys.argv = ['batch_demo.py', '--working-dir', '../', '--saved_fn', 'batch', '--arch', 'ttnet', '--gpu_idx', '0',
                '--pretrained_path', '../checkpoints/ttnet_3rd_phase/ttnet.pth', '--seg_thresh', '0.5',
                '--event_thresh', '0.5', '--thresh_ball_pos_mask', '0.05']
    configs = parse_configs()
    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    model = create_model(configs)
    model.cuda()
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx, False)
    model.eval()

    os.makedirs('../results/demo/batch', exist_ok=True)
    for i in range(2, 8):
        video_path = '../dataset/test/videos/test_{}.mp4'.format(i)
        out_path = '../results/demo/batch/test_{}.mp4'.format(i)
        if not os.path.isfile(video_path):
            print('skip (not found):', video_path)
            continue
        print('=== processing test_{}.mp4 ==='.format(i))
        process_video(model, configs, video_path, out_path)

    print('=== ALL DONE ===')
