# -*- coding: utf-8 -*-
"""验证模型结构、权重加载与 CUDA 推理是否正常。"""
import os
import sys

sys.path.append('./')
sys.path.append('../')

import torch
import numpy as np

from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs


def main():
    # 模拟 demo 参数
    sys.argv = [
        'verify.py',
        '--working-dir', '../',
        '--saved_fn', 'demo',
        '--arch', 'ttnet',
        '--gpu_idx', '0',
        '--pretrained_path', '../checkpoints/ttnet_3rd_phase/ttnet.pth',
        '--seg_thresh', '0.5',
        '--event_thresh', '0.5',
        '--thresh_ball_pos_mask', '0.05',
    ]
    configs = parse_configs()
    print('tasks:', configs.tasks)
    print('input_size:', configs.input_size)

    model = create_model(configs)
    model.cuda()
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx, configs.overwrite_global_2_local)
    model.eval()

    # 检查参数统计
    total = sum(p.numel() for p in model.parameters())
    print('total params:', total)

    # 随机输入跑一次前向,验证 CUDA 推理
    resize_batch_input = torch.rand((1, 27, 128, 320)).cuda()
    with torch.no_grad():
        pred_ball_global, pred_ball_local, pred_events, pred_seg = model.run_demo(resize_batch_input)
    print('pred_ball_global type:', type(pred_ball_global))
    print('pred_ball_global[0] shape:', pred_ball_global[0].shape, 'pred_ball_global[1] shape:', pred_ball_global[1].shape)
    print('pred_ball_local[0] shape:', pred_ball_local[0].shape)
    print('pred_events shape:', pred_events.shape)
    print('pred_seg shape:', pred_seg.shape)
    print('VERIFY OK')


if __name__ == '__main__':
    main()
