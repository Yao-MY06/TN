# -*- coding: utf-8 -*-
"""对比两个权重在测试集上的球检测 RMSE(global / local / overall)。"""
import sys
import numpy as np
import torch

sys.path.append('./')
sys.path.append('../')

from data_process.ttnet_dataloader import create_test_dataloader
from models.model_utils import create_model, load_pretrained_model, make_data_parallel
from config.config import parse_configs
from utils.post_processing import get_prediction_ball_pos_right_test


def evaluate_ball(pretrained_path, num_batches=30):
    sys.argv = ['t', '--working-dir', '../', '--saved_fn', 't', '--gpu_idx', '0', '--batch_size', '1',
                '--pretrained_path', pretrained_path, '--smooth-labelling']
    configs = parse_configs()
    configs.distributed = False

    model = create_model(configs)
    model = make_data_parallel(model, configs)
    model = load_pretrained_model(model, pretrained_path, 0, False)
    model.eval()

    test_loader = create_test_dataloader(configs)

    w_original, h_original = 1920., 1080.
    w, h = configs.input_size

    mse_global = mse_local = mse_overall = 0.0
    cnt_global = cnt_local = cnt_overall = 0

    with torch.no_grad():
        for batch_idx, (resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg) in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
            target_seg = target_seg.to(configs.device, non_blocking=True)
            resized_imgs = resized_imgs.to(configs.device, non_blocking=True).float()
            pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, _, _ = model(
                resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg)

            org_ball_pos_xy = org_ball_pos_xy.numpy()
            global_ball_pos_xy = global_ball_pos_xy.numpy()

            for sample_idx in range(resized_imgs.size(0)):
                gx, gy = global_ball_pos_xy[sample_idx]
                sample_pred_global = (pred_ball_global[0][sample_idx], pred_ball_global[1][sample_idx])
                px, py = get_prediction_ball_pos_right_test(sample_pred_global, configs.thresh_ball_pos_mask)

                if (gx > 0) and (gy > 0) and (px > 0) and (py > 0):
                    mse_global += (px - gx) ** 2 + (py - gy) ** 2
                    cnt_global += 1

                pred_org_x = px * (w_original / w)
                pred_org_y = py * (h_original / h)

                if pred_ball_local is not None:
                    local_ball_xy = local_ball_pos_xy.cpu().numpy()
                    lx, ly = local_ball_xy[sample_idx]
                    sample_pred_local = (pred_ball_local[0][sample_idx], pred_ball_local[1][sample_idx])
                    lpx, lpy = get_prediction_ball_pos_right_test(sample_pred_local, configs.thresh_ball_pos_mask)
                    if (lx > 0) and (ly > 0):
                        mse_local += (lpx - lx) ** 2 + (lpy - ly) ** 2
                        cnt_local += 1
                        pred_org_x += lpx - w / 2
                        pred_org_y += lpy - h / 2

                ox, oy = org_ball_pos_xy[sample_idx]
                mse_overall += (ox - pred_org_x) ** 2 + (oy - pred_org_y) ** 2
                cnt_overall += 1

    rmse_global = np.sqrt(mse_global / cnt_global) if cnt_global else -1
    rmse_local = np.sqrt(mse_local / cnt_local) if cnt_local else -1
    rmse_overall = np.sqrt(mse_overall / cnt_overall) if cnt_overall else -1
    return rmse_global, rmse_local, rmse_overall, cnt_global, cnt_local, cnt_overall


if __name__ == '__main__':
    models = {
        '自训练(3rd_phase)': '../checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_30.pth',
        '预训练(ttnet.pth)': '../checkpoints/ttnet_3rd_phase/ttnet.pth',
    }
    for name, path in models.items():
        rg, rl, ro, cg, cl, co = evaluate_ball(path)
        print('{}: RMSE global={:.1f}px(320x128尺度, {}样本), local={:.1f}px({}), overall={:.1f}px(1920x1080, {})'.format(
            name, rg, cg, rl, cl, ro, co))
