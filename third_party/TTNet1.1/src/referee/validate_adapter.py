# -*- coding: utf-8 -*-
"""验证 TTNet 适配器输出的坐标是否与官方标注真值一致。

只在 test_2 上做抽样验证（该视频在数据集里有 ball_markup.json 逐帧标注）。
这是确认「适配器坐标还原正确」的最强证据。
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.ttnet_adapter import TTNetAdapter, W_RATIO, H_RATIO, W_IN, H_IN
from referee.run_referee import build_configs, DEFAULT_WEIGHTS


def main(game='test_2', weights=DEFAULT_WEIGHTS, max_frames=1500, device='cuda:0'):
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    ann_path = os.path.join(root, 'dataset', 'test', 'annotations', game, 'ball_markup.json')
    video = os.path.join(root, 'dataset', 'test', 'videos', '{}.mp4'.format(game))

    with open(ann_path) as f:
        ball = json.load(f)

    configs = build_configs()
    adapter = TTNetAdapter(configs, weights, device=device)

    errs = []
    errs_global_only = []
    n = 0
    for obs in adapter.observations_from_video(video, max_frames=max_frames, progress_every=500):
        gt = ball.get(str(obs.frame_id))
        if gt is None:
            continue
        gx, gy = float(gt['x']), float(gt['y'])
        if gx < 0 or gy < 0:
            continue   # 标注为「无球」
        n += 1

        if obs.ball.visible:
            px, py = obs.ball.center_px
            errs.append(np.hypot(px - gx, py - gy))
        else:
            errs.append(np.nan)   # 漏检单独统计

    errs = np.array(errs)
    valid = errs[~np.isnan(errs)]
    miss = int(np.isnan(errs).sum())

    print()
    print('=' * 66)
    print('适配器坐标验证 — {} ({} 个有标注帧)'.format(game, n))
    print('=' * 66)
    print('  漏检帧数      : {} ({:.1%})'.format(miss, miss / max(n, 1)))
    if len(valid):
        print('  定位误差 RMSE : {:.2f} px'.format(float(np.sqrt((valid ** 2).mean()))))
        print('  定位误差 中位 : {:.2f} px'.format(float(np.median(valid))))
        print('  定位误差 P90  : {:.2f} px'.format(float(np.percentile(valid, 90))))
        print('  定位误差 最大 : {:.2f} px'.format(float(valid.max())))
        print('  <25px 命中率  : {:.2%}'.format(float((valid < 25).mean())))
    print('=' * 66)


if __name__ == '__main__':
    main()
