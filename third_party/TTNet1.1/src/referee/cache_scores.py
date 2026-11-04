# -*- coding: utf-8 -*-
"""缓存逐帧事件分数与几何，供离线参数扫描（跑一次推理，反复调参）。

用法::

    python -m referee.cache_scores --game test_2 --out ../results/cache_test_2.json

产出的缓存包含每帧的 bounce/net 分数、球可见性、落点、球台框，
之后 evaluate_events 的 --from-cache 模式可以在秒级完成阈值实验。
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.ttnet_adapter import TTNetAdapter
from referee.run_referee import build_configs, DEFAULT_WEIGHTS


def build_cache(game='test_2', max_frames=None, device='cuda:0'):
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    video = os.path.join(root, 'dataset', 'test', 'videos', '{}.mp4'.format(game))

    configs = build_configs()
    adapter = TTNetAdapter(configs, DEFAULT_WEIGHTS, device=device)

    rows = []
    for obs in adapter.observations_from_video(video, max_frames=max_frames, progress_every=1000):
        rows.append({
            'f': obs.frame_id,
            't': round(obs.timestamp_s, 5),
            'v': obs.ball.visible,
            'p': ([round(obs.ball.center_px[0], 2), round(obs.ball.center_px[1], 2)]
                  if obs.ball.visible else None),
            'b': round(obs.event_scores.get('bounce') or 0.0, 5),
            'n': round(obs.event_scores.get('net_hit') or 0.0, 5),
            'tb': obs.extras.get('table_box'),
        })

    meta = {
        'game': game,
        'fps': 120.0,
        'n_frames': len(rows),
        'low_conf_dropped': adapter.low_conf_dropped,
        'frozen_dropped': adapter.frozen_frames_dropped,
    }
    return {'meta': meta, 'rows': rows}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', default='test_2')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--out', default='../results/cache_test_2.json')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    data = build_cache(a.game, a.max_frames, a.device)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    m = data['meta']
    print('已缓存 {} 帧 → {}'.format(m['n_frames'], a.out))
    print('低置信弃权 {}，冻结丢弃 {}'.format(m['low_conf_dropped'], m['frozen_dropped']))
