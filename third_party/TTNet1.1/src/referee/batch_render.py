# -*- coding: utf-8 -*-
"""批量渲染多个测试视频的比分 UI 版本。

设计考虑
--------
1. **串行而非并行**：单卡 RTX 2080 Ti，并行会争抢显存与磁盘带宽，串行总耗时更短。
2. **每个视频独立缓存**：`infer_test_N_full.json`，中断后可跳过已完成的视频重跑。
3. **失败不致命**：单个视频失败不影响后续，最后汇总报告。
4. **进度可视化**：每个视频预计耗时较长（test_4 达 5 分钟视频），需明确进度。

用法::

    python -m referee.batch_render --games test_3 test_4 test_5 test_6 test_7
    python -m referee.batch_render --games test_3 --out-dir "C:/path"
"""

import os
import sys
import json
import time
import argparse
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.render_with_ui import run


DEFAULT_OUT_DIR = r'C:/Users/PC/Desktop/TTNet_test1_结果'
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


def process_one(game, out_dir, device='cuda:0', skip_existing_video=True,
                bounce_thresh=0.65, net_thresh=0.65,
                weights=None, tag=None, engine='yolo'):
    """处理单个视频，返回结果 dict。

    :param weights: 权重路径，None 用默认（自训练）
    :param tag: 文件名/缓存名后缀，用于区分不同权重的结果，
                例如 tag='pretrained' → test_3_pretrained.mp4 / infer_test_3_pretrained.json
    :param engine: 计分引擎，'yolo'（默认，YOLO 共用状态机）/ 'legacy'（TTNet 原状态机）
    """
    video = os.path.join(ROOT, 'dataset', 'test', 'videos', '{}.mp4'.format(game))
    suffix = '_{}'.format(tag) if tag else ''
    out_video = os.path.join(out_dir, '{}_比分版{}.mp4'.format(game, suffix))
    cache = os.path.join(ROOT, 'results',
                         'infer_{}_full{}.json'.format(game, suffix))

    if not os.path.isfile(video):
        return {'game': game, 'status': 'missing_video'}

    if skip_existing_video and os.path.isfile(out_video) and os.path.getsize(out_video) > 1024 * 1024:
        print('\n[跳过] {} 已存在（{} MB）'.format(
            os.path.basename(out_video), os.path.getsize(out_video) // 1024 // 1024))
        return {'game': game, 'status': 'skipped', 'output': out_video}

    t0 = time.time()
    print('\n' + '#' * 72)
    print('# {}{}  [engine={}]'.format(game, '  [{}]'.format(tag) if tag else '', engine))
    print('#' * 72)
    kw = {'bounce_thresh': bounce_thresh, 'net_thresh': net_thresh,
          'device': device, 'cache_path': cache, 'engine': engine}
    if weights:
        kw['weights'] = weights
    summary = run(video, out_video, **kw)
    summary['elapsed_s'] = round(time.time() - t0, 1)
    summary['status'] = 'ok'
    summary['game'] = game
    summary['tag'] = tag
    summary['engine'] = engine
    summary['weights'] = weights or '(default)'
    return summary


def main(games, out_dir, device='cuda:0', skip_existing_video=True,
         weights=None, tag=None, summary_name='批量结果汇总.json',
         engine='yolo'):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    t_all = time.time()

    for idx, g in enumerate(games, 1):
        print('\n>>> 进度 {}/{}: {}'.format(idx, len(games), g))
        try:
            r = process_one(g, out_dir, device=device,
                            skip_existing_video=skip_existing_video,
                            weights=weights, tag=tag, engine=engine)
        except Exception as e:
            traceback.print_exc()
            r = {'game': g, 'status': 'failed', 'error': str(e)}
        results.append(r)

    # ---- 汇总
    print('\n' + '=' * 72)
    print('批量渲染汇总（总耗时 {:.1f} 分钟）'.format((time.time() - t_all) / 60))
    print('=' * 72)
    for r in results:
        if r['status'] == 'ok':
            fs = r['final_score']
            det = r['detected_events']
            print('  {:<8} OK   {:<5}分 A{}-B{} | frames {} | B:{} N:{} | {:.0f}s'.format(
                r['game'], len(r.get('points', [])), fs.get('A', 0), fs.get('B', 0),
                r['frames'], det.get('BOUNCE', 0), det.get('NET_HIT', 0),
                r.get('elapsed_s', 0)))
        elif r['status'] == 'skipped':
            print('  {:<8} ⏭  已存在，跳过'.format(r['game']))
        else:
            print('  {:<8} ❌ {}'.format(r['game'], r.get('error', r['status'])))

    # 写汇总 JSON
    summary_path = os.path.join(out_dir, summary_name)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\n汇总已写入: {}'.format(summary_path))
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='+', default=['test_3', 'test_4', 'test_5', 'test_6', 'test_7'])
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--no-skip', action='store_true', help='不跳过已存在的输出视频')
    ap.add_argument('--weights', default=None,
                    help='权重路径；不填用默认（自训练 ttnet_3rd_phase_epoch_30.pth）')
    ap.add_argument('--tag', default=None,
                    help='结果后缀，用于区分不同权重，如 pretrained → test_3_比分版_pretrained.mp4')
    ap.add_argument('--summary-name', default=None, help='汇总 JSON 文件名')
    ap.add_argument('--engine', choices=('yolo', 'legacy'), default='yolo',
                    help='计分引擎：yolo（默认，YOLO 共用状态机）/ legacy（TTNet 原状态机）')
    a = ap.parse_args()
    sn = a.summary_name or ('批量结果汇总{}.json'.format(
        '_{}'.format(a.tag) if a.tag else ''))
    main(a.games, a.out_dir, device=a.device, skip_existing_video=not a.no_skip,
         weights=a.weights, tag=a.tag, summary_name=sn, engine=a.engine)
