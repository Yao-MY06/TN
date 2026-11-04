# -*- coding: utf-8 -*-
"""将 images 目录下的训练帧预缩放到 320x128(训练输入尺寸),加速数据加载。
线程池实现(Windows 下比多进程稳定),幂等(已是 320x128 的帧直接跳过)。"""
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

W, H = 320, 128


def scale_one(path):
    img = cv2.imread(path)
    if img is None:
        return 0
    if img.shape[0] == H and img.shape[1] == W:
        return -1  # 已缩放,跳过
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(path, img)
    return 1


def main():
    frames = []
    for typ in ['training', 'test']:
        frames += sorted(glob.glob('dataset/{}/images/*/*.jpg'.format(typ)))
    print('total frames:', len(frames), flush=True)
    t0 = time.time()
    scaled = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(scale_one, p): p for p in frames}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r == 1:
                scaled += 1
            elif r == -1:
                skipped += 1
            else:
                failed += 1
            if (i + 1) % 5000 == 0:
                print('processed {}/{} ({:.0f}s)'.format(i + 1, len(frames), time.time() - t0), flush=True)
    print('done: scaled {}, skipped {}, failed {}, {:.0f}s'.format(scaled, skipped, failed, time.time() - t0))


if __name__ == '__main__':
    main()
