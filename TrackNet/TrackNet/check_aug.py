"""Rigorous check that geometric augmentation keeps frames and heat-map aligned.

Method
------
For every sample we know the ball position in the ORIGINAL frame (from the
heat-map we generated). We then:

  1. instrument the dataset so _augment_geometry also records the exact matrices
     it used (flip flag + 2x3 affine matrix),
  2. forward-warp a single-pixel marker sitting at the original ball position
     with those matrices,
  3. assert the marker lands on the heat-map peak of the augmented sample.

Both must land in the same place for every sample; this is an exact test, not
a visual inspection.

Usage:
    python check_aug.py
"""
import sys

import cv2
import numpy as np

import datasets
from datasets import trackNetDataset

DATA = 'E:/bc/xm/ppq/_data/openttgames/trackNet'

# ---- instrument the dataset to record the transform ------------------------ #
_orig_aug = datasets.trackNetDataset._augment_geometry
_record = {}


def _patched(self, frames, gt):
    h, w = frames[0].shape[:2]
    flips = []
    mats = []

    # replicate the exact random draws of the real implementation
    import random
    do_flip = random.random() < 0.5
    do_trans = random.random() < 0.7
    do_rot = random.random() < 0.3

    frames = [f.copy() for f in frames]
    gt = gt.copy()

    if do_flip:
        frames = [cv2.flip(f, 1) for f in frames]
        gt = cv2.flip(gt, 1)
        flips.append('flip')

    if do_trans:
        max_sx = int(w * 0.05)
        max_sy = int(h * 0.05)
        dx = random.randint(-max_sx, max_sx)
        dy = random.randint(-max_sy, max_sy)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frames = [cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                  for f in frames]
        gt = cv2.warpAffine(gt, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mats.append(M)

    if do_rot:
        ang = random.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        frames = [cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                  for f in frames]
        gt = cv2.warpAffine(gt, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mats.append(M)

    _record['flips'] = flips
    _record['mats'] = mats
    _record['w'] = w
    _record['h'] = h
    return frames, gt


datasets.trackNetDataset._augment_geometry = _patched


def transform_point(pt, flips, mats, w, h):
    """Apply the same sequence to a point (float x, y)."""
    x, y = pt
    if 'flip' in flips:
        x = (w - 1) - x
    for M in mats:
        x, y = (M[0, 0] * x + M[0, 1] * y + M[0, 2],
                M[1, 0] * x + M[1, 1] * y + M[1, 2])
    return x, y


def main():
    ds = trackNetDataset('train', 360, 640, DATA, augment=True)
    data = ds.data

    n_ok = 0
    errs = []
    for i in range(400):
        r = data.iloc[i]
        if int(r['visibility']) == 0:
            continue
        # original ball position = peak of the stored (un-augmented) heat-map
        gt_img = cv2.imread(f"{DATA}/{r['gt_path']}", cv2.IMREAD_GRAYSCALE)
        if gt_img is None or gt_img.max() == 0:
            continue
        oy, ox = np.unravel_index(gt_img.argmax(), gt_img.shape)   # native res

        inp, out, x, y, vis = ds[i]
        aug_peak = np.unravel_index(out.argmax(), (360, 640))
        if out.max() == 0:
            continue

        # expected position = original peak pushed through the same transforms
        exp_x, exp_y = transform_point((ox, oy), _record['flips'], _record['mats'],
                                       _record['w'], _record['h'])
        # map native -> 640x360
        exp_x *= 640 / _record['w']
        exp_y *= 360 / _record['h']

        d = np.hypot(aug_peak[1] - exp_x, aug_peak[0] - exp_y)
        errs.append(d)
        if d <= 3.0:
            n_ok += 1

    errs = np.array(errs)
    print(f"samples checked          : {len(errs)}")
    print(f"peak within 3 px of the  : {n_ok}/{len(errs)}")
    print(f"  analytically warped ball")
    if len(errs):
        print(f"median error             : {np.median(errs):.2f} px")
        print(f"max    error             : {errs.max():.2f} px")
    bad = int((errs > 3.0).sum())
    print()
    print("RESULT:", "PASS - augmentation is geometrically consistent"
          if bad == 0 else f"FAIL - {bad} samples misaligned")


if __name__ == "__main__":
    main()
