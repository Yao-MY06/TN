from torch.utils.data import Dataset
import os
import random
import pandas as pd
import cv2
import math
import numpy as np


class trackNetDataset(Dataset):
    """TrackNet dataset with optional on-the-fly augmentation.

    Augmentation design notes
    -------------------------
    A sample is 3 stacked frames (t-2, t-1, t) plus a 1-channel heat-map target.
    Two families of augmentation are applied differently:

    * Colour / brightness / noise  -> applied PER FRAME, independently. Real
      domain shift (different hall, lighting, camera) changes each frame the
      same way on average but not pixel-for-pixel; per-frame jitter teaches the
      model to rely on the ball rather than on absolute colour.

    * Geometry (scale, translate, rotate, flip) -> applied IDENTICALLY to all
      three frames AND to the heat-map, so the motion cue stays intact.

    The geometric transform is built once at the *native* resolution (1280x720
    or whatever the stored frames are) and then everything is resized to the
    network input size, so the heat-map peak always lands on the ball.
    """

    def __init__(self, mode, input_height=360, input_width=640,
                 path_dataset='./datasets/trackNet', augment=False):
        self.path_dataset = path_dataset
        assert mode in ['train', 'val'], 'incorrect mode'
        self.data = pd.read_csv(os.path.join(self.path_dataset, 'labels_{}.csv'.format(mode)))
        print('mode = {}, samples = {}'.format(mode, self.data.shape[0]))
        self.height = input_height
        self.width = input_width
        # augmentation is a training-only concern
        self.augment = bool(augment) and mode == 'train'

    def __len__(self):
        return self.data.shape[0]

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        path, path_prev, path_preprev, path_gt, x, y, status, vis = self.data.loc[idx, :]

        path = os.path.join(self.path_dataset, path)
        path_prev = os.path.join(self.path_dataset, path_prev)
        path_preprev = os.path.join(self.path_dataset, path_preprev)
        path_gt = os.path.join(self.path_dataset, path_gt)
        if math.isnan(x):
            x = -1
            y = -1

        frame_paths = [path, path_prev, path_preprev]
        inputs, outputs = self._load_sample(frame_paths, path_gt)

        return inputs, outputs, x, y, vis

    # ------------------------------------------------------------------ #
    def _load_sample(self, frame_paths, path_gt):
        raw = [cv2.imread(p) for p in frame_paths]
        gt = cv2.imread(path_gt)

        if any(im is None for im in raw) or gt is None:
            # Never silently train on a black frame: that would quietly poison
            # the model. Fail loudly instead so the caller can fix the paths.
            missing = [p for p, im in zip(frame_paths, raw) if im is None]
            if gt is None:
                missing.append(path_gt)
            raise FileNotFoundError(f"cannot read: {missing[:3]}")

        # keep everything 3-channel BGR
        raw = [im if im.ndim == 3 and im.shape[2] == 3 else cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
               for im in raw]
        gt = gt if gt.ndim == 3 else cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)

        if self.augment:
            raw, gt = self._augment_geometry(raw, gt)
            raw = [self._photometric(im) for im in raw]

        imgs = [cv2.resize(im, (self.width, self.height)) for im in raw]
        imgs = np.concatenate(imgs, axis=2).astype(np.float32) / 255.0

        # Keep the heat-map treatment identical to the original implementation:
        # every grey level is a class label, so the default (bilinear) resize is
        # the right thing here and preserves the smooth gaussian target.
        gt = cv2.resize(gt, (self.width, self.height))
        gt = gt[:, :, 0]
        gt = np.reshape(gt, (self.width * self.height))

        imgs = np.rollaxis(imgs, 2, 0)
        return imgs, gt

    # ------------------------------------------------------------------ #
    # geometry: same transform for the 3 frames and the heat-map
    # ------------------------------------------------------------------ #
    def _augment_geometry(self, frames, gt):
        """Geometric augmentation.

        Design constraints (learned the hard way):

        * The three frames AND the heat-map must receive the *identical*
          transform, otherwise the motion cue and the target disagree.
        * Anything that changes the frame size must be avoided or folded into
          the final resize, because resizing twice smears the gaussian peak.
          Here we therefore only translate / rotate (size preserving), plus a
          horizontal flip.
        * The image must be padded the same way as the heat-map. If the ball
          leaves the frame, BOTH should go to zero -- using BORDER_REPLICATE
          for the image and BORDER_CONSTANT for the target would manufacture
          frames where the ball is visible but the label is empty.
        """
        frames = [f.copy() for f in frames]
        gt = gt.copy()
        h, w = frames[0].shape[:2]

        # --- horizontal flip (task is ball localisation -> orientation-free) ---
        if random.random() < 0.5:
            frames = [cv2.flip(f, 1) for f in frames]
            gt = cv2.flip(gt, 1)

        # --- translation, size preserving, zero padding on both ---
        if random.random() < 0.7:
            max_sx = int(w * 0.05)
            max_sy = int(h * 0.05)
            dx = random.randint(-max_sx, max_sx)
            dy = random.randint(-max_sy, max_sy)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            size = (w, h)
            frames = [cv2.warpAffine(f, M, size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                      for f in frames]
            gt = cv2.warpAffine(gt, M, size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # --- small rotation, size preserving ---
        if random.random() < 0.3:
            ang = random.uniform(-6, 6)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            size = (w, h)
            frames = [cv2.warpAffine(f, M, size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                      for f in frames]
            gt = cv2.warpAffine(gt, M, size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        return frames, gt

    # ------------------------------------------------------------------ #
    # photometric: independent per frame
    # ------------------------------------------------------------------ #
    def _photometric(self, img):
        out = img.astype(np.float32)

        # brightness / contrast
        if random.random() < 0.7:
            alpha = random.uniform(0.8, 1.2)     # contrast
            beta = random.uniform(-25, 25)       # brightness
            out = out * alpha + beta

        # hue / saturation in HSV (hall lighting differs a lot)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-8, 8)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # gaussian noise
        if random.random() < 0.3:
            out = out + np.random.normal(0, random.uniform(2, 8), out.shape)

        # blur (motion / focus differences)
        if random.random() < 0.2:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out.astype(np.uint8), (k, k), 0).astype(np.float32)

        return np.clip(out, 0, 255).astype(np.uint8)
