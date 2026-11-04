# -*- coding: utf-8 -*-
"""TTNet 推理引擎：单窗口 (27, 128, 320) → 原始输出。

两个实现，接口相同（``infer(seq) -> dict``）：

- ``TorchEngine``：用 TTNet 仓库自己的 ``create_model`` + ``load_pretrained_model``
  构造并加载权重，调用 ``run_demo``。原机（有 GPU）用它。CPU 兼容处理在外层做，
  不改 TTNet 仓库文件（见 ``_patch_cpu_normalize``）。
- ``NumpyEngine``：不依赖 torch 的逐层复刻（global / local / event，可选 seg），
  从 .pth 直接解包权重。用途：没有 torch 的电脑上校验权重、跑测试、慢速 live。
  与 ``run_demo`` 逐项对齐，包括两处原实现的“怪行为”（照抄，不修）：
    1. 归一化的 mean/std 用 ``repeat_interleave`` 展开成 [m0]*9+[m1]*9+[m2]*9，
       而输入通道是 [R0,G0,B0,R1,...]，两者并不对应；
    2. ``__crop_original_batch_right__`` 里的阈值掩码写进了一个临时 list，
       对后续 argmax 不起作用，所以裁剪中心永远是原始分布的 argmax。

输入约定（两引擎一致，也与 chagoup ttnet_adapter 一致）：9 帧 RGB 小图
（cv2.resize 到 320x128，INTER_LINEAR），``np.dstack(frames).transpose(2,0,1)``，
取值 0..255，未归一化。

输出 dict：
    global_x (320,), global_y (128,), local_x (320,), local_y (128,)  —— sigmoid 概率
    events (2,)   —— [bounce, net]，顺序取自 configs.events_dict
    crop (x_min, x_max, y_min, y_max, x_pad, y_pad)  —— 局部阶段的裁剪参数（原图坐标）
    seg (3, 128, 320) —— 仅 compute_seg=True 时
"""
from __future__ import annotations

import collections
import os
import pickle
import sys
import types
import zipfile

import numpy as np

W_ORG, H_ORG = 1920, 1080
W_IN, H_IN = 320, 128
NUM_FRAMES = 9
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BN_EPS = 1e-5


# ---------------------------------------------------------------------------
# 帧预处理（cache 与 live 必须共用这一个函数，保证输入逐字节一致）
# ---------------------------------------------------------------------------
def small_rgb(frame_bgr):
    """原图 BGR → 320x128 RGB uint8（与 ttnet_adapter.observations_from_video 相同）。"""
    import cv2
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (W_IN, H_IN))


def stack_window(frames):
    """9 张 (128,320,3) 小图 → (27,128,320) float32，通道顺序 [R0,G0,B0,R1,...]。"""
    if len(frames) != NUM_FRAMES:
        raise ValueError('需要 9 帧，得到 {}'.format(len(frames)))
    return np.dstack(frames).transpose(2, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 不依赖 torch 的 .pth 读取
# ---------------------------------------------------------------------------
_DTYPES = {
    'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16,
    'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
    'BFloat16Storage': None,
}


class _Opaque(dict):
    """未知类的占位（configs 是 EasyDict、优化器状态等，都用不到）。"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)

    def __setattr__(self, k, v):
        self[k] = v


def load_pth_state_dict(path, strip_prefix='model.'):
    """读 torch.save 的 zip 格式 checkpoint，返回 {name: np.ndarray}，不需要 torch。

    只支持新式 zip 格式（torch>=1.6 默认）。``strip_prefix`` 去掉
    Unbalance_Loss_Model 包装带来的 ``model.`` 前缀。
    返回 (state_dict, meta)，meta 含 epoch 与 checkpoint 顶层键。
    """
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    root = names[0].split('/')[0]

    def rebuild(storage, offset, size, stride, *rest):
        key, dtype = storage
        raw = zf.read('{}/data/{}'.format(root, key))
        flat = np.frombuffer(raw, dtype=dtype)
        if len(size) == 0:
            return flat[offset].copy()
        itemsize = flat.itemsize
        arr = np.lib.stride_tricks.as_strided(
            flat[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride))
        return np.ascontiguousarray(arr)

    class _U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if mod == 'torch._utils' and name == '_rebuild_tensor_v2':
                return rebuild
            if mod == 'torch' and name in _DTYPES:
                return name  # 旧式 typed storage：返回类名字符串
            if mod == 'torch' and name in ('float32', 'float16', 'float64', 'int64', 'int32', 'uint8', 'bool'):
                return name
            return type(name, (_Opaque,), {})

        def persistent_load(self, pid):
            # ('storage', storage_type, key, location, numel)
            _, stype, key, _loc, _numel = pid
            if isinstance(stype, str) and stype in _DTYPES:
                dt = _DTYPES[stype]
            elif isinstance(stype, str):
                dt = np.dtype(stype)
            else:
                dt = np.float32
            if dt is None:
                raise NotImplementedError('bfloat16 权重未支持')
            return (key, dt)

    with zf.open('{}/data.pkl'.format(root)) as f:
        ckpt = _U(f).load()
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    out = collections.OrderedDict()
    for k, v in sd.items():
        if strip_prefix and k.startswith(strip_prefix):
            k = k[len(strip_prefix):]
        out[k] = np.asarray(v)
    meta = {'keys': list(ckpt.keys()) if isinstance(ckpt, dict) else [],
            'epoch': ckpt.get('epoch') if isinstance(ckpt, dict) else None}
    return out, meta


# ---------------------------------------------------------------------------
# numpy 算子（单样本，CHW，float32）
# ---------------------------------------------------------------------------
def _conv2d(x, w, b, pad=0):
    o, c, kh, kw = w.shape
    if kh == 1 and kw == 1 and pad == 0:
        out = w.reshape(o, c) @ x.reshape(c, -1)
        out = out.reshape(o, x.shape[1], x.shape[2])
    else:
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad))) if pad else x
        win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(1, 2))  # C,Ho,Wo,kh,kw
        ho, wo = win.shape[1], win.shape[2]
        cols = win.transpose(1, 2, 0, 3, 4).reshape(ho * wo, c * kh * kw)
        out = (cols @ w.reshape(o, -1).T).T.reshape(o, ho, wo)
    if b is not None:
        out = out + b[:, None, None]
    return out.astype(np.float32, copy=False)


def _tconv2d(x, w, b, stride, pad, out_pad):
    ci, co, k, _ = w.shape
    c, h, wd = x.shape
    z = np.zeros((c, (h - 1) * stride + 1, (wd - 1) * stride + 1), np.float32)
    z[:, ::stride, ::stride] = x
    lo, hi = k - 1 - pad, k - 1 - pad + out_pad
    z = np.pad(z, ((0, 0), (lo, hi), (lo, hi)))
    wf = np.ascontiguousarray(w[:, :, ::-1, ::-1].transpose(1, 0, 2, 3))
    return _conv2d(z, wf, b, 0)


def _bn(x, p, name):
    g, bt = p[name + '.weight'], p[name + '.bias']
    m, v = p[name + '.running_mean'], p[name + '.running_var']
    s = g / np.sqrt(v + BN_EPS)
    return (x * s[:, None, None] + (bt - m * s)[:, None, None]).astype(np.float32)


def _relu(x):
    return np.maximum(x, 0, out=x)


def _pool2(x):
    c, h, w = x.shape
    return x[:, :h // 2 * 2, :w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def _linear(x, p, name):
    return x @ p[name + '.weight'].T + p[name + '.bias']


def _nearest_src(out_size, in_size):
    """torch upsample_nearest 的源索引（float32 scale，floorf）。"""
    if out_size == in_size:
        return np.arange(out_size)
    if out_size == 2 * in_size:
        return np.arange(out_size) >> 1
    scale = np.float32(in_size) / np.float32(out_size)
    idx = np.floor(np.arange(out_size, dtype=np.float32) * scale).astype(np.int64)
    return np.minimum(idx, in_size - 1)


_SRC_Y = _nearest_src(H_ORG, H_IN)
_SRC_X = _nearest_src(W_ORG, W_IN)


class NumpyEngine:
    name = 'numpy'

    def __init__(self, weights_path=None, state_dict=None, compute_seg=False):
        if state_dict is None:
            state_dict, self.meta = load_pth_state_dict(weights_path)
        else:
            self.meta = {}
        self.p = {k: (v.astype(np.float32) if v.dtype != np.int64 else v) for k, v in state_dict.items()}
        self.compute_seg = compute_seg
        self.mean = np.repeat(np.asarray(MEAN, np.float32), NUM_FRAMES)[:, None, None]
        self.std = np.repeat(np.asarray(STD, np.float32), NUM_FRAMES)[:, None, None]

    # ---- 子网络
    def _norm(self, x):
        return ((x / np.float32(255.0) - self.mean) / self.std).astype(np.float32)

    def _block(self, x, name, pool=True):
        x = _relu(_bn(_conv2d(x, self.p[name + '.conv.weight'], self.p[name + '.conv.bias'], 1), self.p,
                      name + '.batchnorm'))
        return _pool2(x) if pool else x

    def _ball_stage(self, x, s):
        p = self.p
        x = _relu(_bn(_conv2d(x, p[s + '.conv1.weight'], p[s + '.conv1.bias']), p, s + '.batchnorm'))
        b2 = self._block(self._block(x, s + '.convblock1'), s + '.convblock2')
        b3 = self._block(b2, s + '.convblock3')
        b4 = self._block(b3, s + '.convblock4')
        b5 = self._block(b4, s + '.convblock5')
        feat = self._block(b5, s + '.convblock6')
        v = feat.reshape(-1)
        f = np.maximum(_linear(v, p, s + '.fc1'), 0)
        fx = np.maximum(_linear(f, p, s + '.fcx1'), 0)
        fy = np.maximum(_linear(f, p, s + '.fcy1'), 0)
        px = _sigmoid(_linear(fx, p, s + '.fcx2'))
        py = _sigmoid(_linear(fy, p, s + '.fcy2'))
        return px, py, feat, (b2, b3, b4, b5)

    def _events(self, gfeat, lfeat):
        p, s = self.p, 'events_spotting'
        x = np.concatenate([gfeat, lfeat], axis=0)
        x = _relu(_bn(_conv2d(x, p[s + '.conv1.weight'], p[s + '.conv1.bias']), p, s + '.batchnorm'))
        x = self._block(x, s + '.convblock', pool=False)
        x = self._block(x, s + '.convblock', pool=False)
        v = np.maximum(_linear(x.reshape(-1), p, s + '.fc1'), 0)
        return _sigmoid(_linear(v, p, s + '.fc2'))

    def _deconv(self, x, s):
        p = self.p
        x = _relu(_bn(_conv2d(x, p[s + '.conv1.weight'], p[s + '.conv1.bias']), p, s + '.batchnorm1'))
        x = _relu(_bn(_tconv2d(x, p[s + '.tconv.weight'], p[s + '.tconv.bias'], 2, 1, 1), p, s + '.batchnorm_tconv'))
        x = _relu(_bn(_conv2d(x, p[s + '.conv2.weight'], p[s + '.conv2.bias']), p, s + '.batchnorm2'))
        return x

    def _seg(self, b2, b3, b4, b5):
        p, s = self.p, 'segmentation'
        x = self._deconv(b5, s + '.deconvblock5') + b4
        x = self._deconv(x, s + '.deconvblock4') + b3
        x = self._deconv(x, s + '.deconvblock3') + b2
        x = self._deconv(x, s + '.deconvblock2')
        x = _relu(_tconv2d(x, p[s + '.tconv.weight'], p[s + '.tconv.bias'], 2, 0, 0))
        x = _relu(_conv2d(x, p[s + '.conv1.weight'], p[s + '.conv1.bias'], 0))
        return _sigmoid(_conv2d(x, p[s + '.conv2.weight'], p[s + '.conv2.bias'], 1))

    @staticmethod
    def crop_params(gx, gy):
        """复刻 __crop_original_batch_right__（掩码不生效，见模块说明）。"""
        xc = int(int(np.argmax(gx)) * (W_ORG / W_IN))
        yc = int(int(np.argmax(gy)) * (H_ORG / H_IN))
        x_min = max(0, xc - W_IN // 2)
        y_min = max(0, yc - H_IN // 2)
        x_max = min(W_ORG, x_min + W_IN)
        y_max = min(H_ORG, y_min + H_IN)
        h_crop, w_crop = y_max - y_min, x_max - x_min
        x_pad = y_pad = 0
        if h_crop != H_IN or w_crop != W_IN:
            x_pad = int((W_IN - w_crop) / 2)
            y_pad = int((H_IN - h_crop) / 2)
        return x_min, x_max, y_min, y_max, x_pad, y_pad

    def infer(self, seq):
        seq = np.asarray(seq, np.float32)
        gx, gy, gfeat, blocks = self._ball_stage(self._norm(seq), 'ball_global_stage')
        x_min, x_max, y_min, y_max, x_pad, y_pad = crop = self.crop_params(gx, gy)
        local = np.zeros_like(seq)
        rows, cols = _SRC_Y[y_min:y_max], _SRC_X[x_min:x_max]
        local[:, y_pad:y_pad + (y_max - y_min), x_pad:x_pad + (x_max - x_min)] = seq[:, rows][:, :, cols]
        lx, ly, lfeat, _ = self._ball_stage(self._norm(local), 'ball_local_stage')
        out = {'global_x': gx, 'global_y': gy, 'local_x': lx, 'local_y': ly,
               'events': self._events(gfeat, lfeat), 'crop': crop}
        if self.compute_seg:
            out['seg'] = self._seg(*blocks)
        return out


# ---------------------------------------------------------------------------
# torch 引擎（原机）
# ---------------------------------------------------------------------------
def _patch_cpu_normalize(ttnet, device):
    """TTNet.__normalize__ 写死了 ``.cuda()``。在实例上替换成按 device 搬运的版本，
    不改 TTNet 仓库文件。GPU 上行为不变。"""
    def _normalize(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return (x / 255. - self.mean) / self.std
    ttnet.__normalize__ = types.MethodType(_normalize, ttnet)


def build_ttnet_configs(no_cuda):
    from easydict import EasyDict
    c = EasyDict()
    c.arch = 'ttnet'
    c.dropout_p = 0.5
    c.multitask_learning = False
    c.tasks = ['global', 'local', 'event', 'seg']
    c.input_size = (W_IN, H_IN)
    c.thresh_ball_pos_mask = 0.05
    c.num_frames_sequence = NUM_FRAMES
    c.sigma = 1.0
    c.events_weights_loss = (1.0, 3.0)
    c.num_events = 2
    c.tasks_loss_weight = [1.0, 1.0, 1.0, 1.0]
    c.device = 'cpu' if no_cuda else 'cuda'
    return c


class TorchEngine:
    name = 'torch'

    def __init__(self, weights_path, ttnet_src, device='cuda:0', compute_seg=False):
        import torch
        if ttnet_src not in sys.path:
            sys.path.insert(0, ttnet_src)
        from models.model_utils import create_model, load_pretrained_model
        self.torch = torch
        self.device = torch.device(device)
        cpu = self.device.type == 'cpu'
        configs = build_ttnet_configs(no_cuda=cpu)
        model = create_model(configs).to(self.device)
        gpu_idx = None if cpu else (self.device.index or 0)
        model = load_pretrained_model(model, weights_path, gpu_idx, overwrite_global_2_local=False)
        model.eval()
        ttnet = model.model
        _patch_cpu_normalize(ttnet, self.device)
        self.ttnet = ttnet
        self.compute_seg = compute_seg

    def infer(self, seq):
        torch = self.torch
        t = torch.from_numpy(np.asarray(seq, np.float32)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            g, l, ev, seg = self.ttnet.run_demo(t)
        def a(v):
            return v.detach().float().cpu().numpy().ravel()
        out = {'global_x': a(g[0]), 'global_y': a(g[1]), 'local_x': a(l[0]), 'local_y': a(l[1]),
               'events': a(ev), 'crop': NumpyEngine.crop_params(a(g[0]), a(g[1]))}
        if self.compute_seg:
            out['seg'] = seg.detach().float().cpu().numpy()[0]
        return out


def make_engine(kind, weights_path, ttnet_src=None, device='cuda:0', compute_seg=False):
    """kind: 'torch' | 'numpy' | 'auto'（有 torch 用 torch，否则 numpy）。"""
    if kind == 'auto':
        try:
            import torch  # noqa: F401
            kind = 'torch'
        except ImportError:
            kind = 'numpy'
    if kind == 'torch':
        if not ttnet_src:
            raise SystemExit('torch 引擎需要 --ttnet-src 指向 TTNet/src')
        return TorchEngine(weights_path, ttnet_src, device=device, compute_seg=compute_seg)
    return NumpyEngine(weights_path, compute_seg=compute_seg)
