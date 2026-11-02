"""TrackNet 适配器：把 TrackNet 的输出转换成统一 Observation 逐帧观测。

两种输入模式：
  1) 直接跑视频（推荐，能给出热图置信度）
     python tracknet_adapter.py --video test_2.mp4 --model_path ../TrackNet/TrackNet/exps/tt_g45_aug/model_best.pt \
            --out observations.jsonl
  2) 转换已有的逐帧坐标 CSV（track_video.py 产出，速度快，但没有置信度 → 填 None）
     python tracknet_adapter.py --csv track.csv --fps 120 --image_size 1920 1080 --out observations.jsonl
     若 CSV 带 interp 列（interpolate_track.py 的补帧产物），补出来的帧会被标成
     predicted=true 且 confidence=null，下游不会把它当成真实检出。

输出：JSONL，每帧一行，字段见 observation.py / 方案文档第 7.1 节。

关于 confidence（务必写进对接文档）：
  TrackNet 是逐像素 256 类分类，argmax 后的类值 = 该像素的高斯热图灰度（0~255）。
  这里取「检出中心 ±2 像素窗口内热图灰度的最大值 / 255」作为置信度，落在 [0, 1]。
  后处理只接受热图 ≥127 的区域，因此绝大多数检出 ≥ 0.50；少数霍夫圆落在
  拖影边缘（邻域内无高值像素）时会接近 0，这类点基本都是误检，下游可据此过滤。
  它是热图峰的强度，不是概率，也**不能**和 YOLO 的检测置信度直接比较。

  ``--min_conf``（默认 0.5）在**本适配器出口**就把低于阈值的检出丢成
  "本帧无检测"（visible=false），而不是留给下游过滤：实测 game_4 上 52.833s
  的 conf=0.000 假点 y 偏 26px，进轨迹后会顶掉真实的己方首跳位置，
  直接导致发球阶段误判。设 0 可关闭过滤（用于对比）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TT_DIR = os.path.join(_HERE, "..", "TrackNet", "TrackNet")
sys.path.insert(0, _TT_DIR)
sys.path.insert(0, _HERE)

from model import BallTrackerNet          # noqa: E402
from general import postprocess           # noqa: E402
from observation import make_observation  # noqa: E402

MODEL_H, MODEL_W = 360, 640
SPACE_SCALE = 2        # postprocess() 返回的坐标是模型输入的 2 倍（1280x720）


def probe_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    meta = dict(
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
    )
    cap.release()
    return meta


def _heatmap_peak(classmap, x_scaled, y_scaled, radius=2):
    """检出中心邻域的热图峰值 / 255。x_scaled/y_scaled 为 SPACE_SCALE 空间坐标。

    绝大多数检出 ≥ 0.5（后处理只接受 ≥127 的热图区域）；少数霍夫圆落在
    拖影边缘、邻域内没有高值像素，会接近 0，这类点基本是误检。
    """
    # 模型输出把空间维展平了：(B, 256, 230400) -> 每样本是 (230400,)
    cm = np.asarray(classmap)
    if cm.ndim == 1:
        cm = cm.reshape(MODEL_H, MODEL_W)
    hx = int(round(x_scaled / SPACE_SCALE))
    hy = int(round(y_scaled / SPACE_SCALE))
    h, w = cm.shape
    hx = min(max(hx, 0), w - 1)
    hy = min(max(hy, 0), h - 1)
    y0, y1 = max(0, hy - radius), min(h, hy + radius + 1)
    x0, x1 = max(0, hx - radius), min(w, hx + radius + 1)
    return float(cm[y0:y1, x0:x1].max()) / 255.0


def run_video(video, model_path, out, max_frames=None, batch_size=8,
              min_radius=2, max_radius=12, amp=True, image_size=None,
              min_conf=0.5):
    import torch
    from observation import write_observations

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallTrackerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    meta = probe_video(video)
    n = meta["n"] if max_frames is None else min(meta["n"], max_frames)
    w, h, fps = meta["w"], meta["h"], meta["fps"]
    if image_size:
        w, h = image_size
    sx, sy = w / (MODEL_W * SPACE_SCALE), h / (MODEL_H * SPACE_SCALE)
    print(f"video: {video}")
    print(f"  {n} frames | {w}x{h} | {fps:.1f} fps | device={device} | model={model_path}")

    cap = cv2.VideoCapture(video)
    buf = deque(maxlen=3)
    pending = []          # [(frame_id, 9ch array)]
    obs_buffer = []       # 已生成、待写盘的观测（保证按帧序输出）
    n_lowconf = [0]       # 被 min_conf 丢弃的低置信检出帧数

    def flush():
        if not pending:
            return
        arr = np.stack([p[1] for p in pending])
        t = torch.from_numpy(arr).float().to(device)
        with torch.no_grad():
            if amp and device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(t)
                logits = logits.float()
            else:
                logits = model(t)
        maps = logits.argmax(dim=1).detach().cpu().numpy()
        for (fid, _), cm in zip(pending, maps):
            # postprocess 会就地修改入参（*=255），因此传副本，原图留给置信度计算
            x, y = postprocess(cm.copy(), height=MODEL_H, width=MODEL_W,
                               min_radius=min_radius, max_radius=max_radius)
            if x is None:
                obs_buffer.append(make_observation(fid, fid / fps, "tracknet", (w, h)))
            else:
                conf = _heatmap_peak(cm, x, y)
                # 低置信检出直接丢弃（见文件头 "关于 confidence"）。
                # 典型误检：霍夫圆落在球拖影/台面白边/横幅边缘上，热图邻域里
                # 根本没有高值像素，conf 接近 0，但 postprocess 仍会返回一个圆。
                # 这类点坐标偏移可达 20~30px，进轨迹后会把落台/击球全带偏。
                if min_conf > 0 and conf < min_conf:
                    n_lowconf[0] += 1
                    obs_buffer.append(make_observation(fid, fid / fps, "tracknet", (w, h)))
                    continue
                obs_buffer.append(
                    make_observation(fid, fid / fps, "tracknet", (w, h),
                                     center_px=(x * sx, y * sy), confidence=conf)
                )
        pending.clear()

    frame_id = -1
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id >= n:
            break
        small = cv2.resize(frame, (MODEL_W, MODEL_H))
        buf.append(small)
        if len(buf) < 3:
            # 前两帧凑不出三帧输入 → 直接输出"无检测"
            obs_buffer.append(make_observation(frame_id, frame_id / fps, "tracknet", (w, h)))
        else:
            # 通道顺序与 track_video.py / 训练时一致：由新到旧 (t, t-1, t-2)
            stack = np.concatenate((buf[2], buf[1], buf[0]), axis=2).astype(np.float32) / 255.0
            pending.append((frame_id, np.rollaxis(stack, 2, 0)))
            if len(pending) >= batch_size:
                flush()
        if frame_id % 500 == 0:
            print(f"  {frame_id}/{n} frames, {len(obs_buffer)} observations, "
                  f"{time.time() - t0:.1f}s", flush=True)
    cap.release()
    flush()

    obs_buffer.sort(key=lambda o: o["frame_id"])
    n_written = write_observations(out, obs_buffer)
    dt = time.time() - t0
    if n_lowconf[0]:
        print(f"  low-confidence filtered: {n_lowconf[0]} frames "
              f"(heatmap peak < {min_conf:.2f}, treated as no-detection)")
    _summary(obs_buffer, n_written, dt, fps, out)
    return obs_buffer


def run_csv(csv_path, out, fps, image_size):
    import pandas as pd
    from observation import write_observations

    df = pd.read_csv(csv_path)
    required = {"frame", "x", "y", "detected"}
    if not required.issubset(df.columns):
        raise ValueError(f"csv must have columns {required}, got {list(df.columns)}")

    w, h = image_size
    obs = []
    has_interp = "interp" in df.columns   # interpolate_track.py 的补帧标记
    n_pred = 0
    for row in df.itertuples(index=False):
        fid = int(row.frame)
        is_pred = bool(getattr(row, "interp", 0)) if has_interp else False
        if int(row.detected) == 1 and not (pd.isna(row.x) or pd.isna(row.y)):
            # CSV 里没有热图，无法给出可信分数 → confidence = None
            obs.append(make_observation(fid, fid / fps, "tracknet", (w, h),
                                        center_px=(row.x, row.y), confidence=None,
                                        predicted=is_pred))
            n_pred += int(is_pred)
        else:
            obs.append(make_observation(fid, fid / fps, "tracknet", (w, h)))
    n_written = write_observations(out, obs)
    if n_pred:
        print(f"  NOTE: {n_pred} frames are trajectory-predicted (predicted=true, confidence=null)")
    _summary(obs, n_written, 0.0, fps, out)
    return obs


def _summary(obs, n_written, dt, fps, out):
    total = len(obs)
    vis = sum(1 for o in obs if o["ball"]["visible"])
    n_pred = sum(1 for o in obs if o["ball"].get("predicted", False))
    confs = [o["ball"]["confidence"] for o in obs if o["ball"]["confidence"] is not None]
    print(f"  observations: {n_written} -> {out}")
    print(f"  visible: {vis}/{total} ({vis / max(total, 1) * 100:.1f}%)"
          + (f"  [其中 {n_pred} 帧为轨迹补帧，predicted=true]" if n_pred else ""))
    if confs:
        print(f"  confidence: median {np.median(confs):.3f}  min {min(confs):.3f}  max {max(confs):.3f}")
    if dt > 0:
        print(f"  elapsed {dt:.1f}s ({total / dt:.1f} fps)")


def main():
    ap = argparse.ArgumentParser(description="TrackNet -> unified Observation adapter")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="source video (runs inference)")
    src.add_argument("--csv", help="per-frame CSV produced by track_video.py")
    ap.add_argument("--model_path", help="checkpoint (required with --video)")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--fps", type=float, default=None,
                    help="video fps; required for --csv (unless it can be probed)")
    ap.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="original frame size, default: probe from video / 1920 1080 for --csv")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--min_radius", type=int, default=2)
    ap.add_argument("--max_radius", type=int, default=12, help="table tennis ~12 (tennis 7)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out_mot", default=None,
                    help="also export MOT Challenge txt next to the JSONL "
                         "(standard format: evaluate with eval_mot.py / py-motmetrics, "
                         "or load it in supervision / trackers)")
    ap.add_argument("--ball_radius", type=float, default=None,
                    help="box radius used for the MOT export when only a centre point "
                         "exists (default: 1.2%% of image width; 1920 -> 23 px)")
    ap.add_argument("--min_conf", type=float, default=0.5,
                    help="丢弃热图峰值低于该值的检出（当作本帧无检测）。"
                         "后处理只接受热图 >=127 的区域，正常检出 >=0.50；"
                         "接近 0 的是霍夫圆落在拖影/白边上的误检。0 = 关闭过滤。")
    args = ap.parse_args()

    if args.video:
        if not args.model_path:
            ap.error("--model_path is required with --video")
        run_video(args.video, args.model_path, args.out, args.max_frames, args.batch_size,
                  args.min_radius, args.max_radius, amp=not args.no_amp,
                  image_size=args.image_size, min_conf=args.min_conf)
    else:
        if args.min_conf > 0:
            print("  NOTE: --min_conf is ignored in --csv mode (CSV has no confidence)")
        fps = args.fps
        if fps is None:
            ap.error("--fps is required with --csv")
        image_size = tuple(args.image_size) if args.image_size else (1920, 1080)
        run_csv(args.csv, args.out, fps, image_size)

    if args.out_mot:
        from observation import load_observations, write_mot
        n_mot = write_mot(args.out_mot, load_observations(args.out),
                          ball_radius=args.ball_radius)
        print(f"  MOT rows: {n_mot} -> {args.out_mot}")


if __name__ == "__main__":
    main()
