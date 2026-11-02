"""统一观测（Observation）/ 事件（Event）数据结构与校验。

三个模型（YOLO / TrackNet / TTNet）的适配器都必须输出这里定义的 Observation，
之后由统一的轨迹、事件处理层生成 Event，再交给共用计分状态机。

字段定义见 TTNet_TrackNet_YOLO_共用状态机方案.md 第 7 节：
    observation = {
        "frame_id": 1234,
        "timestamp_s": 20.567,        # 被预测帧的视频时间，不是推理完成时间
        "source": "tracknet",         # yolo / tracknet / ttnet
        "image_size": [1920, 1080],   # 原始图像宽、高
        "ball": {
            "visible": True,
            "center_px": [850.0, 420.0],   # 原始图像坐标，左上角原点
            "bbox_xyxy": None,
            "confidence": 0.92,            # 模型原始分数；无可靠分数填 None
        },
        "event_scores": {"bounce": None, "net_hit": None},
    }
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

SOURCES = ("yolo", "tracknet", "ttnet")
# 事件层枚举。命名取 YOLO 分支的暂定值：裸 HIT 无法区分 racket hit 与 net hit，
# 而观测层的 event_scores.net_hit 已经在用 net_hit 这个名字，故事件层写成
# RACKET_HIT / NET_TOUCH 以免混淆。注意两层的对应关系：
#   event_scores.bounce    <-> EVENT type BOUNCE
#   event_scores.net_hit   <-> EVENT type NET_TOUCH
#   （没有击球分数的槽：RACKET_HIT 只能由事件层从轨迹推断，模型不直接给分）
EVENT_TYPES = ("BOUNCE", "RACKET_HIT", "NET_TOUCH", "OUT")
SIDES = ("A", "B", "UNKNOWN")


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #
def make_observation(
    frame_id: int,
    timestamp_s: float,
    source: str,
    image_size,
    center_px: Optional[Iterable[float]] = None,
    confidence: Optional[float] = None,
    bbox_xyxy: Optional[Iterable[float]] = None,
    bounce_score: Optional[float] = None,
    net_hit_score: Optional[float] = None,
    predicted: bool = False,
) -> Dict[str, Any]:
    """构造一条统一观测。

    center_px 为 None 表示这一帧没有检测到球 —— 此时 visible=False，
    并且 center / bbox / confidence 一律置 None（不填上一帧位置，也不用 (0,0)）。

    predicted=True 表示 center_px 是**轨迹推断出来的位置**，不是模型看到的
    （方案文档 7.3：轨迹预测要明确标记，避免被误认为真实检测）。此时
    confidence 必须为 None —— 没有检测就没有分数。
    """
    visible = center_px is not None
    if not visible:
        bbox_xyxy = None
        confidence = None
        predicted = False
    elif predicted:
        confidence = None

    return {
        "frame_id": int(frame_id),
        "timestamp_s": round(float(timestamp_s), 6),
        "source": source,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "ball": {
            "visible": bool(visible),
            "center_px": [round(float(center_px[0]), 2), round(float(center_px[1]), 2)]
            if visible
            else None,
            "bbox_xyxy": [round(float(v), 2) for v in bbox_xyxy] if bbox_xyxy is not None else None,
            "confidence": None if confidence is None else round(float(confidence), 4),
            "predicted": bool(predicted),
        },
        "event_scores": {
            "bounce": None if bounce_score is None else round(float(bounce_score), 4),
            "net_hit": None if net_hit_score is None else round(float(net_hit_score), 4),
        },
    }


def make_event(
    event_id: str,
    timestamp_s: float,
    event_type: str,
    side: str = "UNKNOWN",
    position_px: Optional[Iterable[float]] = None,
    confidence: Optional[float] = None,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """构造事件（由事件处理层调用，不是适配器的输出）。"""
    return {
        "event_id": event_id,
        "timestamp_s": round(float(timestamp_s), 6),
        "type": event_type,
        "side": side,
        "position_px": None
        if position_px is None
        else [round(float(position_px[0]), 2), round(float(position_px[1]), 2)],
        "confidence": None if confidence is None else round(float(confidence), 4),
        "sources": list(sources or []),
    }


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_observation(obs: Dict[str, Any]) -> List[str]:
    """返回错误列表；空列表表示通过。"""
    errs: List[str] = []

    for k in ("frame_id", "timestamp_s", "source", "image_size", "ball", "event_scores"):
        if k not in obs:
            errs.append(f"missing field: {k}")
    if errs:
        return errs

    if not isinstance(obs["frame_id"], int) or obs["frame_id"] < 0:
        errs.append("frame_id must be a non-negative int")
    if not _is_num(obs["timestamp_s"]) or obs["timestamp_s"] < 0:
        errs.append("timestamp_s must be a non-negative number")
    if obs["source"] not in SOURCES:
        errs.append(f"source must be one of {SOURCES}, got {obs['source']!r}")

    size = obs["image_size"]
    if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(_is_num(v) and v > 0 for v in size)):
        errs.append("image_size must be [width, height] with positive numbers")
        size = None

    ball = obs["ball"]
    if not isinstance(ball, dict):
        errs.append("ball must be an object")
        return errs
    for k in ("visible", "center_px", "bbox_xyxy", "confidence"):
        if k not in ball:
            errs.append(f"ball missing field: {k}")
    if errs:
        return errs

    if not isinstance(ball["visible"], bool):
        errs.append("ball.visible must be bool")

    # predicted 是可选字段（方案文档 7.3）：缺省视为 False，便于队友的适配器
    # 先不实现轨迹补帧也能通过校验。
    pred = ball.get("predicted", False)
    if not isinstance(pred, bool):
        errs.append("ball.predicted must be bool")
        pred = False
    if pred and not ball["visible"]:
        errs.append("ball.predicted must be False when visible=False")

    if ball["visible"]:
        c = ball["center_px"]
        if not (isinstance(c, (list, tuple)) and len(c) == 2 and all(_is_num(v) for v in c)):
            errs.append("ball.center_px must be [x, y] when visible")
        else:
            if c == [0, 0]:
                errs.append("ball.center_px must not be (0, 0) — use visible=False instead")
            if size is not None:
                w, h = size
                if not (0 <= c[0] <= w and 0 <= c[1] <= h):
                    errs.append(f"ball.center_px outside image bounds {c}")
        if ball["confidence"] is not None and not (
            _is_num(ball["confidence"]) and 0.0 <= ball["confidence"] <= 1.0
        ):
            errs.append("ball.confidence must be None or a number in [0, 1]")
        if ball["bbox_xyxy"] is not None:
            b = ball["bbox_xyxy"]
            if not (isinstance(b, (list, tuple)) and len(b) == 4 and all(_is_num(v) for v in b)):
                errs.append("ball.bbox_xyxy must be None or [x1, y1, x2, y2]")
        if pred and ball["confidence"] is not None:
            errs.append("ball.confidence must be None when predicted=True "
                        "(a trajectory estimate has no detection score)")
    else:
        for k in ("center_px", "bbox_xyxy", "confidence"):
            if ball[k] is not None:
                errs.append(f"ball.{k} must be None when visible=False")

    ev = obs["event_scores"]
    if not isinstance(ev, dict):
        errs.append("event_scores must be an object")
    else:
        for k in ("bounce", "net_hit"):
            if k not in ev:
                errs.append(f"event_scores missing field: {k}")
            elif ev[k] is not None and not (_is_num(ev[k]) and 0.0 <= ev[k] <= 1.0):
                errs.append(f"event_scores.{k} must be None or a number in [0, 1]")

    return errs


def validate_event(event: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for k in ("event_id", "timestamp_s", "type", "side", "position_px", "confidence", "sources"):
        if k not in event:
            errs.append(f"missing field: {k}")
    if errs:
        return errs
    if event["type"] not in EVENT_TYPES:
        errs.append(f"type must be one of {EVENT_TYPES}")
    if event["side"] not in SIDES:
        errs.append(f"side must be one of {SIDES}")
    if not _is_num(event["timestamp_s"]) or event["timestamp_s"] < 0:
        errs.append("timestamp_s must be a non-negative number")
    return errs


# --------------------------------------------------------------------------- #
# 写出
# --------------------------------------------------------------------------- #
def write_observations(path: str, observations: Iterable[Dict[str, Any]], strict: bool = True) -> int:
    """写 JSONL（每帧一行）。strict=True 时逐条校验，不通过直接抛异常。"""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for obs in observations:
            if strict:
                errs = validate_observation(obs)
                if errs:
                    raise ValueError(f"invalid observation at frame {obs.get('frame_id')}: {errs}")
            f.write(json.dumps(obs, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_observations(path: str) -> List[Dict[str, Any]]:
    """读回 JSONL，并做一次统一校验。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obs = json.loads(line)
            errs = validate_observation(obs)
            if errs:
                raise ValueError(f"{path}:{line_no}: {errs}")
            out.append(obs)
    return out


# --------------------------------------------------------------------------- #
# MOT Challenge 交换格式（业界事实标准，py-motmetrics / supervision / trackers 都能吃）
# --------------------------------------------------------------------------- #
# 每行 10 个字段，逗号分隔，**帧号从 1 开始**：
#   frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
# 2D 场景下 x/y/z 一律 -1。漏检帧不写行（该格式只表示"这一帧有这个目标"）。
# 参考：MOTChallenge 2015, arXiv:1504.01942, Table II
#
# 为什么要有这个：我们自己的 JSONL 信息更全（可见性、事件分数、时间戳），
# 但不通用。导出 MOT 之后可以直接用现成工具算 MOTA / IDF1 / MOTP，
# 也能被 supervision / trackers / motrackers 直接读取，不用自己写评估。

def _mot_radius(image_size, ball_radius: Optional[float]) -> float:
    """球没有 bbox 时用中心点 + 固定半径构造一个框。

    默认取图像宽度的 1.2%：1920 宽的视频 -> 半径 23px、直径 46px，
    与 OpenTTGames 实测球径（约 47px @1920）基本一致。
    """
    if ball_radius is not None:
        return float(ball_radius)
    return max(2.0, 0.012 * float(image_size[0]))


def obs_to_mot(obs: Dict[str, Any], track_id: int = 1,
               ball_radius: Optional[float] = None):
    """一条 Observation -> MOT 十元组；漏检返回 None。"""
    ball = obs["ball"]
    if not ball["visible"]:
        return None

    if ball["bbox_xyxy"] is not None:
        x1, y1, x2, y2 = [float(v) for v in ball["bbox_xyxy"]]
    else:
        r = _mot_radius(obs["image_size"], ball_radius)
        cx, cy = [float(v) for v in ball["center_px"]]
        x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r

    conf = ball["confidence"]
    return (
        int(obs["frame_id"]) + 1,          # MOT 是 1-based，我们是 0-based
        int(track_id),
        round(x1, 2),
        round(y1, 2),
        round(x2 - x1, 2),
        round(y2 - y1, 2),
        1.0 if conf is None else round(float(conf), 4),
        -1, -1, -1,                        # 2D：无世界坐标
    )


def write_mot(path: str, observations: Iterable[Dict[str, Any]],
              track_id: int = 1, ball_radius: Optional[float] = None,
              include_predicted: bool = False) -> int:
    """导出 MOT Challenge txt。返回写出的行数（= 有检测的帧数）。

    默认**跳过 predicted=True 的帧**：MOT 格式用来评估“模型看到了什么”，
    把轨迹推断出来的位置写进去会把指标算高（而且它没有真实分数）。
    需要包含时显式传 include_predicted=True。
    """
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        for obs in observations:
            if obs["ball"].get("predicted", False) and not include_predicted:
                continue
            row = obs_to_mot(obs, track_id, ball_radius)
            if row is None:
                continue
            f.write(",".join(str(v) for v in row) + "\n")
            n += 1
    return n


def mot_file_for(seq_name: str, out_dir: str) -> str:
    """MOT 评估约定：结果文件名必须是 <seq>.txt。"""
    return os.path.join(out_dir, f"{seq_name}.txt")
