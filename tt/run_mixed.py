"""
run_mixed.py
============
混合事件源运行器：**轨迹层沿用对方实现（推 RACKET_HIT），落台改用我方 detect_bounce 结果**。

背景
----
对方交付的 ``run_state_machine.py`` 自带 ``TrajectoryEventProcessor``，能同时推出
``RACKET_HIT`` 与 ``BOUNCE``。实测（test_2, 3600 帧）：

* 它的 RACKET_HIT 40 条 —— 可用，我方模型做不出来（击球瞬间被遮挡，速度不可测）
* 它的 BOUNCE     23 条 / F1 0.667 —— 检出率偏低
* 我方 detect_bounce 34 条 / F1 0.774，且 23 条全部落在它那 23 条里

所以最优组合是：击球用它、落台用我方的。

做法
----
不改对方任何源文件，只在外部包一层 ``MixedEventSource``：

* ``inner.update(obs)`` 照常推进轨迹并产出事件
* 丢弃其中的 ``BOUNCE``（保留 RACKET_HIT / NET_TOUCH / OUT）
* 把第 i 个我方 BOUNCE 在 ``obs.timestamp_s`` 越过它的时间戳时注入

用法::

    python run_mixed.py \
        --jsonl  ../adapter/examples/test_2_observations.jsonl \
        --bounces ../adapter/examples/test_2_events.jsonl \
        --preset test2 \
        --video  ../_data/test_2_1080p_crf22.mp4 \
        --output ../outputs/<今天>/video/test_2_mixed_hud.mp4

``outputs/`` 按日期归档，具体结构见 ``outputs/README.md``。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
SM_ROOT = os.path.join(HERE, "_unpacked")
sys.path.insert(0, SM_ROOT)

from core.geometry import ROIBox, TableCalibration, preset  # noqa: E402
from core.homography import TableHomography  # noqa: E402
from core.schemas import (  # noqa: E402
    BallObservation,
    EventType,
    FrameObservation,
    RefereeEvent,
)
from core.state_machine import MatchRules, MatchState, TableTennisStateMachine  # noqa: E402
from core.trajectory_processor import TrajectoryEventProcessor  # noqa: E402
from visualizer.broadcast_hud import BroadcastHUD, HUDState  # noqa: E402

# 同一帧内的事件顺序：击球必然发生在落台之前
_TYPE_ORDER = {
    EventType.RACKET_HIT.value: 0,
    EventType.NET_TOUCH.value: 1,
    EventType.BOUNCE.value: 2,
    EventType.OUT.value: 3,
}

# 交付件内置预设只有 game2 / rally / plos / test4 / test2，**没有 game_4**。
# game_4 也不是 game_2 的机位（球台在画面里整体低 60~130px、机身更远），
# 借用 game2 会让 net_x 偏 27px、球台多边形上移，落台侧和出界全错。
# 因此这里用我方 calib/table_quads.json 的 game_4 自动拟合四角现场构造一个。
_QUAD_PATH = os.path.join(HERE, "..", "calib", "table_quads.json")


def build_calib(name: str) -> TableCalibration:
    if name != "game4":
        return preset(name)
    import json

    with open(_QUAD_PATH, encoding="utf-8") as f:
        quad = json.load(f)["game_4"]["points"]
    (flx, fly), (frx, _fry), (_nrx, nry), (nlx, nly) = quad
    net_x = int(round(((flx + frx) / 2.0 + (nlx + _nrx) / 2.0) / 2.0))
    return TableCalibration(
        name="game4",
        base_width=1920,
        base_height=1080,
        table_polygon=tuple(map(tuple, quad)),
        roi_box=ROIBox(y_min=250, y_max=920, x_min=150, x_max=1750),
        net_x=net_x,
        # 与内置 game2 预设的取法一致：上沿约为远边上方 70px，下沿略高于近边
        net_zone=(int(fly) - 70, int(max(nry, nly)) - 12),
    )


def mask_low_confidence(
    observations: List[FrameObservation], min_conf: float
) -> tuple[List[FrameObservation], int]:
    """把低置信检出改写成"本帧无检测"，**只给轨迹层用**。

    为什么需要
    ----------
    TrackNet 的后处理（霍夫圆兜底）偶尔会返回一个不在球上的圆：热图邻域里
    没有高值像素，置信度接近 0，坐标偏移可达 20~30px。这类假点一旦进入轨迹，
    会把速度/方向全带偏 —— game_4 52.833s 的 conf=0.000 点 y 偏 26px，正是
    53.25s 那次 SERVE_FAULT 误判的起点。

    为什么不直接丢进 OUT 判定
    ------------------------
    球出界之后飞得远、目标变小、运动模糊重，模型置信度会**自然衰减**到 0.5
    以下，而那时的位置恰恰是"出界"这件事唯一的证据（game_4 45.342s 的 OUT
    就依赖台外那几帧低置信点）。所以这里只做一份"轨迹视角"的副本，
    出界判据仍然看原始观测。

    实测阈值可分性：真球置信度只有 0.51 / 0.749 / 0.906 三档，
    假点集中在 0 ~ 0.43，0.5 是干净的分割线。
    """
    if min_conf <= 0:
        return observations, 0
    out: List[FrameObservation] = []
    n = 0
    for o in observations:
        c = o.ball.confidence
        if o.ball.visible and c is not None and c < min_conf:
            n += 1
            out.append(
                FrameObservation(
                    frame_id=o.frame_id,
                    timestamp_s=o.timestamp_s,
                    source=o.source,
                    image_size=o.image_size,
                    ball=BallObservation(visible=False),
                    event_scores=o.event_scores,
                )
            )
        else:
            out.append(o)
    return out, n


def find_dropouts(observations: List[FrameObservation]) -> List[tuple[float, float]]:
    """列出所有"连续漏检段"，用 (起点时刻, 恢复时刻) 表示。

    起点是最后一个**可见**帧的时间，终点是下一个可见帧的时间，两者之差就是
    这段盲区的时长。发球守卫要用它判断"这块区域当时到底看没看清"。
    """
    spans: List[tuple[float, float]] = []
    last_visible: Optional[float] = None
    gap_start: Optional[float] = None
    for o in observations:
        if o.ball.visible:
            if gap_start is not None:
                spans.append((gap_start, o.timestamp_s))
                gap_start = None
            last_visible = o.timestamp_s
        else:
            if last_visible is not None and gap_start is None:
                gap_start = last_visible
    if gap_start is not None and observations:
        spans.append((gap_start, observations[-1].timestamp_s))
    return spans


class MixedEventSource:
    """把对方的轨迹事件流与外部注入的 BOUNCE 合成一条有序事件流。"""

    def __init__(
        self,
        inner,
        external_bounces: List[RefereeEvent],
        drop_inner: bool = True,
        delay_s: float = 0.0,
        sm=None,
        dropouts: Optional[List[tuple[float, float]]] = None,
        serve_guard_lookback: float = 0.45,
        serve_guard_max_hole: float = 0.08,
    ):
        self.inner = inner
        self.pending = list(external_bounces)
        self.drop_inner = drop_inner
        # 外部事件注入延迟：对方轨迹层的击球检测要看到 vx 反转后几帧才触发，
        # 时间戳系统性滞后 0~17ms（game_4 实测）。外部落台到点立即注入会抢在
        # 击球前面，状态机看到的是"上一拍的击球方 + 落台落在自己半台"→ OWN_COURT。
        # 延迟释放让顺序自动变成 击球→落台。检测时间不变，只有被击球超过时
        # 才把传递时间钳到最新时刻（见下方 last_released_t）。
        self.delay_s = delay_s
        # 已经交给状态机的最大时间戳。状态机要求事件时间戳单调不减
        # （state_machine.py:223 `event timestamp went backwards`），延迟注入时
        # 落台的原时间戳可能已被更晚送达的击球超过（game_4 13.900 vs 13.917）。
        self.last_released_t = -1.0
        self.stats: Dict[str, int] = Counter()
        self.injected: List[RefereeEvent] = []
        # 判罚后的锁定窗口：同一回合已经判过分，之后的落台不再注入。
        # 挡住的是"死球后球在台面上颠/滚"被规则层反复识别成落台
        # （test_2 的 9.6/9.8/10.0s 三连就是这种情况）。
        self.lockout_until = -1.0
        # OUT 的独立锁定窗口。判罚之后 2s 内出现的外来 OUT 事件，绝大多数是
        # **同一颗球**继续飞出边界（回合已在上一条落台/未接住时结案），
        # 放进去会给同一分判两次。BOUNCE 不能用这个窗口——它会吞掉下一回合的
        # 真实落台（test_2 上实测 F1 0.774→0.6）。
        self.out_lockout_until = -1.0
        # ---- 发球守卫（SERVE_FAULT 的证据链检查）----
        # 状态机判 SERVE_FAULT 的依据是"己方首跳没出现"，而这是**缺失证据**，
        # 不是**反证**：只要发球方半台那一段模型没看清，首跳就完全可能漏检。
        # 所以判罚前先查发球窗口里的盲区，盲区超过阈值就降级为不判分。
        self.sm = sm
        self.dropouts = dropouts or []
        self.serve_guard_lookback = serve_guard_lookback
        self.serve_guard_max_hole = serve_guard_max_hole
        self.suppressed_serve: List[RefereeEvent] = []

    def _longest_hole(self, t0: float, t1: float) -> float:
        """[t0, t1] 区间内最长的一段连续漏检时长（秒）。"""
        best = 0.0
        for a, b in self.dropouts:
            lo, hi = max(a, t0), min(b, t1)
            if hi > lo:
                best = max(best, hi - lo)
        return best

    def _serve_fault_unsafe(self, ev: RefereeEvent) -> bool:
        """这条落台会不会被状态机判成 SERVE_FAULT，而证据链又有洞？"""
        sm = self.sm
        if sm is None or self.serve_guard_max_hole <= 0:
            return False
        if sm.state != MatchState.SERVE_IN_PROGRESS or sm._ctx.serve_own_bounce_seen:
            return False
        if ev.side == sm._ctx.server:
            return False        # 合法首跳，不涉及 SERVE_FAULT
        # 窗口从"发球击球时刻再往前回溯"起算：轨迹层给的击球时刻本身可能就是
        # 假点带出来的（game_4 52.850s 的击球紧跟在 conf=0.000 假点之后），
        # 只盯 [击球, 落台] 会漏掉真实的盲区（实测那段洞在 52.55~52.83s，
        # 全部落在击球之前）。
        hit_t = sm._ctx.last_event_time or ev.timestamp_s
        return self._longest_hole(hit_t - self.serve_guard_lookback, ev.timestamp_s) >= self.serve_guard_max_hole

    def update(self, obs: FrameObservation) -> List[RefereeEvent]:
        events = list(self.inner.update(obs))

        if self.drop_inner:
            kept = []
            for ev in events:
                if ev.type == EventType.BOUNCE.value:
                    self.stats["inner_bounce_dropped"] += 1
                else:
                    kept.append(ev)
            events = kept

        # 内层事件（击球）的时间戳就是当前帧时间、天然单调，但也必须计入
        # last_released_t —— 否则外部落台的时间戳钳制会漏掉它们（monotonic 断言）。
        for ev in events:
            self.last_released_t = max(self.last_released_t, ev.timestamp_s)

        t = obs.timestamp_s
        while self.pending and self.pending[0].timestamp_s + self.delay_s <= t + 1e-9:
            ev = self.pending.pop(0)
            if ev.type == EventType.BOUNCE.value and ev.timestamp_s < self.lockout_until:
                self.stats["bounce_suppressed_by_lockout"] += 1
                continue
            if ev.type == EventType.OUT.value and ev.timestamp_s < self.out_lockout_until:
                self.stats["out_suppressed_by_point_lockout"] += 1
                continue
            if ev.type == EventType.BOUNCE.value and self._serve_fault_unsafe(ev):
                # 证据链有洞 → 不提交这条落台，状态机因此不会判 SERVE_FAULT，
                # 之后按 SERVE_INCOMPLETE 超时收尾（不判分）。
                self.stats["serve_fault_suppressed"] += 1
                hit_t = self.sm._ctx.last_event_time or ev.timestamp_s
                self.suppressed_serve.append(
                    (ev, self._longest_hole(hit_t - self.serve_guard_lookback, ev.timestamp_s))
                )
                # 同时把回合上下文清空：这条落台被吞掉后若什么都不做，下一拍
                # 的对打落台会被当成"发球首跳"，后面整轮判罚全部错位。
                self.sm._reset_rally(keep_server=True)
                continue
            if ev.timestamp_s < self.last_released_t:
                # 落台的**检测时间**没变，变的是它被交给状态机的时刻。
                # 钳到已送达的最新时刻，差值就是注入延迟（实测 8~17ms）。
                # 我方原始检测结果（bounces JSONL）与 GT 评测不受影响。
                self.stats["timestamp_clamped"] += 1
                ev.timestamp_s = self.last_released_t
            self.last_released_t = max(self.last_released_t, ev.timestamp_s)
            events.append(ev)
            self.injected.append(ev)
            self.stats[f"external_{ev.type.lower()}_injected"] += 1

        events.sort(key=lambda e: (e.timestamp_s, _TYPE_ORDER.get(e.type, 9)))
        return events

    @property
    def trajectory(self):
        return self.inner.trajectory


def load_bounces(path: str, allowed_sides=("A", "B")) -> List[RefereeEvent]:
    """读我方 BOUNCE 事件（已是 §7.4 结构），转成对方的 RefereeEvent。"""
    out: List[RefereeEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d: Dict[str, Any] = json.loads(line)
            if d.get("type") != EventType.BOUNCE.value:
                continue
            if d.get("side") not in allowed_sides:
                continue
            out.append(RefereeEvent.from_dict(d))
    out.sort(key=lambda e: e.timestamp_s)
    return out


def merge_jitter(
    events: List[RefereeEvent], max_dt: float = 0.6, max_dist_px: float = 60.0
) -> tuple[List[RefereeEvent], List[RefereeEvent]]:
    """合并「死球抖动」造成的同侧连击落台。

    球在台面上停住或滚动时，模型会在地上球影 / 台面木纹上反复抖动，规则层每次抖动
    （y 出现局部极大 + 前后 drop 够大）都会报一次落台。物理上不可能：**真落台之后球
    要么被击打（下一次落台在对面半台），要么就再也不会弹这么高**。

    判据：同一半台 + 间隔 < ``max_dt`` + 位置相距 < ``max_dist_px``（球径量级）。

    test_2 全片 34 条里只有 9.600 / 9.800 / 10.017 三条命中，未误伤任何其他落台。
    """
    kept: List[RefereeEvent] = []
    dropped: List[RefereeEvent] = []
    for ev in events:
        if kept:
            prev = kept[-1]
            dt = ev.timestamp_s - prev.timestamp_s
            same_side = ev.side == prev.side
            near = (
                ev.position_px is not None
                and prev.position_px is not None
                and math.dist(ev.position_px, prev.position_px) < max_dist_px
            )
            if same_side and dt < max_dt and near:
                dropped.append(ev)
                continue
        kept.append(ev)
    return kept, dropped


def patch_ball_marker(hud) -> None:
    """把球位标记换成醒目的红点。

    对方的 ``BroadcastHUD._draw_ball`` 里颜色表只认 ``yolo / patch / pred`` 三个 key，
    其他来源一律走 ``_C_DIM``（灰色 150,150,150）。我们传 ``track_method="tracknet"``
    就落到灰色兜底，在 1080p 画面上几乎看不见——而这恰好是 TrackNet 最该被看见的东西。

    不改对方源文件，只在这个实例上覆写方法。
    """
    import cv2

    RED = (60, 60, 255)  # BGR

    def _draw_ball(img, h):
        if h.current_pos is None:
            # 本帧模型没检出：不画点，只在旁边标一下，避免红点"停"在上一帧位置
            return
        x, y = int(round(h.current_pos[0])), int(round(h.current_pos[1]))
        r_out = hud._px(16)
        # 外环 + 中心小点：圈住球而不是盖住球，观众仍能看见球本身
        cv2.circle(img, (x, y), r_out, RED, hud._px(2), cv2.LINE_AA)
        cv2.circle(img, (x, y), hud._px(3), RED, -1, cv2.LINE_AA)
        # 标签画在红圈正上方，先描黑边再压红字，避免压在球拍 logo / 深色球台上看不清
        lx, ly = x - hud._px(36), y - r_out - hud._px(10)
        cv2.putText(img, "TRACKNET", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * hud._s * 1.0, (0, 0, 0), max(2, hud._px(3)), cv2.LINE_AA)
        cv2.putText(img, "TRACKNET", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * hud._s * 1.0, RED, max(1, hud._px(1)), cv2.LINE_AA)

    hud._draw_ball = _draw_ball


def patch_scoreboard(hud) -> None:
    """修比分板上发球指示器与选手名重叠的问题。

    对方 ``_draw_scoreboard`` 把 serve 圆点画在 ``x0+pad``，正好压在 "P1" 文字上
    （服务方为 B 时压 "P2"），叠成 "P1erve"。这里把发球指示整体挪到比分板右上角。

    同样不改对方源文件，只覆写这个实例的方法。
    """
    import cv2
    from visualizer import broadcast_hud as bh

    C_BG, C_BORDER, C_TEXT = bh._C_BG, bh._C_BORDER, bh._C_TEXT
    C_DIM, C_P1, C_P2 = bh._C_DIM, bh._C_P1, bh._C_P2
    C_SERVER, C_EVENT = bh._C_SERVER, bh._C_EVENT

    def _draw_scoreboard(img, h):
        pad = hud._px(14)
        x0, y0 = hud._px(18), hud._px(18)
        w, hh = hud._px(430), hud._px(132)
        x1, y1 = x0 + w, y0 + hh

        cv2.rectangle(img, (x0, y0), (x1, y1), C_BG, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), C_BORDER, hud._px(2))
        hud._put(img, "AI TABLE TENNIS REFEREE", (x0 + pad, y0 + hud._px(26)), 0.52, C_EVENT, 1)

        cy = y0 + hud._px(72)
        a, b = h.score
        hud._put(img, "P1", (x0 + pad, cy), 0.72, C_P1, 2)
        hud._put(img, f"{a}", (x0 + hud._px(92), cy), 1.35, C_TEXT, 3)
        hud._put(img, ":", (x0 + hud._px(148), cy), 1.10, C_DIM, 2)
        hud._put(img, f"{b}", (x0 + hud._px(186), cy), 1.35, C_TEXT, 3)
        hud._put(img, "P2", (x0 + hud._px(250), cy), 0.72, C_P2, 2)

        # 发球方：挪到比分板右上角，避开 P1 / P2 文字
        if h.server in ("A", "B"):
            sx = x0 + w - hud._px(92)
            sy = y0 + hud._px(26)
            pid = "P1" if h.server == "A" else "P2"
            cv2.circle(img, (sx, sy - hud._px(4)), hud._px(5), C_SERVER, -1, cv2.LINE_AA)
            hud._put(img, f"{pid} serve", (sx + hud._px(10), sy), 0.40, C_SERVER, 1)

        state_text = "GAME OVER" if h.game_over else h.state
        state_color = (60, 60, 255) if h.game_over else C_DIM
        hud._put(
            img, f"{state_text}   f{h.frame_id:04d}   {h.timestamp_s:6.2f}s",
            (x0 + pad, y0 + hud._px(108)), 0.42, state_color, 1,
        )

        if h.last_decision and h.decision_age_s < 3.0:
            by = y1 + hud._px(26)
            cv2.rectangle(
                img, (x0, y1 + hud._px(8)), (x0 + hud._px(560), y1 + hud._px(40)), C_BG, -1
            )
            cv2.rectangle(
                img, (x0, y1 + hud._px(8)), (x0 + hud._px(560), y1 + hud._px(40)),
                C_BORDER, hud._px(1),
            )
            hud._put(img, h.last_decision, (x0 + pad, by), 0.52, C_EVENT, 1)

    hud._draw_scoreboard = _draw_scoreboard


def derive_out_events(
    observations: List[FrameObservation],
    calib,
    min_out_px: float = 35.0,
    min_gap_s: float = 0.4,
    need_descending: bool = True,
    bounce_times: Optional[List[float]] = None,
    bounce_guard_s: float = 0.6,
    cooldown_s: float = 1.5,
    stats: Optional[Dict[str, int]] = None,
) -> List[RefereeEvent]:
    """从「球在台外失联」推断出界（OUT）。

    为什么需要这个
    --------------
    状态机包的 ``TrajectoryEventProcessor`` 确实有 OUT 判据
    （``trajectory_processor.py`` 里 ``_is_bounce`` 命中且落点在台外时发出），
    但它被两道前置门卡住：

    1. OUT 挂在落台分支内部，必须先抓到 **y 方向拐点**（球在台外反弹）；
    2. ``_in_out_region`` 还要求落点在图像上位于球台**近边之下**，远端与侧向
       因为透视压缩严重被主动排除。

    球横向飞出边界后直接消失在台外时，两个条件都不满足，事件流里什么都不会发生，
    状态机等满超时只能给 ``NO_EVIDENCE``。本项目 ``detect_bounce`` 同样不覆盖这种情况
    （它只判**台面内**的落台，球落在台外地面物理上也不是落台）。

    判据
    ----
    球最后一次被看到时：

    1. 位置在球台多边形**外侧**超过 ``min_out_px``（用像素距离，约 1.5 个球径）
    2. 此后连续 ``min_gap_s`` 秒没有检出
    3. 失联前 y 在递增（球在下坠，而不是飞越台面上方的高球）
    4. 失联前的 ``bounce_guard_s`` 秒内没有落台事件——球已经在台面上弹过一下，
       之后再飞出去是"对方没接住"，不是出界；这条守卫挡掉的是回合末球弹完离台
       最常见的误判
    5. 与上一条 OUT 至少间隔 ``cooldown_s``（同一颗球飞出边界会被连续判多次）

    归属不在这里决定——事件只报"哪一侧看到的出界"，得分方由状态机按
    ``last_hitter`` 的对手算。
    """
    import cv2
    import numpy as np

    poly = np.array(list(calib.table_polygon), dtype=np.float32)
    events: List[RefereeEvent] = []
    prev = None
    count = 0
    last_out_t = -1e9
    bt = sorted(bounce_times or [])
    stats = stats if stats is not None else {}

    def _recent_bounce(t: float) -> bool:
        # 只回看、不前瞻：失联之后才发生的落台无法作为"这颗球已经弹过"的证据
        return any(t - bounce_guard_s <= b <= t + 1e-9 for b in bt)

    def _try(idx_prev: int, t_resume: float) -> Optional[RefereeEvent]:
        nonlocal count, last_out_t
        o = observations[idx_prev]
        if o.ball.center_px is None:
            return None
        x, y = o.ball.center_px
        signed = cv2.pointPolygonTest(poly, (float(x), float(y)), True)
        if signed > -min_out_px:  # 还在台内或贴着台边，不算出界
            return None
        if need_descending:
            earlier = [
                p.ball.center_px[1]
                for p in observations[max(0, idx_prev - 4):idx_prev + 1]
                if p.ball.visible and p.ball.center_px is not None
            ]
            if len(earlier) < 2 or earlier[-1] <= earlier[0]:
                return None
        t = o.timestamp_s
        if _recent_bounce(t):
            stats["out_suppressed_by_bounce"] = stats.get("out_suppressed_by_bounce", 0) + 1
            return None
        if t - last_out_t < cooldown_s:
            stats["out_suppressed_by_cooldown"] = stats.get("out_suppressed_by_cooldown", 0) + 1
            return None
        last_out_t = t
        count += 1
        return RefereeEvent(
            event_id=f"out_{count:05d}",
            timestamp_s=t,
            type=EventType.OUT.value,
            side="A" if x < calib.net_x else "B",
            position_px=(float(x), float(y)),
            confidence=None,
            sources=["tracknet"],
        )

    for i, o in enumerate(observations):
        if not o.ball.visible:
            continue
        if prev is not None:
            gap = o.timestamp_s - observations[prev].timestamp_s
            if gap >= min_gap_s:
                ev = _try(prev, o.timestamp_s)
                if ev:
                    events.append(ev)
        prev = i

    if prev is not None:
        tail = observations[-1].timestamp_s - observations[prev].timestamp_s
        if tail >= min_gap_s:
            ev = _try(prev, observations[-1].timestamp_s)
            if ev:
                events.append(ev)
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description="混合事件源：对方轨迹层 + 我方落台")
    ap.add_argument("--jsonl", required=True, help="§7.1 逐帧观测 JSONL（我方适配器输出）")
    ap.add_argument("--bounces", required=True, help="我方 BOUNCE 事件 JSONL（§7.4 结构）")
    ap.add_argument("--preset", default="test2")
    ap.add_argument("--video", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--log", default=None, help="判罚日志输出路径")
    ap.add_argument("--events-out", default=None, help="把最终事件流（含注入的落台/出界）写成 JSONL")
    ap.add_argument("--keep-inner-bounce", action="store_true", help="调试用：不丢弃对方内部推的 BOUNCE")
    ap.add_argument(
        "--post-point-lockout",
        type=float,
        default=0.0,
        help="判罚后多少秒内不再注入落台（避免同一回合重复判分），0 = 关闭",
    )
    ap.add_argument("--jitter-merge", action="store_true", help="合并死球抖动导致的同侧连击落台")
    ap.add_argument("--jitter-max-dt", type=float, default=0.6, help="抖动合并的时间窗（秒）")
    ap.add_argument("--jitter-max-dist", type=float, default=60.0, help="抖动合并的位置窗（像素）")
    ap.add_argument(
        "--bounce-delay",
        type=float,
        default=0.07,
        help="外部 BOUNCE/OUT 注入延迟（秒）。对方轨迹层的击球检测滞后 0~17ms，"
        "外部落台到点立即注入会抢在击球前面（状态机误判 OWN_COURT）；延迟释放让顺序"
        "自动变成 击球→落台，事件时间戳本身不改。0 = 关闭。默认 0.07（120fps 下 9 帧）。",
    )
    ap.add_argument("--max-frames", type=int, default=0, help="只渲染前 N 帧（快速预览用），0 = 全片")
    ap.add_argument("--no-out", action="store_true", help="关闭出界事件推断（用于对比）")
    ap.add_argument("--out-min-px", type=float, default=35.0, help="判定台外的最小像素距离（约 1.5 个球径）")
    ap.add_argument("--out-gap-s", type=float, default=0.4, help="失联多久算球已落地（秒）")
    ap.add_argument("--out-allow-ascending", action="store_true", help="不要求失联前球在下坠（放宽）")
    ap.add_argument(
        "--out-bounce-guard",
        type=float,
        default=0.6,
        help="失联前这么多秒内若有落台，则不算出界（球已弹过台面，之后飞出去是对方没接住）",
    )
    ap.add_argument(
        "--out-cooldown",
        type=float,
        default=1.5,
        help="两次出界事件的最小间隔（同一颗球飞出边界会被连续判多次），0 = 关闭",
    )
    ap.add_argument(
        "--out-after-point-guard",
        type=float,
        default=2.0,
        help="判罚后这么多秒内不再注入 OUT（同一颗球继续飞出，不是新的一分），0 = 关闭",
    )
    ap.add_argument(
        "--trail-points",
        type=int,
        default=240,
        help="HUD 尾迹只画最近 N 个轨迹点；对方实现每帧遍历全量历史，3600 帧时是 O(n^2)",
    )
    ap.add_argument(
        "--obs-min-conf",
        type=float,
        default=0.5,
        help="置信度低于该值的检出**不喂给轨迹层**（当作本帧无检测），但出界判据仍看原始观测。"
        "0 = 关闭。默认 0.5：真球置信度只有 0.51/0.749/0.906 三档，假点集中在 0~0.43。",
    )
    ap.add_argument(
        "--serve-guard-lookback",
        type=float,
        default=0.45,
        help="发球守卫向前回溯多久（秒）。必须 >0：轨迹层给出的发球击球时刻可能本身就是"
        "假点带出来的，真实盲区会落在击球之前。",
    )
    ap.add_argument(
        "--serve-guard-max-hole",
        type=float,
        default=0.08,
        help="发球窗口内允许的最长连续漏检（秒）。超过就撤下这条落台、不判 SERVE_FAULT"
        "（因为无法排除己方首跳漏检）。0 = 关闭守卫。",
    )
    args = ap.parse_args()

    for p in (args.jsonl, args.bounces):
        if not os.path.isfile(p):
            print(f"[错误] 找不到文件: {p}")
            return 1

    calib = build_calib(args.preset)
    sm = TableTennisStateMachine(MatchRules())
    homography = TableHomography(calib, image_size=(1920, 1080))

    cap = writer = hud = None
    fps, w, h = 120.0, 1920, 1080
    if args.video and os.path.isfile(args.video):
        import cv2

        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        homography = TableHomography(calib, image_size=(w, h))

    if args.output and cap:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        hud = BroadcastHUD(calib, homography)
        patch_ball_marker(hud)
        patch_scoreboard(hud)

    bounces = load_bounces(args.bounces)
    jitter_dropped: List[RefereeEvent] = []
    if args.jitter_merge:
        bounces, jitter_dropped = merge_jitter(bounces, args.jitter_max_dt, args.jitter_max_dist)

    # 观测全部读进内存：出界判据需要看「球最后出现在哪、之后多久没再出现」，
    # 这是双向依赖，流式处理做不了。3600 帧的内存开销可以忽略。
    with open(args.jsonl, "r", encoding="utf-8") as f:
        observations = [FrameObservation.from_dict(json.loads(line)) for line in f if line.strip()]

    # 轨迹层视角：低置信假点当作无检测（见 mask_low_confidence 的说明）。
    # 出界判据仍用原始 observations —— 球出界后置信度自然衰减，那几帧恰恰是唯一证据。
    track_obs, n_lowconf = mask_low_confidence(observations, args.obs_min_conf)
    dropouts = find_dropouts(track_obs)

    out_events: List[RefereeEvent] = []
    out_stats: Dict[str, int] = {}
    if not args.no_out:
        out_events = derive_out_events(
            observations,
            calib,
            args.out_min_px,
            args.out_gap_s,
            not args.out_allow_ascending,
            bounce_times=[e.timestamp_s for e in bounces],
            bounce_guard_s=args.out_bounce_guard,
            cooldown_s=args.out_cooldown,
            stats=out_stats,
        )

    externals = sorted(bounces + out_events, key=lambda e: e.timestamp_s)
    print("=" * 74)
    print("【混合事件源】对方轨迹层(RACKET_HIT) + 我方 detect_bounce(BOUNCE)")
    print(f"观测文件 : {os.path.basename(args.jsonl)}")
    print(f"落台事件 : {os.path.basename(args.bounces)}  ({len(bounces)} 条)")
    if out_events:
        for e in out_events:
            print(f"出界事件 : {e.timestamp_s:6.3f}s  side={e.side}  pos=({e.position_px[0]:.0f}, {e.position_px[1]:.0f})")
    if jitter_dropped:
        ts = ", ".join(f"{e.timestamp_s:.3f}s" for e in jitter_dropped)
        print(f"抖动合并 : 丢弃 {len(jitter_dropped)} 条同侧连击 ({ts})")
    if out_stats:
        print(
            f"出界守卫 : 疑似出界但前 {args.out_bounce_guard}s 内有落台 → 丢弃 "
            f"{out_stats.get('out_suppressed_by_bounce', 0)} 条；"
            f"冷却窗内重复 → 丢弃 {out_stats.get('out_suppressed_by_cooldown', 0)} 条"
        )
    print(f"球台预设 : {args.preset} (网 X = {calib.net_x}px)")
    if n_lowconf:
        print(f"低置信过滤 : {n_lowconf} 帧置信度 < {args.obs_min_conf:.2f}，轨迹层视为无检测（出界判据不受影响）")
    print("=" * 74)

    log_lines: List[str] = []
    counts: Counter = Counter()
    decisions: List[str] = []
    dumped_events: List[Dict[str, Any]] = []
    landings = []
    last_text, last_t = None, -999.0
    source = None
    n_obs = 0

    for obs in track_obs:
        if args.max_frames and n_obs >= args.max_frames:
            break
        n_obs += 1

        if source is None:
            src = obs.source.value if hasattr(obs.source, "value") else str(obs.source)
            inner = TrajectoryEventProcessor(calib, source=src)
            source = MixedEventSource(
                inner,
                externals,
                drop_inner=not args.keep_inner_bounce,
                delay_s=args.bounce_delay,
                sm=sm,
                dropouts=dropouts,
                serve_guard_lookback=args.serve_guard_lookback,
                serve_guard_max_hole=args.serve_guard_max_hole,
            )

        t = obs.timestamp_s
        for d in sm.tick(t):
            last_text, last_t = d.describe(), t
            decisions.append(f"[{t:6.3f}s] {d.describe()}")
            log_lines.append(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] >> 判罚: {d.describe()}")
            print(log_lines[-1])
            source.out_lockout_until = t + args.out_after_point_guard
            if args.post_point_lockout > 0:
                source.lockout_until = t + args.post_point_lockout

        for ev in source.update(obs):
            counts[ev.type] += 1
            if ev.position_px is not None:
                pos = (round(float(ev.position_px[0]), 1), round(float(ev.position_px[1]), 1))
            else:
                pos = None
            dumped_events.append(
                {
                    "event_id": ev.event_id,
                    "timestamp_s": round(float(ev.timestamp_s), 6),
                    "type": ev.type,
                    "side": ev.side,
                    "position_px": list(pos) if pos else None,
                    "confidence": ev.confidence,
                }
            )
            if ev.type == EventType.BOUNCE.value and ev.position_px is not None:
                landings.append(homography.locate_landing_from_event(ev))
            log_lines.append(
                f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] 事件: {ev.type:<11} 半台: {ev.side:<7} ID: {ev.event_id}"
            )
            print(log_lines[-1])
            for d in sm.update(ev):
                last_text, last_t = d.describe(), t
                decisions.append(f"[{t:6.3f}s] {d.describe()}")
                log_lines.append(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] >> 判罚: {d.describe()}")
                print(log_lines[-1])
                source.out_lockout_until = t + args.out_after_point_guard
                if args.post_point_lockout > 0:
                    source.lockout_until = t + args.post_point_lockout

        if cap and writer:
            ok, frame = cap.read()
            if ok:
                # 红点画的是**本帧模型检出位置**，不是轨迹最后一点——
                # 轨迹最后一点可能是卡尔曼外推的预测值，画成红点会让人误以为
                # "模型在这里真的看到了球"。漏检就让它消失，这才是能力边界。
                cur = obs.ball.center_px if obs.ball.visible else None
                hud_state = HUDState(
                    frame_id=obs.frame_id,
                    timestamp_s=t,
                    fps=fps,
                    score=(sm.points["A"], sm.points["B"]),
                    state=sm.state.value,
                    server=sm.server,
                    game_over=sm.game_over,
                    last_decision=last_text,
                    decision_age_s=t - last_t,
                    track_points=source.trajectory[-args.trail_points:],
                    landings=landings[-hud.landing_history:],
                    current_pos=cur,
                    track_method="tracknet",
                )
                writer.write(hud.render(frame, hud_state))

    if cap:
        cap.release()
    if writer:
        writer.release()
        print(f"\n[渲染完成] {args.output}")

    print("\n" + "=" * 74)
    print("评测完成")
    print(f"观测帧数        : {n_obs}")
    print(f"事件统计        : {dict(counts)}")
    print(f"事件源统计      : {dict(source.stats)}")
    print(f"未注入的我方落台: {len(source.pending)} 条（超出视频时长）")
    if source.suppressed_serve:
        print(f"发球守卫拦截    : {len(source.suppressed_serve)} 条落台被撤下（发球窗口内有盲区，"
              f"不判 SERVE_FAULT）")
        for ev, hole in source.suppressed_serve:
            print(f"   - {ev.timestamp_s:6.3f}s  side={ev.side}  窗口内最长盲区 {hole:.3f}s")
    print(f"最终比分        : P1(A) {sm.points['A']} : {sm.points['B']} P2(B)")
    print(f"判罚条数        : {len(decisions)}")
    for d in decisions:
        print(f"   - {d}")
    print("=" * 74)

    if args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"日志已写入: {args.log}")

    if args.events_out:
        with open(args.events_out, "w", encoding="utf-8") as f:
            for ev in dumped_events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print(f"事件流已写入: {args.events_out}  ({len(dumped_events)} 条)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
