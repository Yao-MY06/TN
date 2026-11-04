# -*- coding: utf-8 -*-
"""球拍检测（阶段 0 的轻量用法）：单向否决不可信的击球事件。

本模块刻意**不实现**「球拍检测 → 认定击球」，原因见
`PADDLE_DETECTION_PLAN.md` 第 2 节的三组实测：

1. COCO 零样本球拍检出率仅 22.6%，无法逐帧使用；
2. 两侧球拍同时可用的帧只占 2.4%，无法判定「谁在击球」；
3. 球-球拍距离在现有 TTNet HIT 上判别力仅 3~7%
   （根因是 TTNet 的 HIT 时间戳本身不可信：球撞台面也会触发）。

但实验同时证明：**检出的那些球拍框是准确的**。
因此唯一安全且有价值的用法是**单向否决**：

    只在「明确看到两侧球拍都远离球」时，否决该时刻的击球候选；
    没检出球拍时不做任何判断，因此不会因漏检而误杀。

这正是本项目反复总结的原则：**证据不足时不表态，证据充分时才否决。**

配套的零成本改进（不依赖检测器）
--------------------------------
TTNet 的 HIT 在「球撞台面」时也会触发。台面弹跳与真实击球的区别是
球的高度：真实击球时球在台面上方（空中），台面弹跳时球贴台。

实测 test_4：以 bounce 落点 y 的 P5 作为「台面高度带」下界，
168 个 HIT 中有 36% 落在台面高度带内 —— 这些是可疑的。
这条判据完全不需要新模型，可直接启用。

用法::

    from referee.paddle_detector import PaddleEvidenceFilter

    f = PaddleEvidenceFilter(table_box=[60*6, 60*8.4375, 187*6, 18*8.4375])
    f.observe_ball(t, ball_xy)          # 每帧喂球位置
    suspicious = f.is_hit_suspicious(t, ball_xy)   # 判断某次 HIT 是否可疑
"""

import math


class PaddleEvidenceFilter:
    """基于球拍与球台几何的「击球可信度」过滤器。

    当前实现只包含**不依赖新模型**的两条判据（见模块 docstring），
    球拍检测接入后由 `set_paddle_source()` 启用第三条。
    """

    def __init__(self, table_box=None, table_margin_ratio=0.06,
                 paddle_far_px=400.0, table_zone_ratio=0.05):
        """
        :param table_box: 球台外接框 [x, y, w, h]（原始像素）。
            可由 TTNet 的分割结果直接给出。
        :param table_margin_ratio: 球台 ROI 外扩比例
        :param paddle_far_px: 「球拍明显远离球」的距离阈值。
            实验 2 中真实击球时球-拍距离中位 963px（受 TTNet HIT 时间戳
            误差污染），因此取 400px 是保守值：只有明确很远才否决。
        :param table_zone_ratio: 判定「台面高度带」时取的 bounce 落点分位。
            用 P5 作为下界，避免极端值影响。
        """
        self.table_box = list(table_box) if table_box else None
        self.table_margin_ratio = table_margin_ratio
        self.paddle_far_px = paddle_far_px
        self.table_zone_ratio = table_zone_ratio

        self._bounce_ys = []          # 已知落台的 y 坐标，用于估计台面高度带
        self._table_y_lo = None       # 台面高度带下界（y 越大越靠下）

        self._paddle_frames = {}      # ts -> [(bbox, side)]

        self.stats = {
            'hit_total': 0,
            'rejected_on_table': 0,     # 因球在台面高度带内而否决
            'rejected_paddle_far': 0,   # 因两侧球拍都远离球而否决
            'paddle_checked': 0,        # 实际做了球拍判断的次数
        }

    # ------------------------------------------------------------ 输入
    def observe_bounce(self, ts, position_px):
        """喂入一个已确认的落台点，用于在线估计台面高度带。"""
        if position_px:
            self._bounce_ys.append(float(position_px[1]))
            self._refresh_table_zone()

    def _refresh_table_zone(self):
        """用落点 y 分位估计台面高度带下界。"""
        if len(self._bounce_ys) < 30:
            return
        ys = sorted(self._bounce_ys)
        self._table_y_lo = ys[int(len(ys) * self.table_zone_ratio)]

    def set_table_zone(self, y_lo):
        """直接指定台面高度带下界（已知球台几何时更准）。"""
        self._table_y_lo = float(y_lo)

    def set_paddle_source(self, paddle_frames):
        """注入球拍检测结果，启用第三条判据。

        :param paddle_frames: {timestamp_s: [(bbox_xyxy, side), ...]}
            来自 PaddleDetector + PaddleTracker 的输出。
            未注入时，球拍判据自动跳过（不表态）。
        """
        self._paddle_frames = paddle_frames or {}

    # ------------------------------------------------------------ 判据
    def is_hit_suspicious(self, ts, ball_xy, tolerance_s=0.10):
        """判断某次 HIT 是否可疑。

        :return: (suspicious: bool, reasons: list[str])
            reasons 用于审计 —— 必须能说清「为什么否决」，否则无法调参。
        """
        self.stats['hit_total'] += 1
        reasons = []

        # ---- 判据 A：球在台面高度带内 → 更可能是台面弹跳，不是击球
        #
        # 真实击球时球在台面上方的空中；球撞台面时球贴台。
        # 这是纯粹的几何事实，不需要新模型。
        if (self._table_y_lo is not None and ball_xy is not None
                and float(ball_xy[1]) >= self._table_y_lo):
            reasons.append('ball_in_table_zone')

        # ---- 判据 B：明确看到两侧球拍都远离球 → 不可能是击球
        #
        # 只在**确实检出了球拍**时才判断；没检出就跳过（不表态）。
        # 这是零样本检测唯一能被安全利用的方式。
        near = self._nearest_paddle(ts, ball_xy, tolerance_s)
        if near is not None:
            self.stats['paddle_checked'] += 1
            if near > self.paddle_far_px:
                reasons.append('all_paddles_far(%.0fpx)' % near)

        if 'ball_in_table_zone' in reasons:
            self.stats['rejected_on_table'] += 1
        if any(r.startswith('all_paddles_far') for r in reasons):
            self.stats['rejected_paddle_far'] += 1

        return bool(reasons), reasons

    def _nearest_paddle(self, ts, ball_xy, tolerance_s):
        """返回 ts 附近球到最近球拍框的距离；无可用数据时返回 None。"""
        if not self._paddle_frames or ball_xy is None:
            return None
        best = None
        for t, boxes in self._paddle_frames.items():
            if abs(t - ts) > tolerance_s:
                continue
            for bbox, _side in boxes:
                d = _point_box_distance(ball_xy, bbox)
                best = d if best is None else min(best, d)
        return best


def _point_box_distance(pt, bbox):
    """点到矩形的最短距离；点在框内返回 0。"""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    dx = max(x1 - pt[0], 0.0, pt[0] - x2)
    dy = max(y1 - pt[1], 0.0, pt[1] - y2)
    return math.hypot(dx, dy)
