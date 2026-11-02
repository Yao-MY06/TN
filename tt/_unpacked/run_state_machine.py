"""
run_state_machine.py
====================
乒乓球视觉智能裁判 · 状态机与事件处理层运行入口

功能：
  1. 读取任何符合组长方案 §7.1 规范的逐帧观测 JSONL 文件（TrackNet / YOLO / TTNet 均可）；
  2. 运行轨迹处理（时钟驱动卡尔曼平滑 + 双帧确认 + 物理事件提取）；
  3. 驱动 6 状态乒乓球计分状态机，实时输出判罚日志与最终比分；
  4. （可选）传入原始视频，同步渲染带转播级比分看板（HUD）的 MP4 录像。

用法示例：
  # 1. 基础用法（纯命令行运行，0.1秒出比分）：
  python run_state_machine.py --jsonl examples/sample_test2_video_600f.jsonl --preset test2

  # 2. 进阶用法（同时渲染带左上角比分看板的视频）：
  python run_state_machine.py \
      --jsonl examples/sample_test2_video_600f.jsonl \
      --preset test2 \
      --video path/to/test_2.mp4 \
      --output output_referee.mp4
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import cv2

# 保证当前包在 python 检索路径上
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.geometry import preset
from core.homography import TableHomography
from core.schemas import EventType, FrameObservation
from core.state_machine import MatchRules, TableTennisStateMachine
from core.trajectory_processor import TrajectoryEventProcessor
from visualizer.broadcast_hud import BroadcastHUD, HUDState


def main():
    parser = argparse.ArgumentParser(description="乒乓球视觉智能裁判 · 状态机运行入口")
    parser.add_argument("--jsonl", required=True, help="输入的逐帧观测 JSONL 文件路径")
    parser.add_argument("--preset", default="test2", help="球台机位标定预设 (支持: test2, game2, rally, plos)")
    parser.add_argument("--video", default=None, help="[可选] 原始视频路径，用于渲染比分 HUD")
    parser.add_argument("--output", default=None, help="[可选] 输出视频保存路径 (如 output.mp4)")
    args = parser.parse_args()

    if not os.path.isfile(args.jsonl):
        print(f"[错误] 找不到输入的观测文件: {args.jsonl}")
        return 1

    try:
        calib = preset(args.preset)
    except KeyError:
        print(f"[错误] 未知的机位预设: {args.preset}。可用选项: test2, game2, rally, plos")
        return 1

    sm_rules = MatchRules()
    state_machine = TableTennisStateMachine(sm_rules)
    homography = TableHomography(calib, image_size=(1920, 1080))

    # 视频与渲染准备
    cap = None
    writer = None
    hud = None
    fps = 120.0
    w, h = 1920, 1080

    if args.video:
        if not os.path.isfile(args.video):
            print(f"[警告] 找不到视频文件 {args.video}，跳过视频渲染。")
        else:
            cap = cv2.VideoCapture(args.video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            homography = TableHomography(calib, image_size=(w, h))

    if args.output and cap:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        hud = BroadcastHUD(calib, homography)

    counts = Counter()
    landings = []
    last_decision_text = None
    last_decision_t = -999.0
    processor = None

    print("=" * 72)
    print(f"【乒乓球智能裁判】开始评测")
    print(f"输入观测文件: {os.path.basename(args.jsonl)}")
    print(f"球台几何预设: {args.preset} (球网X像素位置: {calib.net_x})")
    print("=" * 72)

    obs_total = 0
    t0 = time.time()

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            obs = FrameObservation.from_dict(data)
            obs_total += 1

            if processor is None:
                source_str = obs.source.value if hasattr(obs.source, "value") else str(obs.source)
                processor = TrajectoryEventProcessor(calib, source=source_str)

            t = obs.timestamp_s

            # 1. 状态机时钟步进（检查发球超时等）
            for decision in state_machine.tick(t):
                last_decision_text, last_decision_t = decision.describe(), t
                print(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] >> 判罚: {decision.describe()}")

            # 2. 轨迹处理器摄入观测，提取物理事件
            for ev in processor.update(obs):
                counts[ev.type] += 1
                if ev.type == EventType.BOUNCE.value:
                    landings.append(homography.locate_landing_from_event(ev))
                print(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] 事件: {ev.type:<11} 半台: {ev.side:<7} ID: {ev.event_id}")

                # 3. 将物理事件送入计分状态机
                for decision in state_machine.update(ev):
                    last_decision_text, last_decision_t = decision.describe(), t
                    print(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] >> 判罚: {decision.describe()}")

            # 4. 可选：渲染带比分面板的视频帧
            if cap and writer:
                ok, frame = cap.read()
                if ok:
                    current = processor.trajectory[-1] if processor.trajectory else None
                    hud_state = HUDState(
                        frame_id=obs.frame_id,
                        timestamp_s=t,
                        fps=fps,
                        score=(state_machine.points["A"], state_machine.points["B"]),
                        state=state_machine.state.value,
                        server=state_machine.server,
                        game_over=state_machine.game_over,
                        last_decision=last_decision_text,
                        decision_age_s=t - last_decision_t,
                        track_points=processor.trajectory,
                        landings=landings[-hud.landing_history:],
                        current_pos=(current.x, current.y) if current is not None else None,
                        track_method="tracknet",
                    )
                    writer.write(hud.render(frame, hud_state))

    if cap:
        cap.release()
    if writer:
        writer.release()
        print(f"\n[渲染成功] 视频已保存至: {args.output}")

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"评测完成！")
    print(f"处理观测总帧数 : {obs_total} 帧")
    print(f"状态机处理用时 : {elapsed:.3f} 秒 (吞吐速度: {obs_total/elapsed:.1f} FPS)")
    print(f"提取物理事件统计: {dict(counts)}")
    print(f"最终判定比分   : 选手P1 (半台A) {state_machine.points['A']} : {state_machine.points['B']} 选手P2 (半台B)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
