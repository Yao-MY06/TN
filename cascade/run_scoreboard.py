"""回放/确认识别结果驱动的计分牌与 UI/语音演示入口。"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascade.scoreboard_bridge import ScoreboardBridge, ensure_voice_path


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_window(value: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in str(value).lower().split("x", 1))
    except Exception as exc:
        raise ValueError(f"--ui-window 需要 宽x高，例如 1024x600：{value!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError("--ui-window 的宽高必须 > 0")
    return width, height


class ScoreboardRuntime:
    """UI、语音输入与视觉判罚共用的串行运行器。"""

    def __init__(self, bridge: ScoreboardBridge, *, voice: str = "none", ui: bool = False,
                 ui_port: int = 8765, ui_window: str = "1024x600",
                 open_browser: bool = False, input_device: Optional[int] = None,
                 quiet: bool = False):
        if voice not in ("none", "text", "mic"):
            raise ValueError("voice 只能是 none/text/mic")
        self.bridge = bridge
        self.voice = voice
        self.quiet = quiet
        self.pipeline = None
        self.ui_server = None
        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._speaker = bridge.speaker

        if ui:
            from tt_referee.ui.server import ScoreServer
            width, height = _parse_window(ui_window)
            self.ui_server = ScoreServer(
                bridge.engine, port=ui_port, allow_demo=False,
                players={"A": "选手 A", "B": "选手 B"}, title="乒乓球智能裁判")
            self.ui_server.start(open_browser=open_browser,
                                 window_size=(width, height), compact=True)
            self._lock = self.ui_server._server.lock
            if not quiet:
                print(f"[UI] 计分界面：{self.ui_server.url}（{width}x{height}，实时只读）")

        if voice != "none":
            self._start_voice(voice, input_device)

    def _start_voice(self, voice: str, input_device: Optional[int]) -> None:
        ensure_voice_path()
        try:
            from tt_voice.app import Pipeline
            from tt_voice.config import load_config
            from tt_voice.feedback import NullSpeaker
        except Exception as exc:
            raise RuntimeError(f"语音模块依赖不可用：{exc}") from exc
        cfg = load_config()
        speaker = self._speaker or NullSpeaker()
        if self._speaker is None:
            self._speaker = speaker
        self.pipeline = Pipeline(cfg, engine=None, board=self.bridge.board,
                                 speaker=speaker, start_awake=(voice == "text"))
        self.pipeline.start()
        if voice == "text":
            self._worker = threading.Thread(target=self._text_worker, daemon=True)
        else:
            from tt_voice.audio_io import MicSource
            self._worker = threading.Thread(
                target=self._mic_worker, args=(cfg, input_device), daemon=True)
        self._worker.start()
        if not self.quiet:
            print(f"[语音] 模式：{voice}" + ("" if voice != "text" else "（stdin 口令，唤醒词可同句）"))

    def _text_worker(self) -> None:
        for line in sys.stdin:
            if self._stop.is_set():
                return
            text = line.strip()
            if text:
                self._queue.put(("text", text))

    def _mic_worker(self, cfg, input_device: Optional[int]) -> None:
        from tt_voice.audio_io import MicSource
        try:
            with MicSource(cfg.sample_rate, cfg.block_size, input_device) as source:
                for block in source:
                    if self._stop.is_set():
                        return
                    self._queue.put(("pcm", block))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    @staticmethod
    def _print_results(results) -> None:
        for result in results:
            if result.message:
                print(f"[语音] {result.message}")

    def poll(self, now: Optional[float] = None) -> List[Any]:
        now = time.monotonic() if now is None else now
        out: List[Any] = []
        with self._lock:
            while True:
                try:
                    kind, value = self._queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "error":
                    raise RuntimeError(f"麦克风运行失败：{value}")
                if self.pipeline is None:
                    continue
                if kind == "text":
                    out += self.pipeline.push_text(value, now=now)
                elif kind == "pcm":
                    out += self.pipeline.feed(value, now=now)
        if out and not self.quiet:
            self._print_results(out)
        return out

    def apply_decision(self, record: Dict[str, Any], video_t: float) -> Dict[str, Any]:
        with self._lock:
            return self.bridge.apply_decision(record, video_t)

    def tick(self, video_t: float) -> None:
        with self._lock:
            self.bridge.tick(video_t)

    def close(self) -> None:
        self._stop.set()
        if self.ui_server is not None:
            self.ui_server.stop()
            self.ui_server = None
        if self._speaker is not None:
            try:
                self._speaker.close()
            except Exception:
                pass
        if self.pipeline is not None and self.pipeline.engine is not None:
            try:
                self.pipeline.engine.close()
            except Exception:
                pass


def _apply_due(runtime: ScoreboardRuntime, decisions: List[Dict[str, Any]], index: int,
               video_t: float) -> int:
    while index < len(decisions) and float(decisions[index].get("virtual_t", 0.0)) <= video_t:
        record = decisions[index]
        runtime.apply_decision(record, float(record.get("virtual_t", video_t)))
        index += 1
    return index


def _run_clock(runtime: ScoreboardRuntime, decisions: List[Dict[str, Any]], speed: float) -> float:
    start = time.monotonic()
    index = 0
    last_t = 0.0
    while True:
        now = time.monotonic()
        video_t = (now - start) * speed
        last_t = video_t
        runtime.tick(video_t)
        runtime.poll(now)
        index = _apply_due(runtime, decisions, index, video_t)
        if index >= len(decisions) and not runtime.bridge.engine.pending:
            break
        time.sleep(0.01)
    return last_t


def _run_video(runtime: ScoreboardRuntime, decisions: List[Dict[str, Any]],
               video: str, speed: float, max_frames: Optional[int]) -> float:
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 120.0
    start = time.monotonic()
    index = 0
    frame_id = 0
    last_t = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_id >= max_frames:
                break
            video_t = frame_id / fps
            last_t = video_t
            target = start + video_t / max(speed, 1e-9)
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            runtime.tick(video_t)
            runtime.poll(time.monotonic())
            index = _apply_due(runtime, decisions, index, video_t)
            cv2.imshow("ppq scoreboard replay", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            frame_id += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return last_t


def run_scoreboard(run_dir: str, *, video: Optional[str] = None, speed: float = 1.0,
                   max_frames: Optional[int] = None, ui: bool = False,
                   ui_port: int = 8765, ui_window: str = "1024x600",
                   open_browser: bool = False, voice: str = "none",
                   say: bool = True, dead_ball_mode: str = "pending",
                   pending_timeout_s: float = 5.0, input_device: Optional[int] = None,
                   out_dir: Optional[str] = None, quiet: bool = False) -> Dict[str, Any]:
    if speed <= 0:
        raise ValueError("speed 必须 > 0")
    if pending_timeout_s < 0:
        raise ValueError("pending_timeout_s 必须 >= 0")
    decisions_path = os.path.join(run_dir, "decisions.jsonl")
    if not os.path.isfile(decisions_path):
        raise FileNotFoundError(f"找不到判罚文件：{decisions_path}")
    decisions = sorted(_read_jsonl(decisions_path),
                       key=lambda row: float(row.get("virtual_t", 0.0)))
    bridge = ScoreboardBridge(say=say, dead_ball_mode=dead_ball_mode,
                              pending_timeout_s=pending_timeout_s)
    runtime = ScoreboardRuntime(
        bridge, voice=voice, ui=ui, ui_port=ui_port, ui_window=ui_window,
        open_browser=open_browser, input_device=input_device, quiet=quiet)
    try:
        last_t = (_run_video(runtime, decisions, video, speed, max_frames)
                  if video else _run_clock(runtime, decisions, speed))
        runtime.tick(last_t + pending_timeout_s)
    finally:
        runtime.close()
    paths = bridge.write_outputs(out_dir or run_dir)
    result = bridge.final_payload()
    result.update({"run_dir": os.path.abspath(run_dir), "paths": paths})
    if not quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="回放级联判罚并驱动计分牌 UI/语音")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--video", default=None, help="可选视频；不给则按速度用墙钟推进")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--ui", action="store_true")
    ap.add_argument("--ui-port", type=int, default=8765)
    ap.add_argument("--ui-window", default="1024x600")
    ap.add_argument("--open-browser", action="store_true")
    ap.add_argument("--voice", choices=["none", "text", "mic"], default="none")
    ap.add_argument("--say", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dead-ball-mode", choices=["pending", "ignore"], default="pending")
    ap.add_argument("--pending-timeout-s", type=float, default=5.0)
    ap.add_argument("--input-device", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None) -> int:
    a = _build_parser().parse_args(argv)
    run_scoreboard(
        a.run_dir, video=a.video, speed=a.speed, max_frames=a.max_frames,
        ui=a.ui, ui_port=a.ui_port, ui_window=a.ui_window,
        open_browser=a.open_browser, voice=a.voice, say=a.say,
        dead_ball_mode=a.dead_ball_mode, pending_timeout_s=a.pending_timeout_s,
        input_device=a.input_device, out_dir=a.out_dir, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())