"""把级联判罚接到语音仓库的 ScoreEngine。

视觉判罚、语音口令与 UI 共用同一个 ``ScoreEngine``。本模块只做映射与日志，
不修改级联状态机，也不把记分牌状态回写到级联。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from ._paths import PPQ_ROOT

VOICE_ROOT = os.path.join(PPQ_ROOT, "pingpong-voice-referee")


def ensure_voice_path(root: str = VOICE_ROOT) -> str:
    """只把语音仓库加入 sys.path；两个仓库同名包，不能同时挂载。"""
    root = os.path.abspath(root)
    if not os.path.isdir(os.path.join(root, "tt_referee")):
        raise FileNotFoundError(
            f"找不到语音仓库 tt_referee：{root}（需要 ppq/pingpong-voice-referee）")
    loaded = sys.modules.get("tt_referee")
    if loaded is not None:
        origin = os.path.abspath(getattr(loaded, "__file__", "") or "")
        if not origin.startswith(root + os.sep):
            raise RuntimeError(
                f"tt_referee 已从其他路径加载：{origin}；请只使用 {root} 这一份")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load_score_api(root: str = VOICE_ROOT):
    ensure_voice_path(root)
    try:
        from tt_referee.scoring import STATE_PENDING, ActionResult, ScoreEngine, ScoringRules
        from tt_referee.voice_compat import SharedScoreBoard
    except Exception as exc:
        raise RuntimeError(f"加载语音仓库计分模块失败：{exc}") from exc
    return {
        "STATE_PENDING": STATE_PENDING,
        "ActionResult": ActionResult,
        "ScoreEngine": ScoreEngine,
        "ScoringRules": ScoringRules,
        "SharedScoreBoard": SharedScoreBoard,
    }


def build_speaker(say: bool = True, root: str = VOICE_ROOT):
    ensure_voice_path(root)
    try:
        from tt_voice.feedback import build_speaker as _build
    except Exception as exc:
        raise RuntimeError(f"加载语音播报模块失败：{exc}") from exc
    return _build(say=say, beep=False)


def _make_cascade_board(api):
    class CascadeScoreBoard(api["SharedScoreBoard"]):
        def apply(self, intent: str, raw_text: str = "", score: float = 0.0,
                  source: str = "voice"):
            if intent == "confirm" and self.engine.pending \
                    and self.engine.pending.get("side") == "UNKNOWN":
                return api["ActionResult"](
                    ok=False, intent="confirm", reason="side_required",
                    message="请说明左方加一分或右方加一分",
                    state=self.snapshot())
            return super().apply(intent, raw_text, score, source)

    return CascadeScoreBoard(None, log_events=False)

class ScoreboardBridge:
    """视觉判罚 → ScoreEngine + 待确认超时 + 可追溯日志。"""

    def __init__(self, board=None, *, say: bool = True, speaker=None,
                 dead_ball_mode: str = "pending", pending_timeout_s: float = 5.0,
                 log=None):
        if dead_ball_mode not in ("pending", "ignore"):
            raise ValueError("dead_ball_mode 只能是 pending 或 ignore")
        if pending_timeout_s < 0:
            raise ValueError("pending_timeout_s 必须 >= 0")
        api = load_score_api()
        self.board = board or _make_cascade_board(api)
        self.engine = self.board.engine
        self.say = bool(say)
        self.speaker = speaker if speaker is not None else (build_speaker(True) if say else None)
        self.dead_ball_mode = dead_ball_mode
        self.pending_timeout_s = float(pending_timeout_s)
        self.logs: List[Dict[str, Any]] = []
        self._logger = log
        self._action_index = len(self.engine.actions)
        self._pending_deadline: Optional[float] = None
        self.physical_points = {"A": 0, "B": 0}
        self.vision_player_points = {"A": 0, "B": 0}
        self.manual_override_count = 0

    # ------------------------------------------------------------ 映射
    def map_winner(self, physical_side: str) -> str:
        if physical_side not in ("A", "B"):
            raise ValueError(f"物理侧只能是 A/B，收到 {physical_side!r}")
        physical = "left" if physical_side == "A" else "right"
        return "A" if self.engine.side_of_a == physical else "B"

    def _physical_of_player(self, player: str) -> str:
        return "left" if (self.engine.side_of_a == "left") == (player == "A") else "right"

    def _score_text(self) -> str:
        if hasattr(self.board, "score_text"):
            return self.board.score_text()
        left = self.engine.points["A"] if self.engine.side_of_a == "left" else self.engine.points["B"]
        right = self.engine.points["B"] if self.engine.side_of_a == "left" else self.engine.points["A"]
        return f"左方{left}比{right}右方"

    def _say(self, text: str) -> None:
        if self.say and self.speaker is not None:
            self.speaker.say(text)

    @staticmethod
    def _result_dict(result) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return {
            "ok": bool(result.ok), "intent": result.intent, "reason": result.reason,
            "message": result.message, "state": result.state,
        }

    def _append_log(self, video_t: float, action: str, **payload: Any) -> Dict[str, Any]:
        entry = {"video_t": round(float(video_t), 6), "action": action,
                 "score": self.engine.snapshot(), **payload}
        self.logs.append(entry)
        if self._logger is not None:
            self._logger(entry)
        return entry

    def _detect_manual_override(self, video_t: float) -> None:
        actions = self.engine.actions[self._action_index:]
        for action in actions:
            self.manual_override_count += 1
            self._append_log(video_t, "manual_override", engine_action=dict(action),
                             note="记分牌被语音/UI/人工修改；级联状态机未回写")
        self._action_index = len(self.engine.actions)

    # ------------------------------------------------------------ 判罚
    def apply_decision(self, record: Dict[str, Any], video_t: float) -> Dict[str, Any]:
        self._detect_manual_override(video_t)
        kind = str(record.get("kind") or "")
        rule = str(record.get("rule") or "unknown")
        physical = record.get("winner")
        mapped = None
        result = None
        action = "ignored"

        if kind == "POINT":
            if physical not in ("A", "B"):
                entry = self._append_log(video_t, "ignored", input=record,
                                         reason="invalid_point_writer")
                self._action_index = len(self.engine.actions)
                return entry
            if self.engine.pending and self.engine.pending.get("source") == "vision":
                rejected = self.engine.reject_pending()
                self._append_log(video_t, "reject_pending_before_point",
                                 input=record, result=self._result_dict(rejected))
                self._action_index = len(self.engine.actions)
            mapped = self.map_winner(physical)
            result = self.engine.finish_rally(
                mapped, reason=f"vision:{rule}", confidence=1.0, source="vision")
            action = "point"
            if result.ok:
                self.physical_points[physical] += 1
                self.vision_player_points[mapped] += 1
                physical_name = "左方" if physical == "A" else "右方"
                self._say(f"{physical_name}得分，{self._score_text()}")
        elif kind == "DEAD_BALL":
            if self.dead_ball_mode == "ignore":
                action = "dead_ball_ignored"
            else:
                result = self.engine.propose_pending(
                    "UNKNOWN", reason=f"vision:{rule}", confidence=1.0, source="vision")
                self._pending_deadline = float(video_t) + self.pending_timeout_s
                action = "dead_ball_pending"
                self._say("这一分请确认")
        elif kind == "LET":
            result = self.engine.let(reason=f"vision:{rule}", source="vision")
            action = "let"

        entry = self._append_log(
            video_t, action, input=record, physical_winner=physical, mapped_side=mapped,
            result=self._result_dict(result))
        self._action_index = len(self.engine.actions)
        return entry

    def tick(self, video_t: float) -> None:
        self._detect_manual_override(video_t)
        if self._pending_deadline is None:
            return
        if not self.engine.pending:
            self._pending_deadline = None
            return
        if float(video_t) < self._pending_deadline:
            return
        result = self.engine.reject_pending()
        self._append_log(video_t, "pending_timeout", result=self._result_dict(result),
                         deadline_s=round(self._pending_deadline, 6))
        self._pending_deadline = None
        self._action_index = len(self.engine.actions)

    # ------------------------------------------------------------ 输出
    def final_payload(self) -> Dict[str, Any]:
        return {
            "snapshot": self.engine.snapshot(),
            "physical_points": dict(self.physical_points),
            "vision_player_points": dict(self.vision_player_points),
            "manual_override_count": self.manual_override_count,
            "log_entries": len(self.logs),
        }

    def write_outputs(self, out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "scoreboard_log.jsonl")
        final_path = os.path.join(out_dir, "scoreboard_final.json")
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in self.logs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(self.final_payload(), f, ensure_ascii=False, indent=2)
        return {"log": log_path, "final": final_path}