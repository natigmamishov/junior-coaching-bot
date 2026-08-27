"""Deterministic one-turn orchestration around the existing conversation engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional
import re

from core_contracts import PhoneStatus, TurnInput, TurnResult, TurnTrace


TRANSIENT_KEYS = {
    "_history", "_turn_traces", "_last_turn_trace", "_processed_messages",
    "_last_response_by_message_id",
}


def ensure_canonical_state(state: Dict[str, Any], conversation_id: str = "") -> None:
    """Upgrade legacy state in place while keeping old UI/DB fields operational."""
    state.setdefault("schema_version", 1)
    state.setdefault("state_version", 0)
    state.setdefault("conversation_id", conversation_id or state.get("conversation_id") or "")

    if state.get("phone"):
        phone_status = PhoneStatus.PROVIDED.value
    elif state.get("phone_declined"):
        phone_status = PhoneStatus.DECLINED.value
    else:
        phone_status = PhoneStatus.UNKNOWN.value
    state["phone_status"] = phone_status

    state.setdefault("pending_items", [])
    state.setdefault("_processed_messages", [])
    state.setdefault("_last_response_by_message_id", {})
    state.setdefault("_turn_traces", [])

    for index, child in enumerate(state.get("children") or []):
        if not child.get("child_id"):
            child["child_id"] = f"child-{index + 1}"
        child.setdefault("needs", [])
        child.setdefault("observations", [])
        child.setdefault("desired_outcomes", [])
        child.setdefault("fit_status", "unknown")
    if state.get("children"):
        active_index = min(state.get("active_child_index", 0), len(state["children"]) - 1)
        state["active_child_id"] = state["children"][active_index]["child_id"]
    else:
        state.setdefault("active_child_id", None)


def stable_message_id(turn: TurnInput, history_size: int) -> str:
    if turn.channel_message_id:
        return turn.channel_message_id
    raw = f"{turn.conversation_id}|{history_size}|{turn.text.strip()}"
    return sha256(raw.encode("utf-8")).hexdigest()[:32]


def state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in state.items()
        if key not in TRANSIENT_KEYS
    }


def state_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            diff[key] = {"before": before.get(key), "after": after.get(key)}
    return diff


class OutputValidator:
    """Deterministic final checks; it never adds business facts."""

    DIAGNOSIS_PATTERNS = (
        r"\bdiagnozudur\b", r"\bmütləq utancaqdır\b", r"\bkliniki olaraq\b",
    )
    FALSE_ACTION_PATTERNS = (
        "görüşünüz təsdiqləndi", "ödənişiniz təsdiqləndi", "qeydiyyat tamamlandı",
    )

    def validate(self, response: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        violations: List[str] = []
        normalized = " ".join(str(response or "").split())
        if not normalized:
            violations.append("empty_response")
        if len(normalized) > 1400:
            violations.append("response_too_long")
        if normalized.count("?") > 1:
            violations.append("more_than_one_question")
        lowered = normalized.lower()
        if any(re.search(pattern, lowered) for pattern in self.DIAGNOSIS_PATTERNS):
            violations.append("diagnosis_claim")
        succeeded = {a.get("type") for a in actions if a.get("status") == "succeeded"}
        if any(pattern in lowered for pattern in self.FALSE_ACTION_PATTERNS) and not succeeded:
            violations.append("unverified_action_confirmation")
        return {"valid": not violations, "violations": violations, "length": len(normalized)}


class TurnOrchestrator:
    def __init__(self, max_trace_items: int = 100):
        self.max_trace_items = max_trace_items
        self.output_validator = OutputValidator()

    def process(
        self,
        turn: TurnInput,
        state: Dict[str, Any],
        history: Optional[List[Dict[str, str]]],
        handler: Callable[..., str],
        faq_min_score: float,
    ) -> TurnResult:
        ensure_canonical_state(state, turn.conversation_id)
        message_id = stable_message_id(turn, len(history or []))
        version_before = int(state.get("state_version", 0))
        trace = TurnTrace(
            conversation_id=turn.conversation_id,
            message_id=message_id,
            state_version_before=version_before,
        )

        responses = state["_last_response_by_message_id"]
        if message_id in state["_processed_messages"]:
            response = responses.get(message_id, "")
            trace.duplicate = True
            trace.state_version_after = version_before
            trace.response_length = len(response)
            self._store_trace(state, trace)
            return TurnResult(
                response=response,
                trace_id=trace.trace_id,
                state_version_before=version_before,
                state_version_after=version_before,
                duplicate=True,
            )

        before = state_snapshot(state)
        response = handler(
            user_text=turn.text,
            lead=state,
            faq_min_score=faq_min_score,
            history=history,
        )
        ensure_canonical_state(state, turn.conversation_id)
        after = state_snapshot(state)
        diff = state_diff(before, after)
        state["state_version"] = version_before + 1
        trace.state_version_after = state["state_version"]
        trace.state_diff = diff
        trace.decision = {
            "intent": state.get("_last_intent"),
            "lead_stage": state.get("lead_stage"),
            "handoff_status": state.get("handoff_status"),
        }
        trace.actions = [
            action if isinstance(action, dict) else {"type": str(action), "status": "planned"}
            for action in state.get("previous_actions", [])[-10:]
        ]
        trace.validation = self.output_validator.validate(response, trace.actions)
        trace.response_length = len(response or "")

        state["_processed_messages"].append(message_id)
        del state["_processed_messages"][:-200]
        responses[message_id] = response
        if len(responses) > 200:
            for old_key in list(responses)[:-200]:
                responses.pop(old_key, None)
        self._store_trace(state, trace)

        return TurnResult(
            response=response,
            trace_id=trace.trace_id,
            state_version_before=version_before,
            state_version_after=state["state_version"],
            decision=trace.decision,
            actions=trace.actions,
        )

    def _store_trace(self, state: Dict[str, Any], trace: TurnTrace) -> None:
        value = trace.to_dict()
        state["_last_turn_trace"] = value
        traces = state.setdefault("_turn_traces", [])
        traces.append(value)
        del traces[:-self.max_trace_items]
