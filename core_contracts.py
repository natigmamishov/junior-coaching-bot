"""Typed V1 contracts for the Junior Coaching conversation core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PhoneStatus(str, Enum):
    UNKNOWN = "unknown"
    PROVIDED = "provided"
    DECLINED = "declined"
    INVALID = "invalid"


class PendingStatus(str, Enum):
    OPEN = "open"
    READY = "ready"
    ANSWERED = "answered"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class ActionStatus(str, Enum):
    PLANNED = "planned"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class TurnInput:
    conversation_id: str
    text: str
    channel: str = "web"
    channel_message_id: str = ""
    user_id: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    received_at: str = field(default_factory=utc_now)
    locale_hint: str = "az"


@dataclass
class PendingQuestion:
    original_text: str
    normalized_type: str = "unknown"
    missing_slots: List[str] = field(default_factory=list)
    target_child_id: Optional[str] = None
    collected_slots: Dict[str, Any] = field(default_factory=dict)
    status: str = PendingStatus.OPEN.value
    resume_policy: str = "auto_answer"
    original_message_id: Optional[str] = None
    pending_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=utc_now)
    expires_at: Optional[str] = None


@dataclass
class UnderstandingProposal:
    intents: List[Dict[str, Any]] = field(default_factory=list)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    semantic_items: List[Dict[str, Any]] = field(default_factory=list)
    objections: List[Dict[str, Any]] = field(default_factory=list)
    safety_flags: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Decision:
    action_code: str = "respond"
    rationale_code: str = "current_message"
    clarification_required: bool = False
    handoff_required: bool = False
    next_question: Optional[str] = None


@dataclass
class ActionCommand:
    type: str
    conversation_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    idempotency_key: str = ""
    requested_by_turn_id: Optional[str] = None
    requires_human_confirmation: bool = False
    status: str = ActionStatus.PLANNED.value
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    result: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None


@dataclass
class TurnTrace:
    conversation_id: str
    message_id: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=utc_now)
    build_version: str = "v1-core-1"
    schema_version: int = 1
    prompt_version: str = "conversation-v6"
    rule_version: str = "rules-v1"
    kb_version: str = "faq-v1"
    model_version: str = "gpt-4o-mini"
    state_version_before: int = 0
    state_version_after: int = 0
    state_diff: Dict[str, Any] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    pending_resolution: List[Dict[str, Any]] = field(default_factory=list)
    kb_refs: List[Dict[str, Any]] = field(default_factory=list)
    rule_refs: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    response_length: int = 0
    duplicate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TurnResult:
    response: str
    trace_id: str
    state_version_before: int
    state_version_after: int
    duplicate: bool = False
    understanding: Dict[str, Any] = field(default_factory=dict)
    validated_updates: List[Dict[str, Any]] = field(default_factory=list)
    rejected_extractions: List[Dict[str, Any]] = field(default_factory=list)
    pending_resolution: List[Dict[str, Any]] = field(default_factory=list)
    kb_refs: List[Dict[str, Any]] = field(default_factory=list)
    rule_refs: List[str] = field(default_factory=list)
    decision: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)

