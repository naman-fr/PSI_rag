"""
Lightweight request tracing.

Each incoming request gets a ``RequestTrace`` that records wall-clock
timing, pipeline steps, and cumulative token usage so the response can
include debug metadata.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def generate_trace_id() -> str:
    """Return a new UUID-4 trace identifier."""
    return str(uuid.uuid4())


@dataclass
class TraceStep:
    """A single named step inside a request pipeline."""

    name: str
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, **extra: Any) -> None:
        """Mark the step as complete."""
        self.ended_at = time.perf_counter()
        self.metadata.update(extra)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000, 2)


@dataclass
class RequestTrace:
    """Collects timing and usage data for a single request.

    Attributes
    ----------
    trace_id:
        Unique identifier for this trace.
    question:
        The user question that triggered the request.
    username:
        The authenticated user (if available).
    steps:
        Ordered list of pipeline steps.
    tokens_prompt:
        Cumulative prompt tokens consumed.
    tokens_completion:
        Cumulative completion tokens consumed.
    """

    trace_id: str
    question: str
    username: Optional[str] = None
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: Optional[float] = None
    steps: List[TraceStep] = field(default_factory=list)
    tokens_prompt: int = 0
    tokens_completion: int = 0

    # -- convenience ---------------------------------------------------------

    def add_step(self, name: str) -> TraceStep:
        """Create and register a new step."""
        step = TraceStep(name=name)
        self.steps.append(step)
        return step

    def add_tokens(self, prompt: int = 0, completion: int = 0) -> None:
        """Accumulate token counts."""
        self.tokens_prompt += prompt
        self.tokens_completion += completion

    @property
    def total_tokens(self) -> int:
        return self.tokens_prompt + self.tokens_completion

    @property
    def duration_ms(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000, 2)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def start_trace(question: str, username: Optional[str] = None) -> RequestTrace:
    """Create a new ``RequestTrace`` and start the clock.

    Parameters
    ----------
    question:
        User question text.
    username:
        Optional authenticated username.
    """
    return RequestTrace(
        trace_id=generate_trace_id(),
        question=question,
        username=username,
    )


def end_trace(trace: RequestTrace) -> Dict[str, Any]:
    """Finalise a trace and return a serialisable summary dict.

    Automatically closes any open steps.
    """
    trace.ended_at = time.perf_counter()

    for step in trace.steps:
        if step.ended_at is None:
            step.finish()

    return {
        "trace_id": trace.trace_id,
        "question": trace.question,
        "username": trace.username,
        "duration_ms": trace.duration_ms,
        "tokens": {
            "prompt": trace.tokens_prompt,
            "completion": trace.tokens_completion,
            "total": trace.total_tokens,
        },
        "steps": [
            {
                "name": s.name,
                "duration_ms": s.duration_ms,
                **s.metadata,
            }
            for s in trace.steps
        ],
    }
