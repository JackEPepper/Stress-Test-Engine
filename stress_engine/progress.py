"""Small, dependency-free progress reporting for command-line runs.

The calculation engine is also used as a Python library, so progress is an
explicitly injected concern.  Library callers receive a no-op reporter by
default; the CLI can opt into :class:`ConsoleProgressReporter`.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence, TextIO


@dataclass(frozen=True)
class ProgressStep:
    """One planned unit of work and its relative runtime weight."""

    key: str
    label: str
    weight: float = 1.0


class ProgressReporter:
    """No-op progress interface used by library calls and tests.

    Subclasses may render the lifecycle, but orchestration code can always call
    these methods without checking whether progress is enabled.
    """

    def start(self, title: str, steps: Sequence[ProgressStep]) -> None:
        """Start a run with an ordered collection of planned steps."""

    @contextmanager
    def step(self, key: str) -> Iterator[None]:
        """Track a step while leaving exceptions untouched."""
        yield

    def update(
        self,
        message: str,
        *,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        """Describe intermediate work within the active step."""

    def finish(self, summary: str = "Run complete") -> None:
        """Finish a successful run."""


class ConsoleProgressReporter(ProgressReporter):
    """Render line-oriented progress, timings, and adaptive ETAs to a stream."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        initial_seconds_per_weight: float = 0.15,
    ) -> None:
        """Initialize stream rendering and the monotonic timing source."""
        self.stream = stream if stream is not None else sys.stderr
        self._clock = clock
        self._initial_rate = max(float(initial_seconds_per_weight), 0.001)
        self._steps: tuple[ProgressStep, ...] = ()
        self._next_index = 0
        self._active_key: str | None = None
        self._active_started = 0.0
        self._run_started = 0.0
        self._completed_elapsed = 0.0
        self._completed_weight = 0.0
        self._finished = False

    def start(self, title: str, steps: Sequence[ProgressStep]) -> None:
        """Validate a fresh plan and print its initial runtime estimate."""
        planned = tuple(steps)
        if not planned:
            raise ValueError("Progress reporting requires at least one step.")
        keys = [step.key for step in planned]
        if len(set(keys)) != len(keys):
            raise ValueError("Progress step keys must be unique.")
        if any(step.weight <= 0 for step in planned):
            raise ValueError("Progress step weights must be positive.")
        if self._active_key is not None:
            raise RuntimeError("Cannot start a progress run while a step is active.")

        self._steps = planned
        self._next_index = 0
        self._active_key = None
        self._completed_elapsed = 0.0
        self._completed_weight = 0.0
        self._finished = False
        self._run_started = self._clock()

        initial_eta = self._initial_rate * sum(step.weight for step in planned)
        self._write("")
        self._write(f"Credit Stress Engine | {title}")
        self._write(
            f"Planned work: {len(planned)} steps | estimated runtime "
            f"{_format_duration(initial_eta)}"
        )

    @contextmanager
    def step(self, key: str) -> Iterator[None]:
        """Render one planned step, its duration, and any raised failure."""
        if self._finished or not self._steps:
            raise RuntimeError("Progress run has not been started.")
        if self._active_key is not None:
            raise RuntimeError("Progress steps cannot overlap.")
        if self._next_index >= len(self._steps):
            raise RuntimeError("All planned progress steps are already complete.")

        planned = self._steps[self._next_index]
        if key != planned.key:
            raise RuntimeError(
                f"Expected progress step '{planned.key}', received '{key}'."
            )

        # Estimate both the active step and the whole remaining plan from the
        # same blended rate so the two terminal numbers stay internally coherent.
        self._active_key = key
        self._active_started = self._clock()
        rate = self._estimated_rate()
        step_eta = rate * planned.weight
        run_eta = rate * sum(step.weight for step in self._steps[self._next_index :])
        prefix = self._prefix(self._next_index)
        self._write(
            f"{prefix} {planned.label} ... est. {_format_duration(step_eta)} "
            f"| ETA {_format_duration(run_eta)}"
        )

        # The context manager records a terminal outcome on both paths while
        # deliberately re-raising failures for the engine's normal handling.
        try:
            yield
        except BaseException:
            elapsed = max(self._clock() - self._active_started, 0.0)
            self._write(
                f"{prefix} {planned.label} ... FAILED after "
                f"{_format_duration(elapsed)}"
            )
            self._active_key = None
            raise
        else:
            elapsed = max(self._clock() - self._active_started, 0.0)
            self._completed_elapsed += elapsed
            self._completed_weight += planned.weight
            self._next_index += 1
            self._active_key = None
            remaining_weight = sum(
                step.weight for step in self._steps[self._next_index :]
            )
            remaining = self._estimated_rate() * remaining_weight
            suffix = (
                f" | ETA {_format_duration(remaining)}"
                if remaining_weight
                else ""
            )
            self._write(
                f"{prefix} {planned.label} ... DONE "
                f"{_format_duration(elapsed)}{suffix}"
            )

    def update(
        self,
        message: str,
        *,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        """Print an intermediate message and optional within-step ETA."""
        if not self._steps or self._finished:
            return

        suffix = ""
        if (
            self._active_key is not None
            and completed is not None
            and total is not None
            and 0 < completed <= total
        ):
            elapsed = max(self._clock() - self._active_started, 0.0)
            active_remaining = elapsed / completed * (total - completed)
            later_weight = sum(
                step.weight for step in self._steps[self._next_index + 1 :]
            )
            total_remaining = active_remaining + self._estimated_rate() * later_weight
            suffix = f" | ETA {_format_duration(total_remaining)}"
        self._write(f"      {message}{suffix}")

    def finish(self, summary: str = "Run complete") -> None:
        """Validate completion and print the total elapsed runtime."""
        if not self._steps or self._finished:
            return
        if self._active_key is not None:
            raise RuntimeError("Cannot finish progress while a step is active.")
        if self._next_index != len(self._steps):
            missing = len(self._steps) - self._next_index
            raise RuntimeError(f"Cannot finish progress with {missing} incomplete steps.")

        elapsed = max(self._clock() - self._run_started, 0.0)
        self._write(f"{summary} in {_format_duration(elapsed)}.")
        self._finished = True

    def _estimated_rate(self) -> float:
        """Blend initial and observed seconds per weight into a stable ETA rate."""
        if self._completed_weight <= 0:
            return self._initial_rate
        observed = self._completed_elapsed / self._completed_weight
        confidence = min(self._completed_weight / 8.0, 0.85)
        blended = (1.0 - confidence) * self._initial_rate + confidence * observed
        # Fast setup stages are not representative of model and reporting
        # work, so keep early estimates conservative instead of collapsing.
        return max(blended, self._initial_rate * 0.8, 0.01)

    def _prefix(self, zero_based_index: int) -> str:
        """Format a width-stable current/total step prefix."""
        width = len(str(len(self._steps)))
        return f"[{zero_based_index + 1:>{width}}/{len(self._steps)}]"

    def _write(self, message: str) -> None:
        """Write and flush one progress line to keep terminal output live."""
        print(message, file=self.stream, flush=True)


def _format_duration(seconds: float) -> str:
    """Return a compact, stable duration suitable for progress output."""
    seconds = max(float(seconds), 0.0)
    if seconds < 0.05:
        return "<0.1s"
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"
