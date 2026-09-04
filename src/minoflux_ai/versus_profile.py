from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from typing import Iterator

from .reachability import ReachabilityProfile, collect_reachability_profile


# Keep this order stable so JSON consumers and before/after reports remain easy
# to compare. Timings are inclusive: a parent phase may contain child phases.
VERSUS_PROFILE_PHASES = (
    "total_search",
    "root_placement_generation",
    "opponent_placement_generation",
    "srs_reachability",
    "branch_groups",
    "neural_placement_scoring",
    "root_simulate_action",
    "reply_simulate_action",
    "clone_versus_match",
    "clone_game",
    "board_copy",
    "bag_rng_state_copy",
    "garbage_queue_copy",
    "garbage_rng_state_copy",
    "apply_search_action",
    "resolve_lock",
    "root_state_encoding",
    "reply_state_encoding",
    "versus_value_inference",
    "score_versus_state",
    "max_height_and_holes",
    "path_materialization",
    "python_aggregation_tie_breaking",
)


@dataclass(slots=True)
class ProfileStat:
    calls: int = 0
    total_ns: int = 0

    def add(self, elapsed_ns: int, *, calls: int = 1) -> None:
        self.calls += max(0, int(calls))
        self.total_ns += max(0, int(elapsed_ns))


def _empty_stats() -> dict[str, ProfileStat]:
    return {name: ProfileStat() for name in VERSUS_PROFILE_PHASES}


@dataclass(slots=True)
class VersusSearchProfile:
    """Opt-in aggregate timings for versus search.

    Phase times are inclusive, so their percentages are independently useful but
    are not expected to add up to 100 percent.
    """

    stats: dict[str, ProfileStat] = field(default_factory=_empty_stats)
    reachability: ReachabilityProfile = field(default_factory=ReachabilityProfile)

    def record(self, phase: str, elapsed_ns: int, *, calls: int = 1) -> None:
        try:
            stat = self.stats[phase]
        except KeyError as error:
            raise ValueError(f"Unknown versus profile phase: {phase}") from error
        stat.add(elapsed_ns, calls=calls)

    def record_since(self, phase: str, started_ns: int, *, calls: int = 1) -> None:
        self.record(phase, _clock_ns() - started_ns, calls=calls)

    def _snapshot(self, phase: str) -> ProfileStat:
        stat = self.stats[phase]
        if phase == "srs_reachability":
            return ProfileStat(
                calls=self.reachability.calls,
                total_ns=round(self.reachability.total_seconds * 1_000_000_000),
            )
        if phase == "path_materialization":
            path_calls = int(getattr(self.reachability, "path_calls", 0))
            path_seconds = float(getattr(self.reachability, "path_seconds", 0.0))
            if path_calls or path_seconds:
                return ProfileStat(
                    calls=path_calls,
                    total_ns=round(path_seconds * 1_000_000_000),
                )
        return ProfileStat(calls=stat.calls, total_ns=stat.total_ns)

    def phase_rows(self) -> tuple[dict[str, int | float | str], ...]:
        total_ns = self._snapshot("total_search").total_ns
        rows: list[dict[str, int | float | str]] = []
        for phase in VERSUS_PROFILE_PHASES:
            stat = self._snapshot(phase)
            rows.append(
                {
                    "name": phase,
                    "calls": stat.calls,
                    "totalMs": stat.total_ns / 1_000_000.0,
                    "meanUsPerCall": (
                        stat.total_ns / stat.calls / 1_000.0
                        if stat.calls
                        else 0.0
                    ),
                    "percentOfSearch": (
                        stat.total_ns * 100.0 / total_ns
                        if total_ns
                        else 0.0
                    ),
                }
            )
        return tuple(rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "timingKind": "inclusive",
            "phases": list(self.phase_rows()),
            "reachability": self.reachability.to_dict(),
        }

    def format_table(self, *, include_zero: bool = False) -> str:
        rows = self.phase_rows()
        if not include_zero:
            rows = tuple(row for row in rows if int(row["calls"]) > 0)
        lines = [
            "Versus profile (inclusive timings)",
            f"{'phase':34} {'calls':>10} {'total ms':>12} {'mean us/call':>14} {'search %':>10}",
        ]
        for row in rows:
            lines.append(
                f"{str(row['name']):34} "
                f"{int(row['calls']):10d} "
                f"{float(row['totalMs']):12.3f} "
                f"{float(row['meanUsPerCall']):14.3f} "
                f"{float(row['percentOfSearch']):9.2f}%"
            )
        return "\n".join(lines)


_ACTIVE_VERSUS_PROFILE: ContextVar[VersusSearchProfile | None] = ContextVar(
    "minoflux_versus_profile",
    default=None,
)

_clock_ns = time.perf_counter_ns


def active_versus_profile() -> VersusSearchProfile | None:
    return _ACTIVE_VERSUS_PROFILE.get()


def profile_timer_start(profile: VersusSearchProfile | None) -> int:
    """Start a phase timer without reading the clock when profiling is off."""

    return _clock_ns() if profile is not None else 0


def record_profile_elapsed(
    profile: VersusSearchProfile | None,
    phase: str,
    started_ns: int,
    *,
    calls: int = 1,
) -> None:
    """Record a phase timer; the disabled path is one predictable branch."""

    if profile is not None:
        profile.record_since(phase, started_ns, calls=calls)


@contextmanager
def collect_versus_profile(
    profile: VersusSearchProfile | None = None,
) -> Iterator[VersusSearchProfile]:
    """Collect versus and exact-SRS metrics in the current execution context."""

    active = profile if profile is not None else VersusSearchProfile()
    token = _ACTIVE_VERSUS_PROFILE.set(active)
    try:
        with collect_reachability_profile(active.reachability):
            yield active
    finally:
        _ACTIVE_VERSUS_PROFILE.reset(token)
