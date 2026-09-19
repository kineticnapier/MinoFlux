from __future__ import annotations

from dataclasses import dataclass

from minoflux_engine import Game

from .search import SearchAction, SearchConfig, SearchChoice, choose_search_action

try:
    from . import _oracle_native as _native
except (ImportError, OSError):
    _native = None


@dataclass(frozen=True, slots=True)
class OracleConfig:
    beam_width: int = 2_000
    depth: int = 18
    allow_180: bool = True
    reachability_node_limit: int = 8_000

    def normalized(self) -> "OracleConfig":
        return OracleConfig(
            beam_width=max(1, int(self.beam_width)),
            depth=max(1, int(self.depth)),
            allow_180=bool(self.allow_180),
            reachability_node_limit=min(
                50_000,
                max(100, int(self.reachability_node_limit)),
            ),
        )


@dataclass(frozen=True, slots=True)
class OracleChoice:
    action: SearchAction
    score: float


def oracle_native_available() -> bool:
    return _native is not None and int(_native.api_version()) == 1


def search_oracle(
    game: Game,
    config: OracleConfig = OracleConfig(),
) -> OracleChoice | None:
    if not oracle_native_available():
        raise RuntimeError("MinoFlux native oracle extension is unavailable")

    cfg = config.normalized()
    choice: SearchChoice | None = choose_search_action(
        game,
        config=SearchConfig(
            allow_hold=True,
            lookahead_pieces=max(0, cfg.depth - 1),
            beam_width=cfg.beam_width,
            srs_reachable=True,
            allow_180=cfg.allow_180,
            reachability_node_limit=cfg.reachability_node_limit,
        ),
    )
    if choice is None:
        return None
    return OracleChoice(action=choice.action, score=float(choice.score))
