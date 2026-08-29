from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

from minoflux_engine import Game

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .neural import NeuralState, NeuralValueConfig, encode_game_state
from .search import (
    SearchAction,
    SearchConfig,
    apply_search_action,
    choose_search_action,
    clone_game,
    rank_search_actions,
)

NEURAL_DATASET_FORMAT = "minoflux_neural_ranking_dataset_v1"
DEFAULT_DATASET_SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    discount=0.90,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
)


@dataclass(frozen=True, slots=True)
class NeuralDatasetConfig:
    games: int = 40
    max_pieces: int = 500
    seed_base: int = 3_000_001
    seed_step: int = 31
    max_candidates: int = 24
    search_config: SearchConfig = DEFAULT_DATASET_SEARCH
    neural_config: NeuralValueConfig = NeuralValueConfig()

    def normalized(self) -> "NeuralDatasetConfig":
        return NeuralDatasetConfig(
            games=max(1, int(self.games)),
            max_pieces=max(1, int(self.max_pieces)),
            seed_base=int(self.seed_base),
            seed_step=max(1, int(self.seed_step)),
            max_candidates=max(0, int(self.max_candidates)),
            search_config=self.search_config.normalized(),
            neural_config=self.neural_config.normalized(),
        )


@dataclass(frozen=True, slots=True)
class NeuralRankingCandidate:
    board_rows: tuple[int, ...]
    context: tuple[float, ...]
    move: tuple[object, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": list(self.board_rows),
            "context": list(self.context),
            "move": list(self.move),
        }


@dataclass(frozen=True, slots=True)
class NeuralRankingSample:
    seed: int
    piece_index: int
    expert_index: int
    candidates: tuple[NeuralRankingCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "format": NEURAL_DATASET_FORMAT,
            "seed": self.seed,
            "pieceIndex": self.piece_index,
            "expertIndex": self.expert_index,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def pack_board_rows(
    board: Sequence[float],
    config: NeuralValueConfig = NeuralValueConfig(),
) -> tuple[int, ...]:
    cfg = config.normalized()
    expected = cfg.board_height * cfg.board_width
    if len(board) != expected:
        raise ValueError(f"Expected {expected} board cells, got {len(board)}")
    rows: list[int] = []
    for y in range(cfg.board_height):
        mask = 0
        start = y * cfg.board_width
        for x in range(cfg.board_width):
            if float(board[start + x]) >= 0.5:
                mask |= 1 << x
        rows.append(mask)
    return tuple(rows)


def unpack_board_rows(
    rows: Sequence[int],
    config: NeuralValueConfig = NeuralValueConfig(),
) -> tuple[float, ...]:
    cfg = config.normalized()
    if len(rows) != cfg.board_height:
        raise ValueError(f"Expected {cfg.board_height} packed rows, got {len(rows)}")
    return tuple(
        1.0 if int(rows[y]) & (1 << x) else 0.0
        for y in range(cfg.board_height)
        for x in range(cfg.board_width)
    )


def _move_tuple(action: SearchAction) -> tuple[object, ...]:
    placement = action.placement
    return (
        int(action.use_hold),
        placement.piece,
        placement.x,
        placement.y,
        placement.rotation,
    )


def _candidate_state(
    game: Game,
    action: SearchAction,
    neural_config: NeuralValueConfig,
) -> NeuralRankingCandidate:
    child = clone_game(game)
    apply_search_action(child, action)
    state: NeuralState = encode_game_state(child, neural_config)
    return NeuralRankingCandidate(
        board_rows=pack_board_rows(state.board, neural_config),
        context=state.context,
        move=_move_tuple(action),
    )


def _keep_hard_candidates(
    ranked: Sequence[tuple[SearchAction, object]],
    expert_action: SearchAction,
    max_candidates: int,
) -> tuple[tuple[SearchAction, object], ...]:
    if max_candidates <= 0 or len(ranked) <= max_candidates:
        return tuple(ranked)
    kept = list(ranked[:max_candidates])
    if any(action == expert_action for action, _ in kept):
        return tuple(kept)
    expert = next(item for item in ranked if item[0] == expert_action)
    kept[-1] = expert
    return tuple(kept)


def generate_neural_ranking_samples(
    config: NeuralDatasetConfig = NeuralDatasetConfig(),
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> Iterable[NeuralRankingSample]:
    cfg = config.normalized()
    for game_index in range(cfg.games):
        seed = cfg.seed_base + game_index * cfg.seed_step
        game = Game(seed)
        while not game.game_over and game.pieces_placed < cfg.max_pieces:
            # Enumerate the full root once so the dataset can keep hard alternatives.
            ranked = rank_search_actions(
                game,
                weights,
                cfg.search_config,
                limit=None,
            )
            if not ranked:
                break

            # With no future lookahead, the best root candidate is exactly the move
            # Champion would choose, so avoid a second identical SRS/search pass.
            if cfg.search_config.lookahead_pieces == 0:
                expert_action = ranked[0][0]
            else:
                expert = choose_search_action(game, weights, cfg.search_config)
                if expert is None:
                    break
                expert_action = expert.action

            # The label is the action Champion actually chooses. Numeric heuristic
            # scores are deliberately not stored as neural regression targets.
            source = _keep_hard_candidates(ranked, expert_action, cfg.max_candidates)
            actions = [action for action, _ in source]
            if expert_action not in actions:
                raise AssertionError("Expert action disappeared from neural ranking candidates")
            candidates = tuple(
                _candidate_state(game, action, cfg.neural_config)
                for action in actions
            )
            yield NeuralRankingSample(
                seed=seed,
                piece_index=game.pieces_placed,
                expert_index=actions.index(expert_action),
                candidates=candidates,
            )
            apply_search_action(game, expert_action)


def write_neural_ranking_dataset(
    path: str | Path,
    config: NeuralDatasetConfig = NeuralDatasetConfig(),
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    *,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 500,
) -> dict[str, object]:
    cfg = config.normalized()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    samples = 0
    candidates = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for sample in generate_neural_ranking_samples(cfg, weights):
            stream.write(json.dumps(sample.to_dict(), separators=(",", ":")) + "\n")
            samples += 1
            candidates += len(sample.candidates)
            if progress is not None and samples % max(1, int(progress_every)) == 0:
                progress(samples, candidates)
    temporary.replace(target)
    result = {
        "format": NEURAL_DATASET_FORMAT,
        "path": str(target),
        "samples": samples,
        "candidates": candidates,
        "config": {
            "games": cfg.games,
            "maxPieces": cfg.max_pieces,
            "seedBase": cfg.seed_base,
            "seedStep": cfg.seed_step,
            "maxCandidates": cfg.max_candidates,
            "searchConfig": cfg.search_config.to_dict(),
            "neuralConfig": asdict(cfg.neural_config),
        },
    }
    metadata_path = target.with_suffix(target.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
