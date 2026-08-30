from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from minoflux_engine import Game

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .neural import NeuralState, NeuralValueConfig, NeuralValueEvaluator, encode_game_state
from .neural_dataset import NEURAL_DATASET_FORMAT, pack_board_rows
from .search import SearchAction, SearchConfig, apply_search_action, clone_game, rank_search_actions

NEURAL_REVIEW_QUEUE_FORMAT = "minoflux_neural_review_queue_v1"


@dataclass(frozen=True, slots=True)
class HumanReviewConfig:
    games: int = 8
    max_pieces: int = 500
    seed_base: int = 5_000_001
    seed_step: int = 97
    max_samples: int = 160
    # 0 keeps every legal placement. Human review defaults to the full action set so
    # a human can teach a move that both the NN and Champion rank poorly.
    max_candidates: int = 0
    uncertainty_margin: float = 0.08
    danger_height: int = 12
    danger_holes: int = 4
    topout_tail: int = 30
    search_config: SearchConfig = SearchConfig(
        allow_hold=True,
        lookahead_pieces=0,
        beam_width=4,
        discount=0.90,
        srs_reachable=True,
        allow_180=False,
        reachability_node_limit=8_000,
    )
    neural_config: NeuralValueConfig = NeuralValueConfig()

    def normalized(self) -> "HumanReviewConfig":
        raw_max_candidates = int(self.max_candidates)
        return HumanReviewConfig(
            games=max(1, int(self.games)),
            max_pieces=max(1, int(self.max_pieces)),
            seed_base=int(self.seed_base),
            seed_step=max(1, int(self.seed_step)),
            max_samples=max(1, int(self.max_samples)),
            max_candidates=0 if raw_max_candidates <= 0 else min(128, max(2, raw_max_candidates)),
            uncertainty_margin=max(0.0, float(self.uncertainty_margin)),
            danger_height=max(1, int(self.danger_height)),
            danger_holes=max(0, int(self.danger_holes)),
            topout_tail=max(0, int(self.topout_tail)),
            search_config=self.search_config.normalized(),
            neural_config=self.neural_config.normalized(),
        )


def _action_key(action: SearchAction) -> tuple[object, ...]:
    placement = action.placement
    return (
        bool(action.use_hold),
        placement.piece,
        int(placement.x),
        int(placement.y),
        int(placement.rotation),
    )


def _move_dict(action: SearchAction) -> dict[str, object]:
    placement = action.placement
    return {
        "hold": bool(action.use_hold),
        "piece": placement.piece,
        "x": int(placement.x),
        "y": int(placement.y),
        "rotation": int(placement.rotation),
    }


def _source_state(game: Game, config: NeuralValueConfig) -> dict[str, object]:
    state = encode_game_state(game, config)
    return {
        "rows": list(pack_board_rows(state.board, config)),
        "current": game.current,
        "hold": game.hold_piece,
        "next": list(game.queue)[: config.queue_length],
        "combo": int(game.combo),
        "b2b": bool(game.back_to_back),
        "b2bChain": int(game.b2b_chain),
        "surgeCharge": int(game.surge_charge),
    }


def _candidate_state(
    game: Game,
    action: SearchAction,
    neural_value: float,
    config: NeuralValueConfig,
) -> dict[str, object]:
    child = clone_game(game)
    apply_search_action(child, action)
    state: NeuralState = encode_game_state(child, config)
    return {
        "rows": list(pack_board_rows(state.board, config)),
        "context": list(state.context),
        "move": _move_dict(action),
        "nnValue": float(neural_value),
    }


def _select_candidates(
    neural_ranked: Sequence[tuple[SearchAction, object]],
    champion_action: SearchAction,
    max_candidates: int,
) -> list[tuple[SearchAction, object]]:
    if max_candidates <= 0 or len(neural_ranked) <= max_candidates:
        return list(neural_ranked)
    selected = list(neural_ranked[:max_candidates])
    champion_key = _action_key(champion_action)
    if any(_action_key(action) == champion_key for action, _ in selected):
        return selected
    champion_item = next(
        ((action, evaluation) for action, evaluation in neural_ranked if _action_key(action) == champion_key),
        None,
    )
    if champion_item is not None:
        selected[-1] = champion_item
    return selected


def build_review_record(
    game: Game,
    evaluator: NeuralValueEvaluator,
    weights: HeuristicWeights,
    config: HumanReviewConfig,
    *,
    reasons: Sequence[str] | None = None,
) -> dict[str, object] | None:
    cfg = config.normalized()
    neural_ranked = rank_search_actions(
        game,
        weights,
        cfg.search_config,
        limit=None,
        scorer=evaluator,
    )
    if not neural_ranked:
        return None
    champion_ranked = rank_search_actions(
        game,
        weights,
        cfg.search_config,
        limit=1,
    )
    if not champion_ranked:
        return None

    champion_action = champion_ranked[0][0]
    neural_action = neural_ranked[0][0]
    margin = float("inf")
    if len(neural_ranked) > 1:
        margin = float(neural_ranked[0][1].score - neural_ranked[1][1].score)

    detected = list(reasons or ())
    if _action_key(neural_action) != _action_key(champion_action):
        detected.append("nn_champion_disagree")
    if margin <= cfg.uncertainty_margin:
        detected.append("low_margin")
    top_features = neural_ranked[0][1].features.board
    if top_features.max_height >= cfg.danger_height:
        detected.append("high_stack")
    if top_features.holes >= cfg.danger_holes and cfg.danger_holes > 0:
        detected.append("holes")
    if not detected:
        return None

    selected = _select_candidates(neural_ranked, champion_action, cfg.max_candidates)
    return {
        "format": NEURAL_REVIEW_QUEUE_FORMAT,
        "seed": int(game.seed),
        "pieceIndex": int(game.pieces_placed),
        "reasons": sorted(set(detected)),
        "nnMargin": margin if margin != float("inf") else None,
        "source": _source_state(game, cfg.neural_config),
        "candidates": [
            _candidate_state(game, action, evaluation.score, cfg.neural_config)
            for action, evaluation in selected
        ],
        "nnChoice": 0,
        "championMove": _move_dict(champion_action),
    }


def collect_neural_review_queue(
    path: str | Path,
    evaluator: NeuralValueEvaluator,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    config: HumanReviewConfig = HumanReviewConfig(),
) -> dict[str, object]:
    cfg = config.normalized()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    records: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    topouts = 0

    def add_record(game: Game, reasons: Sequence[str] | None = None) -> None:
        if len(records) >= cfg.max_samples:
            return
        key = (int(game.seed), int(game.pieces_placed))
        if key in seen:
            return
        record = build_review_record(game, evaluator, weights, cfg, reasons=reasons)
        if record is not None:
            records.append(record)
            seen.add(key)

    for game_index in range(cfg.games):
        if len(records) >= cfg.max_samples:
            break
        seed = cfg.seed_base + game_index * cfg.seed_step
        game = Game(seed)
        tail: deque[Game] = deque(maxlen=cfg.topout_tail)
        while not game.game_over and game.pieces_placed < cfg.max_pieces:
            if cfg.topout_tail:
                tail.append(clone_game(game))
            neural_ranked = rank_search_actions(
                game,
                weights,
                cfg.search_config,
                limit=2,
                scorer=evaluator,
            )
            if not neural_ranked:
                break
            champion_ranked = rank_search_actions(game, weights, cfg.search_config, limit=1)
            if not champion_ranked:
                break
            neural_action, neural_eval = neural_ranked[0]
            margin = float("inf") if len(neural_ranked) < 2 else neural_eval.score - neural_ranked[1][1].score
            reasons: list[str] = []
            if _action_key(neural_action) != _action_key(champion_ranked[0][0]):
                reasons.append("nn_champion_disagree")
            if margin <= cfg.uncertainty_margin:
                reasons.append("low_margin")
            board = neural_eval.features.board
            if board.max_height >= cfg.danger_height:
                reasons.append("high_stack")
            if cfg.danger_holes > 0 and board.holes >= cfg.danger_holes:
                reasons.append("holes")
            if reasons:
                add_record(game, reasons)
            apply_search_action(game, neural_action)
            if len(records) >= cfg.max_samples:
                break

        if game.game_over:
            topouts += 1
            for tail_game in tail:
                if len(records) >= cfg.max_samples:
                    break
                add_record(tail_game, ("topout_tail",))

    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(target)
    summary = {
        "format": NEURAL_REVIEW_QUEUE_FORMAT,
        "path": str(target),
        "samples": len(records),
        "topouts": topouts,
        "config": {
            **asdict(cfg),
            "search_config": cfg.search_config.to_dict(),
            "neural_config": asdict(cfg.neural_config),
        },
    }
    target.with_suffix(target.suffix + ".meta.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def load_review_queue(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("format") != NEURAL_REVIEW_QUEUE_FORMAT:
                raise ValueError(f"Invalid review queue record at line {line_number}")
            records.append(record)
    if not records:
        raise ValueError("Review queue is empty")
    return records


def human_label_record(record: Mapping[str, object], selected_index: int) -> dict[str, object]:
    if record.get("format") != NEURAL_REVIEW_QUEUE_FORMAT:
        raise ValueError("Not a neural review queue record")
    candidates = record.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        raise ValueError("Review record has no candidates")
    index = int(selected_index)
    if index < 0 or index >= len(candidates):
        raise ValueError("Selected candidate is out of range")
    cleaned: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Review candidate must be an object")
        rows = candidate.get("rows")
        context = candidate.get("context")
        move = candidate.get("move")
        if not isinstance(rows, Sequence) or not isinstance(context, Sequence):
            raise ValueError("Review candidate is missing rows/context")
        cleaned.append({
            "rows": [int(value) for value in rows],
            "context": [float(value) for value in context],
            "move": move,
        })
    return {
        "format": NEURAL_DATASET_FORMAT,
        "seed": int(record.get("seed", 0)),
        "pieceIndex": int(record.get("pieceIndex", 0)),
        "expertIndex": index,
        "candidates": cleaned,
        "source": "human_review",
        "reviewReasons": list(record.get("reasons", ())),
    }


def review_key(record: Mapping[str, object]) -> tuple[int, int]:
    return int(record.get("seed", 0)), int(record.get("pieceIndex", 0))


def append_human_label(
    output_path: str | Path,
    record: Mapping[str, object],
    selected_index: int,
) -> bool:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    key = review_key(record)
    if target.is_file():
        with target.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                existing = json.loads(line)
                if isinstance(existing, Mapping) and review_key(existing) == key:
                    return False
    labeled = human_label_record(record, selected_index)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(labeled, separators=(",", ":")) + "\n")
    return True
