from __future__ import annotations

from array import array
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from minoflux_engine import VersusMatch

from .bitboard import ROW_OCCUPANCY_BYTES
from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .progress import progress_bar
from .search import SearchScorer, apply_search_action
from .versus_neural import (
    VERSUS_SELFPLAY_FORMAT,
    VERSUS_VALUE_FORMAT,
    VersusSelfPlayConfig,
    VersusTrainConfig,
    VersusValueConfig,
    _outcome,
    _require_torch,
    _resolve_device,
    _state_record,
    build_versus_value_model,
    encode_versus_state,
    save_versus_value_checkpoint,
)
from .versus_search import (
    DEFAULT_VERSUS_WEIGHTS,
    SideName,
    VersusStateScorer,
    VersusWeights,
    choose_versus_action,
    score_versus_state,
)


def generate_versus_selfplay_dataset_progress(
    path: str | Path,
    solo_scorer: SearchScorer,
    config: VersusSelfPlayConfig = VersusSelfPlayConfig(),
    *,
    heuristic_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    versus_weights: VersusWeights = DEFAULT_VERSUS_WEIGHTS,
    value_scorer: VersusStateScorer | None = None,
    value_config: VersusValueConfig = VersusValueConfig(),
) -> dict[str, object]:
    cfg = VersusSelfPlayConfig(
        games=max(1, int(config.games)),
        max_turns=max(1, int(config.max_turns)),
        seed_base=int(config.seed_base),
        seed_step=int(config.seed_step),
        garbage_cap=max(1, int(config.garbage_cap)),
        search_config=config.search_config.normalized(),
    )
    value_cfg = value_config.normalized()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    wins = {"player": 0, "ai": 0, "draw": 0}
    total_turns = 0

    game_bar = progress_bar(range(cfg.games), total=cfg.games, desc="Self-play", unit="game")
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for game_index in game_bar:
            seed = cfg.seed_base + game_index * cfg.seed_step
            match = VersusMatch(seed, garbage_cap=cfg.garbage_cap)
            turn_side: SideName = "player" if game_index % 2 == 0 else "ai"
            pending_records: list[tuple[dict[str, object], SideName]] = []
            turns = 0
            turn_bar = progress_bar(
                total=cfg.max_turns,
                desc=f"Game {game_index + 1}/{cfg.games}",
                unit="ply",
                leave=False,
            )
            try:
                while match.winner is None and turns < cfg.max_turns:
                    for perspective in ("player", "ai"):
                        state = encode_versus_state(match, perspective, value_cfg, turn_side)
                        teacher_raw = score_versus_state(match, perspective, weights=versus_weights)
                        teacher_value = math.tanh(teacher_raw / 64.0)
                        pending_records.append(
                            (
                                _state_record(
                                    state,
                                    game_index=game_index,
                                    ply=turns,
                                    side=perspective,
                                    seed=seed,
                                    to_move=turn_side,
                                    teacher_value=teacher_value,
                                ),
                                perspective,
                            )
                        )
                    choice = choose_versus_action(
                        match,
                        turn_side,
                        heuristic_weights,
                        cfg.search_config,
                        versus_weights,
                        scorer=solo_scorer,
                        opponent_scorer=solo_scorer,
                        opponent_heuristic_weights=heuristic_weights,
                        state_scorer=value_scorer,
                    )
                    if choice is None:
                        match.side(turn_side).game.game_over = True
                        match._update_winner()
                        break
                    result = apply_search_action(match.side(turn_side).game, choice.action)
                    match.resolve_lock(turn_side, result)
                    turns += 1
                    turn_bar.update(1)
                    if turns % 8 == 0:
                        turn_bar.set_postfix(sentP=match.player.sent, sentA=match.ai.sent)
                    turn_side = "ai" if turn_side == "player" else "player"
            finally:
                turn_bar.close()

            winner = match.winner or "draw"
            wins[winner] += 1
            total_turns += turns

            for perspective in ("player", "ai"):
                terminal_state = encode_versus_state(match, perspective, value_cfg, None)
                terminal_target = _outcome(winner, perspective)
                pending_records.append(
                    (
                        _state_record(
                            terminal_state,
                            game_index=game_index,
                            ply=turns,
                            side=perspective,
                            seed=seed,
                            to_move=None,
                            teacher_value=terminal_target,
                            terminal=True,
                        ),
                        perspective,
                    )
                )

            for record, perspective in pending_records:
                record["outcome"] = _outcome(winner, perspective)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                total_records += 1
            game_bar.set_postfix(
                P=wins["player"],
                A=wins["ai"],
                D=wins["draw"],
                turns=turns,
                records=total_records,
            )
    game_bar.close()

    return {
        "format": VERSUS_SELFPLAY_FORMAT,
        "path": str(target),
        "games": cfg.games,
        "records": total_records,
        "playerWins": wins["player"],
        "aiWins": wins["ai"],
        "draws": wins["draw"],
        "meanTurns": total_turns / cfg.games,
        "searchConfig": cfg.search_config.to_dict(),
        "valueConfig": asdict(value_cfg),
    }


def _load_selfplay_records_progress(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    records: list[dict[str, object]] = []
    total_bytes = max(1, source.stat().st_size)
    bar = progress_bar(total=total_bytes, desc="Load dataset", unit="B", unit_scale=True)
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                bar.update(len(raw.encode("utf-8")))
                line = raw.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict) or value.get("format") != VERSUS_SELFPLAY_FORMAT:
                    raise ValueError(f"Invalid versus self-play record at line {line_number}")
                records.append(value)
    finally:
        bar.close()
    if not records:
        raise ValueError("Versus self-play dataset is empty")
    return records


def train_versus_value_model_progress(
    dataset_path: str | Path,
    output_path: str | Path,
    train_config: VersusTrainConfig = VersusTrainConfig(),
    model_config: VersusValueConfig = VersusValueConfig(),
    *,
    resume_from: str | Path | None = None,
) -> dict[str, object]:
    torch, _ = _require_torch()
    cfg = model_config.normalized()
    train_cfg = VersusTrainConfig(
        epochs=max(1, int(train_config.epochs)),
        batch_size=max(1, int(train_config.batch_size)),
        learning_rate=max(1e-7, float(train_config.learning_rate)),
        weight_decay=max(0.0, float(train_config.weight_decay)),
        validation_fraction=min(0.5, max(0.0, float(train_config.validation_fraction))),
        teacher_weight=max(0.0, float(train_config.teacher_weight)),
        seed=int(train_config.seed),
        device=str(train_config.device),
    )
    records = _load_selfplay_records_progress(dataset_path)
    rng = random.Random(train_cfg.seed)

    game_ids = sorted({(int(record.get("seed", 0)), int(record.get("game", 0))) for record in records})
    rng.shuffle(game_ids)
    validation_games = int(round(len(game_ids) * train_cfg.validation_fraction))
    validation_games = min(max(0, validation_games), max(0, len(game_ids) - 1))
    validation_set = set(game_ids[:validation_games])
    train_indices = [
        index
        for index, record in enumerate(records)
        if (int(record.get("seed", 0)), int(record.get("game", 0))) not in validation_set
    ]
    validation_indices = [
        index
        for index, record in enumerate(records)
        if (int(record.get("seed", 0)), int(record.get("game", 0))) in validation_set
    ]

    device = _resolve_device(torch, train_cfg.device)
    model = build_versus_value_model(cfg)
    if resume_from is not None:
        payload = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != VERSUS_VALUE_FORMAT:
            raise ValueError("Resume checkpoint is not a versus value model")
        resume_config = VersusValueConfig.from_mapping(payload.get("config", {}))
        if resume_config != cfg:
            raise ValueError("Resume checkpoint has a different versus value configuration")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Resume checkpoint has no state_dict")
        model.load_state_dict(state_dict)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    def pack(indices: Sequence[int]) -> tuple[Any, Any, Any, Any]:
        board_bytes = bytearray()
        context_values = array("f")
        outcomes = array("f")
        teachers = array("f")
        occupancy = ROW_OCCUPANCY_BYTES
        for index in indices:
            record = records[index]
            own_rows = record.get("ownRows")
            opponent_rows = record.get("opponentRows")
            context = record.get("context")
            if not isinstance(own_rows, list) or not isinstance(opponent_rows, list) or not isinstance(context, list):
                raise ValueError("Malformed versus self-play state")
            if len(own_rows) != cfg.board_height or len(opponent_rows) != cfg.board_height:
                raise ValueError("Versus self-play board height does not match model config")
            if len(context) != cfg.context_size:
                raise ValueError(
                    f"Versus self-play context size {len(context)} does not match model config {cfg.context_size}"
                )
            for mask in own_rows:
                board_bytes.extend(occupancy[int(mask)])
            for mask in opponent_rows:
                board_bytes.extend(occupancy[int(mask)])
            context_values.extend(float(value) for value in context)
            outcomes.append(float(record.get("outcome", 0.0)))
            teachers.append(float(record.get("teacherValue", 0.0)))
        count = len(indices)
        boards = torch.frombuffer(board_bytes, dtype=torch.uint8).reshape(
            count, 2, cfg.board_height, cfg.board_width
        ).to(device=device, dtype=torch.float32)
        contexts = torch.frombuffer(context_values, dtype=torch.float32).reshape(count, cfg.context_size).to(device)
        outcome_tensor = torch.frombuffer(outcomes, dtype=torch.float32).to(device)
        teacher_tensor = torch.frombuffer(teachers, dtype=torch.float32).to(device)
        return boards, contexts, outcome_tensor, teacher_tensor

    def loss_for(predictions: Any, outcomes: Any, teachers: Any) -> Any:
        loss = torch.mean((predictions - outcomes) ** 2)
        if train_cfg.teacher_weight:
            loss = loss + train_cfg.teacher_weight * torch.mean((predictions - teachers) ** 2)
        return loss

    def evaluate(indices: Sequence[int], epoch: int) -> float:
        if not indices:
            return 0.0
        model.eval()
        total = 0.0
        count = 0
        starts = range(0, len(indices), train_cfg.batch_size)
        bar = progress_bar(
            starts,
            total=math.ceil(len(indices) / train_cfg.batch_size),
            desc=f"Validate {epoch}/{train_cfg.epochs}",
            unit="batch",
            leave=False,
        )
        with torch.inference_mode():
            for start in bar:
                batch = indices[start : start + train_cfg.batch_size]
                boards, contexts, outcomes, teachers = pack(batch)
                loss = loss_for(model(boards, contexts), outcomes, teachers)
                total += float(loss.detach().cpu()) * len(batch)
                count += len(batch)
                bar.set_postfix(loss=f"{total / max(1, count):.5f}")
        bar.close()
        return total / max(1, count)

    history: list[dict[str, float | int]] = []
    epoch_bar = progress_bar(range(1, train_cfg.epochs + 1), total=train_cfg.epochs, desc="Train", unit="epoch")
    for epoch in epoch_bar:
        rng.shuffle(train_indices)
        model.train()
        total_loss = 0.0
        seen = 0
        starts = range(0, len(train_indices), train_cfg.batch_size)
        batch_bar = progress_bar(
            starts,
            total=math.ceil(len(train_indices) / train_cfg.batch_size),
            desc=f"Epoch {epoch}/{train_cfg.epochs}",
            unit="batch",
            leave=False,
        )
        for start in batch_bar:
            batch = train_indices[start : start + train_cfg.batch_size]
            boards, contexts, outcomes, teachers = pack(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model(boards, contexts), outcomes, teachers)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)
            batch_bar.set_postfix(loss=f"{total_loss / max(1, seen):.5f}")
        batch_bar.close()
        train_loss = total_loss / max(1, seen)
        validation_loss = evaluate(validation_indices, epoch)
        history.append(
            {
                "epoch": epoch,
                "trainLoss": train_loss,
                "validationLoss": validation_loss,
            }
        )
        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{validation_loss:.5f}")
    epoch_bar.close()

    save_bar = progress_bar(total=1, desc="Save checkpoint", unit="file")
    save_versus_value_checkpoint(
        output_path,
        model,
        cfg,
        metadata={
            "dataset": str(dataset_path),
            "records": len(records),
            "trainConfig": asdict(train_cfg),
            "resumeFrom": str(resume_from) if resume_from is not None else None,
            "history": history,
        },
    )
    save_bar.update(1)
    save_bar.close()

    return {
        "format": VERSUS_VALUE_FORMAT,
        "output": str(output_path),
        "device": device,
        "records": len(records),
        "trainRecords": len(train_indices),
        "validationRecords": len(validation_indices),
        "trainGames": len(game_ids) - validation_games,
        "validationGames": validation_games,
        "resumeFrom": str(resume_from) if resume_from is not None else None,
        "history": history,
        "config": asdict(cfg),
    }
