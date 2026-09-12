from __future__ import annotations

import pytest

from minoflux_ai import (
    NeuralValueConfig,
    NeuralValueEvaluator,
    VersusValueConfig,
    VersusValueEvaluator,
    build_neural_value_model,
    build_versus_value_model,
)
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_batch_fast import (
    _ORIGINAL_CHOOSE_VERSUS_ACTIONS_BATCH,
    choose_versus_actions_batch,
)
from minoflux_ai.versus_search import VersusSearchConfig, VersusSearchRequest
from minoflux_engine import VersusMatch


torch = pytest.importorskip("torch")


def _config() -> VersusSearchConfig:
    return VersusSearchConfig(
        placement_search=SearchConfig(
            allow_hold=True,
            lookahead_pieces=0,
            beam_width=4,
            srs_reachable=True,
            allow_180=False,
            reachability_node_limit=8000,
        ),
        candidate_width=16,
        opponent_reply_width=4,
    )


def _evaluators():
    torch.manual_seed(99173)
    solo_cfg = NeuralValueConfig()
    solo = NeuralValueEvaluator(
        build_neural_value_model(solo_cfg),
        solo_cfg,
        device="cpu",
        precision="float32",
        compile_model=False,
    )
    versus_cfg = VersusValueConfig()
    versus = VersusValueEvaluator(
        build_versus_value_model(versus_cfg),
        versus_cfg,
        device="cpu",
        compile_model=False,
    )
    return solo, versus


def _requests(seeds, solo, versus):
    return tuple(
        VersusSearchRequest(
            match=VersusMatch(seed),
            side_name="player",
            config=_config(),
            scorer=solo,
            opponent_scorer=solo,
            state_scorer=versus,
        )
        for seed in seeds
    )


def _signature(choice):
    if choice is None:
        return None
    return (
        choice.action.to_dict(),
        choice.score,
        choice.immediate.score,
        choice.resolution,
        None if choice.opponent_reply is None else choice.opponent_reply.to_dict(),
    )


def test_float32_neural_fusion_preserves_selected_moves_and_scores() -> None:
    seeds = (18001, 18032, 18063)
    old_solo, old_versus = _evaluators()
    old = _ORIGINAL_CHOOSE_VERSUS_ACTIONS_BATCH(
        _requests(seeds, old_solo, old_versus)
    )

    fused_solo, fused_versus = _evaluators()
    fused = choose_versus_actions_batch(
        _requests(seeds, fused_solo, fused_versus)
    )

    assert tuple(map(_signature, fused)) == tuple(map(_signature, old))
