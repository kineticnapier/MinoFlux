from pathlib import Path

from minoflux.versus_ai import (
    DEFAULT_AI_PPS,
    _resolve_versus_value_model,
    build_parser,
)


def test_parser_defaults_to_full_strength_search() -> None:
    args = build_parser().parse_args([])

    assert args.ai_pps == DEFAULT_AI_PPS == 2.5
    assert args.ai_candidates == 16
    assert args.ai_reply_width == 4
    assert args.ai_lookahead == 0
    assert args.ai_beam == 4
    assert not args.no_ai_versus_value


def test_default_versus_value_prefers_v2_then_v1(tmp_path: Path) -> None:
    v1 = tmp_path / "versus-value-v1.pt"
    v2 = tmp_path / "versus-value-v2.pt"
    v1.write_bytes(b"v1")
    v2.write_bytes(b"v2")

    assert _resolve_versus_value_model(None, defaults=(v2, v1)) == v2

    v2.unlink()
    assert _resolve_versus_value_model(None, defaults=(v2, v1)) == v1


def test_explicit_versus_value_wins_and_can_be_disabled(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.pt"
    fallback = tmp_path / "versus-value-v2.pt"
    fallback.write_bytes(b"fallback")

    assert _resolve_versus_value_model(
        str(explicit),
        defaults=(fallback,),
    ) == explicit
    assert _resolve_versus_value_model(
        str(explicit),
        disabled=True,
        defaults=(fallback,),
    ) is None


def test_missing_default_leaves_versus_value_off(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    assert _resolve_versus_value_model(None, defaults=(missing,)) is None
