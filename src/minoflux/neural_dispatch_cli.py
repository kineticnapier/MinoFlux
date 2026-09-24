from __future__ import annotations

import sys

from . import (
    coupled_sweep_cli,
    coupled_train_cli,
    dagger_analysis_cli,
    dagger_inspector_pygame,
    duel_cli,
    neural_cli,
    oracle_cli,
    weighted_sampling_cli,
    weighted_train_cli,
)

_ORACLE_COMMANDS = frozenset({"oracle-smoke", "oracle-profile", "oracle-dataset", "oracle-dagger"})


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in _ORACLE_COMMANDS:
        return oracle_cli.main(raw_argv)
    if raw_argv and raw_argv[0] == "train-weighted":
        return weighted_train_cli.main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "train-coupled":
        return coupled_train_cli.main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "coupled-sweep":
        return coupled_sweep_cli.main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "sampling-overlap":
        return weighted_sampling_cli.main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "dagger-analyze":
        return dagger_analysis_cli.main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "dagger-inspect":
        return dagger_inspector_pygame.main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "duel":
        return duel_cli.main(raw_argv[1:])
    return neural_cli.main(raw_argv)


if __name__ == "__main__":
    raise SystemExit(main())
