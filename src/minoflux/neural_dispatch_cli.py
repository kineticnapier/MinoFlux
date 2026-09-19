from __future__ import annotations

import sys

from . import neural_cli, oracle_cli

_ORACLE_COMMANDS = frozenset({"oracle-smoke", "oracle-profile", "oracle-dataset"})


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in _ORACLE_COMMANDS:
        return oracle_cli.main(raw_argv)
    return neural_cli.main(raw_argv)


if __name__ == "__main__":
    raise SystemExit(main())
