from __future__ import annotations

# Compatibility wrapper: unpack_board_rows lives in neural_dataset, while the
# inspector implementation historically imported it from neural.
import minoflux_ai.neural as _neural
from minoflux_ai.neural_dataset import unpack_board_rows as _unpack_board_rows

_neural.unpack_board_rows = _unpack_board_rows

from . import dagger_inspector_pygame_impl as _impl

_best_index = _impl._best_index
_candidate_snapshot = _impl._candidate_snapshot
_is_disagreement = _impl._is_disagreement
_move_text = _impl._move_text
_visible_records = _impl._visible_records
build_parser = _impl.build_parser
main = _impl.main

if __name__ == "__main__":
    raise SystemExit(main())
