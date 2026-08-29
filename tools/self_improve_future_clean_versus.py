from __future__ import annotations

import json
from pathlib import Path

from self_improve_branch_tournament_fast import versus

result = versus("future_clean")
Path("versus-results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
