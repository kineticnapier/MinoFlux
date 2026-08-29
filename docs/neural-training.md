# Neural value training

MinoFlux can warm-start a neural board evaluator by imitating the current heuristic Champion. The numeric heuristic score is **not** used as a regression target. Each sample contains the Champion-selected action and a set of legal alternatives; training uses a ranking margin so the selected successor state scores above the alternatives.

## 1. Install the optional ML dependency

```powershell
uv sync --extra ml --extra dev
```

PyTorch is optional. Normal engine, game, lab, CEM, and heuristic search imports still work without it.

## 2. Generate ranking data

```powershell
uv run minoflux-neural generate `
  --games 40 `
  --max-pieces 500 `
  --max-candidates 24 `
  --lookahead 0 `
  --beam 4
```

The default output is `data/neural/champion-ranking.jsonl`. Boards are stored as 24 compact 10-bit row masks rather than 240 JSON floats. The default run targets up to 20,000 Champion states and keeps the 24 hardest root alternatives per state. Use `--max-candidates 0` only when disk size is not a concern.

To make the teacher use future lookahead, raise `--lookahead`. This is much more expensive because the label itself then comes from beam search.

## 3. Train

```powershell
uv run minoflux-neural train `
  --dataset data/neural/champion-ranking.jsonl `
  --output data/models/neural-value.pt `
  --epochs 8 `
  --batch-size 64 `
  --device auto
```

`auto` selects CUDA when `torch.cuda.is_available()` is true, otherwise CPU. A batch is a batch of ranking *states*: all candidate successor boards inside that batch are flattened and evaluated in one network forward pass.

Validation is split by whole game seed, not by adjacent positions from the same game, so neighboring states from one trajectory cannot leak into both train and validation.

Continue training from an existing checkpoint with:

```powershell
uv run minoflux-neural train `
  --resume data/models/neural-value.pt `
  --epochs 4
```

## 4. Smoke-test the trained scorer

```powershell
uv run minoflux-neural evaluate `
  --model data/models/neural-value.pt `
  --games 8 `
  --max-pieces 500 `
  --lookahead 0 `
  --beam 4 `
  --device auto
```

This keeps the existing SRS/Hold/search machinery and replaces the placement score with the neural value evaluator. It reports total pieces, Attack/Piece, topouts, and completions.

This is the imitation warm-start stage. A model trained only on Champion labels should not be expected to exceed the Champion reliably. The next stage is self-play/rollout value learning using actual match outcomes.
