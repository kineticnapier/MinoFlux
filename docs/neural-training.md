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

### Optional: MochBot/fusion offline-oracle teacher

MinoFlux can also label the same ranking format with the external `generate_policy_value_labels` binary from MochBot/fusion. fusion is optional and is not imported or built by MinoFlux.

When MinoFlux is running in the same WSL environment as the fusion release binary:

```bash
uv run minoflux-neural fusion-dataset \
  --oracle-bin ~/dev/fusion/target/release/generate_policy_value_labels \
  --output data/neural/fusion-oracle.jsonl \
  --games 200 \
  --max-pieces 500 \
  --workers 12 \
  --allow-180
```

The generator batches requests into worker shards and forces `RAYON_NUM_THREADS=1` inside each oracle child. It intentionally does **not** pass fusion's `--time-budget-ms`: the offline teacher uses the full depth-18 / beam-2000 search rather than the time-budget iterative-widening path.

Current fusion binaries expose `best_move_raw`. MinoFlux decodes that move and accepts it only when piece plus occupied cells identify exactly one MinoFlux direct/Hold root action. If both a direct and Hold action are indistinguishable from the legacy output, the sample is skipped as `oracle-action-ambiguous`; unmatched cross-engine moves are skipped as `oracle-action-unmatched`. There is no nearest-move fallback.

An extended fusion binary may additionally emit `bestHoldUsed` and `bestCells`; when present, those fields are used for exact Hold/direct identity and remove the legacy Hold ambiguity.

Generated records still use `minoflux_neural_ranking_dataset_v1`, so they can be passed directly to the existing `train` command. A sidecar `<output>.meta.json` records worker count, trajectory policy, skip counts, and the oracle profile. The first implementation uses the explicit `heuristic` MinoFlux trajectory while fusion supplies the labels.

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