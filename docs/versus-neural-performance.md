# Versus neural performance profiling

`tools/profile_versus_neural.py` measures the exact neural versus search path without changing search semantics. It uses default-size float32 placement and versus-value networks with all weights set to zero. The zero weights keep ranking deterministic while preserving the model shapes, tensor transfers, and forward work of the production architectures.

The report includes wall time, games/s, turns/s, exact-SRS reachability time, root/reply expansion time, board/garbage update time, feature-encoding time, Python orchestration time, model-forward call counts, forward batch-size mean/p50/p95, neural forward wall time, and optional `nvidia-smi` utilization / memory samples.

Example 100-game CUDA profile:

```powershell
uv run --no-sync python tools/profile_versus_neural.py `
  --mode both `
  --games 100 `
  --max-turns 80 `
  --game-batch 20 `
  --device cuda `
  --output data/benchmarks/versus-profile.json
```

For game-batch scaling, repeat the same seeds with `--game-batch 20`, `40`, `80`, and `160`.

Phase timings are inclusive and therefore should not be summed as disjoint CPU time. `neuralInferenceDutyCycle` and sampled GPU utilization are useful indicators of GPU idle time.
