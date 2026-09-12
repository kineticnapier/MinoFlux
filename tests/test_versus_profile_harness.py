from __future__ import annotations

from tools.profile_versus_neural import _ForwardStats, _percentile


def test_profile_percentiles_and_forward_stats() -> None:
    stats = _ForwardStats()
    for batch, elapsed in ((4, 0.01), (8, 0.02), (16, 0.03), (32, 0.04)):
        stats.record(batch, elapsed)
    payload = stats.to_dict()
    assert payload["calls"] == 4
    assert payload["meanBatchSize"] == 15.0
    assert payload["p50BatchSize"] == 16.0
    assert payload["p95BatchSize"] == 32.0
    assert payload["maxBatchSize"] == 32
    assert abs(float(payload["totalSeconds"]) - 0.10) < 1e-12
    assert _percentile([], 0.95) == 0.0
