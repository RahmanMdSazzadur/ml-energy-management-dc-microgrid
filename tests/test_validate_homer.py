"""HOMER dispatch-replay must reproduce HOMER's own hourly output."""
import numpy as np
import pytest

from src.sim.validate_homer import HOURLY, compare, load_homer, replay

pytestmark = pytest.mark.skipif(not HOURLY.exists(), reason="HOMER export not present")


def test_replay_matches_homer_within_0p1_percent():
    h = load_homer()
    t = compare(h, replay(h))
    assert np.all(np.abs(t["annual_error_pct"].fillna(0)) < 0.1)
    assert t.loc[t.quantity == "grid_purchases_kw", "hourly_max_abs"].item() < 1e-3
