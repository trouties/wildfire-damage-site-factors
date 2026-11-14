"""Evaluation statistics on hand-built inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.evaluate import (
    auc,
    block_ids,
    block_resamples,
    fit_slopes,
    graded_only_labels,
    heterogeneity,
    label_correlation,
)


def test_auc_perfect_reversed_and_tied():
    y = np.array([0, 0, 1, 1])
    assert auc(np.array([1, 2, 3, 4]), y) == 1.0
    assert auc(np.array([4, 3, 2, 1]), y) == 0.0
    assert auc(np.array([5, 5, 5, 5]), y) == 0.5


def test_auc_counts_ties_as_half():
    # Damaged at scores 1 and 2, undamaged at 0 and 1. Pairs (damaged vs
    # undamaged): 1>0, 1=1, 2>0, 2>1 -> (1 + 0.5 + 1 + 1) / 4.
    assert auc(np.array([0, 1, 1, 2]), np.array([0, 1, 0, 1])) == 0.875


def test_auc_needs_both_classes():
    with pytest.raises(ValueError):
        auc(np.array([1.0, 2.0]), np.array([1, 1]))


def test_block_ids_group_points_by_grid_square():
    x = np.array([10, 990, 1010, -10, 500])
    y = np.array([10, 990, 10, 10, 1500])
    ids = block_ids(x, y, 1000)
    assert ids[0] == ids[1]                      # both in square (0, 0)
    assert len({ids[0], ids[2], ids[3], ids[4]}) == 4
    assert sorted(set(ids)) == [0, 1, 2, 3]


def test_block_resamples_draw_whole_blocks_with_both_classes():
    blocks = np.array([0, 0, 1, 1, 1, 2])
    damaged = np.array([1, 0, 0, 0, 0, 1], bool)
    rng = np.random.default_rng(1)
    for idx in block_resamples(blocks, damaged, 50, rng):
        counts = np.bincount(idx, minlength=6)
        # members of one block always appear the same number of times
        assert counts[0] == counts[1] and counts[2] == counts[3] == counts[4]
        assert 0 < damaged[idx].sum() < idx.size


def test_graded_only_labels():
    # Event A: 2 damaged, 1 possibly damaged, 1 no visible damage, 1 not graded.
    # Event B: 2 damaged, 1 possibly damaged -> one undamaged, below the minimum.
    grades = ["Destroyed", "Damaged", "Possibly damaged", "No visible damage", None,
              "Damaged", "Damaged", "Possibly damaged"]
    df = pd.DataFrame({
        "event": list("AAAAABBB"),
        "damage_grade": grades,
        "damaged": pd.array([True, True, None, False, False, True, True, None], "boolean"),
    })
    y = graded_only_labels(df, min_per_class=2)
    assert y[:4].tolist() == [True, True, False, False]
    assert y[4:].isna().all()


def test_heterogeneity_of_two_events():
    # Weights 1/0.05² = 400, weighted mean 0.5, Q = 400 * 0.1² * 2 = 8 on 1 df:
    # I² = (8 - 1) / 8, p = P(chi² with 1 df > 8) = 2 * (1 - Phi(sqrt 8)) = 0.00468.
    i2, p = heterogeneity(pd.Series([0.6, 0.4]), pd.Series([0.05, 0.05]))
    assert i2 == pytest.approx(7 / 8)
    assert p == pytest.approx(0.00468, abs=1e-5)
    assert heterogeneity(pd.Series([0.5, 0.5]), pd.Series([0.05, 0.05]))[0] == 0


def test_label_correlation_by_distance():
    # Two damaged buildings 50 m apart, two undamaged ones 50 m apart, 950 to
    # 1050 m away. p = 1/2, so z = +1 for damaged and -1 for undamaged: pairs
    # 50 m apart share a label (+1), pairs ~1 km apart do not (-1).
    df = pd.DataFrame({"event": ["A"] * 4, "x": [0.0, 50, 1000, 1050], "y": [0.0] * 4})
    corr = label_correlation(df, pd.Series([True, True, False, False]), 100, 1100)
    by_bin = corr.set_index("from_m")
    assert by_bin.at[0, "correlation"] == 1 and by_bin.at[0, "n_pairs"] == 2
    assert by_bin.at[900, "correlation"] == -1        # 950 m
    assert by_bin.at[1000, "correlation"] == -1       # 1000 m and 1050 m
    assert by_bin.at[1000, "n_pairs"] == 3            # 1000, 1050, 1000
    assert by_bin.at[500, "n_events"] == 0


def test_fit_slopes_measures_within_event_effect():
    # Within both events damage rises with x (true slope 3 per unit of x), but
    # the event with larger x has far fewer damaged buildings. A common
    # intercept would turn the slope negative; per-event intercepts must not.
    rng = np.random.default_rng(0)
    n = 20000
    x = np.concatenate([rng.uniform(0, 1, n), rng.uniform(2, 3, n)])
    event = np.repeat(["A", "B"], n)
    logit = np.where(event == "A", 1.0, -8.0) + 3 * x
    damaged = rng.uniform(size=2 * n) < 1 / (1 + np.exp(-logit))
    assert np.corrcoef(x, damaged)[0, 1] < 0      # the pooled association is negative
    mean, sd, beta = fit_slopes(x[:, None], damaged, event)
    assert beta[0] / sd[0] == pytest.approx(3, rel=0.1)
