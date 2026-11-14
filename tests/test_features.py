"""Feature rasters on hand-built grids, checked against hand calculations."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.features.build import count_neighbours
from src.features.dnbr import choose_pair, disc_average, nbr
from src.features.rasters import disc_kernel, disc_mean, distance_to, grid, horn_gradient, sample


def test_grid_snaps_outward_to_cell_multiples():
    transform, shape = grid((105, 210, 395, 480), cell_m=50, pad_m=0)
    assert (transform.c, transform.f) == (100, 500)   # top-left corner
    assert shape == (6, 6)                            # 200..500 rows, 100..400 cols


def test_grid_padding():
    transform, shape = grid((0, 0, 100, 100), cell_m=10, pad_m=25)
    assert (transform.c, transform.f) == (-30, 130)
    assert shape == (16, 16)


def test_disc_kernel_cell_counts():
    # r=1: centre + 4 edge neighbours. r=2: adds the 4 diagonals (dist sqrt 2)
    # and the 4 cells two steps away along the axes.
    assert disc_kernel(1).sum() == 5
    assert disc_kernel(2).sum() == 13


def test_disc_mean_of_single_cell():
    a = np.zeros((7, 7))
    a[3, 3] = 1
    m = disc_mean(a, 1)
    assert m[3, 3] == pytest.approx(1 / 5)
    assert m[3, 4] == pytest.approx(1 / 5)   # (3,3) is in its plus-shaped disc
    assert m[4, 4] == pytest.approx(0, abs=1e-12)


def test_disc_mean_of_constant_is_constant_up_to_the_edge():
    assert np.allclose(disc_mean(np.full((6, 9), 2.5), 3), 2.5)


def _plane(slope_east, slope_north, n=5, cell=10.0):
    # Row 0 is the northern edge, so northing falls with the row index.
    rows, cols = np.mgrid[0:n, 0:n]
    return slope_east * cols * cell + slope_north * (n - 1 - rows) * cell


def test_horn_gradient_of_a_plane():
    dx, dy = horn_gradient(_plane(0.2, -0.1), cell_m=10.0)
    assert dx[1:-1, 1:-1] == pytest.approx(np.full((3, 3), 0.2))
    assert dy[1:-1, 1:-1] == pytest.approx(np.full((3, 3), -0.1))


def test_surface_normal_of_a_south_facing_plane():
    # Ground rising northwards by 0.1 faces south: slope atan(0.1) = 5.71 deg,
    # northness = -0.1 / sqrt(1.01), eastness 0.
    dx, dy = horn_gradient(_plane(0.0, 0.1), cell_m=10.0)
    norm = np.sqrt(1 + dx**2 + dy**2)
    assert math.degrees(math.atan(math.hypot(dx[2, 2], dy[2, 2]))) == pytest.approx(5.7106, abs=1e-4)
    assert -dy[2, 2] / norm[2, 2] == pytest.approx(-0.1 / math.sqrt(1.01))
    assert -dx[2, 2] / norm[2, 2] == pytest.approx(0.0)


def test_distance_to_mask_and_cap():
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, 0] = True
    d = distance_to(mask, cell_m=25, cap_m=100)
    assert d[0, :6].tolist() == [0, 25, 50, 75, 100, 100]


def test_distance_with_no_mask_is_the_cap():
    assert (distance_to(np.zeros((3, 3), dtype=bool), 25, 500) == 500).all()


def test_sample_reads_the_containing_cell_and_rejects_outside_points():
    transform, shape = grid((0, 0, 30, 20), cell_m=10, pad_m=0)   # 2 rows x 3 cols
    a = np.arange(6).reshape(shape)
    assert sample(a, transform, [5, 25, 15], [15, 15, 5]).tolist() == [0, 2, 4]
    with pytest.raises(ValueError):
        sample(a, transform, [35], [5])


def _acq(name, orbit, ok):
    return {"datatake": name, "orbit": orbit, "ok": ok}


def test_choose_pair_post_first_then_pre_on_the_same_orbit():
    pre = [_acq("A", 7, True), _acq("B", 50, True), _acq("C", 7, False)]   # oldest first
    post = [_acq("D", 7, False), _acq("E", 7, True), _acq("F", 50, True)]
    before, after, reason = choose_pair(pre, post, lambda a: a["ok"])
    # E is the first usable after the fire; C is unusable and B is on another
    # orbit, so the latest usable pre-fire acquisition on orbit 7 is A.
    assert (before["datatake"], after["datatake"], reason) == ("A", "E", "")


def test_choose_pair_without_a_same_orbit_pre_fire_acquisition():
    before, after, reason = choose_pair([_acq("B", 50, True)], [_acq("E", 7, True)],
                                        lambda a: a["ok"])
    assert before is None and after["datatake"] == "E" and reason


def test_disc_average_uses_valid_cells_only():
    values = np.arange(25, dtype=float).reshape(5, 5)
    valid = np.zeros((5, 5), dtype=bool)
    valid[2, 2] = valid[2, 3] = True
    avg = disc_average(values, valid, 1)
    # The plus-shaped disc around (2, 2) holds two valid cells: 12 and 13.
    assert avg[2, 2] == pytest.approx(12.5)
    # The disc around (0, 0) holds (0, 0), (0, 1), (1, 0), none of them valid.
    assert np.isnan(avg[0, 0])


def test_nbr():
    out = nbr(np.array([0.3, 0.0]), np.array([0.1, 0.0]))
    assert out[0] == pytest.approx(0.5)   # (0.3 - 0.1) / (0.3 + 0.1)
    assert np.isnan(out[1])


def test_count_neighbours_excludes_self():
    xy = np.array([[0, 0], [60, 0], [170, 0]], dtype=float)
    assert count_neighbours(xy, xy, 100).tolist() == [1, 1, 0]
