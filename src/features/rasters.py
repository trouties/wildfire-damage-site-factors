"""Raster helpers for the static features. All grids are north-up, row 0 at the top."""
from __future__ import annotations

import math

import numpy as np
from affine import Affine
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve


def grid(bounds: tuple[float, float, float, float], cell_m: float,
         pad_m: float) -> tuple[Affine, tuple[int, int]]:
    """Grid covering ``bounds`` grown by ``pad_m``, edges on multiples of ``cell_m``."""
    x0 = math.floor((bounds[0] - pad_m) / cell_m) * cell_m
    y0 = math.floor((bounds[1] - pad_m) / cell_m) * cell_m
    x1 = math.ceil((bounds[2] + pad_m) / cell_m) * cell_m
    y1 = math.ceil((bounds[3] + pad_m) / cell_m) * cell_m
    shape = (round((y1 - y0) / cell_m), round((x1 - x0) / cell_m))
    return Affine(cell_m, 0, x0, 0, -cell_m, y1), shape


def disc_kernel(radius_px: float) -> np.ndarray:
    """1 for cells whose centre lies within ``radius_px`` of the central cell."""
    r = int(radius_px)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x**2 + y**2 <= radius_px**2).astype(float)


def disc_mean(arr: np.ndarray, radius_px: float) -> np.ndarray:
    """Mean of ``arr`` over the disc around every cell. Near the array edge the
    mean is over the cells that exist."""
    k = disc_kernel(radius_px)
    return fftconvolve(arr, k, mode="same") / fftconvolve(np.ones_like(arr), k, mode="same")


def horn_gradient(dem: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """dz/dx (towards east) and dz/dy (towards north) by Horn's 3x3 method.
    Edge cells reuse their own row or column for the missing neighbour."""
    z = np.pad(dem, 1, mode="edge")
    dx = ((z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:])
          - (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])) / (8 * cell_m)
    dy = ((z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])
          - (z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:])) / (8 * cell_m)
    return dx, dy


def distance_to(mask: np.ndarray, cell_m: float, cap_m: float) -> np.ndarray:
    """Distance from every cell centre to the nearest True cell, capped at ``cap_m``."""
    if not mask.any():
        return np.full(mask.shape, float(cap_m))
    return np.minimum(distance_transform_edt(~mask) * cell_m, cap_m)


def sample(arr: np.ndarray, transform: Affine, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Values of the cells containing the points (x, y)."""
    # Written out because affine 3 deprecates `transform * (x, y)` and affine 2
    # has no `transform @ (x, y)`.
    inv = ~transform
    x, y = np.asarray(x, float), np.asarray(y, float)
    col, row = x * inv.a + y * inv.b + inv.c, x * inv.d + y * inv.e + inv.f
    r, c = np.floor(row).astype(int), np.floor(col).astype(int)
    if ((r < 0) | (r >= arr.shape[0]) | (c < 0) | (c >= arr.shape[1])).any():
        raise ValueError("point outside the grid")
    return arr[r, c]
