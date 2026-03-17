"""Feature extraction utilities for IR drop prediction.

Provides helpers for:
  - Channel normalization
  - Power pad distance computation (from SPICE netlist)
  - Local density computation
  - SPICE netlist parsing for supplementary features
"""

import re
import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_channel(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to [0, 1] range."""
    arr = arr.astype(np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - vmin) / (vmax - vmin)).astype(np.float32)


def compute_power_pad_distance(
    grid_h: int,
    grid_w: int,
    pad_locations: list,
    tile_size: float = 1.0,
) -> np.ndarray:
    """Compute effective distance to power pads for each tile.

    Uses the contest formula: eff_dist = 1 / sum(1/dist_to_each_source)
    """
    if not pad_locations:
        return np.ones((grid_h, grid_w), dtype=np.float32)

    y_coords, x_coords = np.mgrid[0:grid_h, 0:grid_w] * tile_size
    inv_dist_sum = np.zeros((grid_h, grid_w), dtype=np.float64)

    for pr, pc in pad_locations:
        dist = np.sqrt((y_coords - pr * tile_size) ** 2 + (x_coords - pc * tile_size) ** 2)
        dist = np.maximum(dist, tile_size * 0.5)
        inv_dist_sum += 1.0 / dist

    eff_dist = np.where(inv_dist_sum > 0, 1.0 / inv_dist_sum, 0.0)
    return eff_dist.astype(np.float32)


def compute_local_density(
    binary_map: np.ndarray,
    kernel_size: int = 11,
) -> np.ndarray:
    """Compute local density via Gaussian-weighted averaging."""
    sigma = kernel_size / 4.0
    density = gaussian_filter(binary_map.astype(np.float32), sigma=sigma)
    return normalize_channel(density)


def parse_spice_voltage_sources(netlist_path: str, unit_micron: int = 2000) -> list:
    """Extract voltage source locations from SPICE netlist.

    Parses lines like: V0 n1_m7_81000_106230 0 1.1
    Returns list of (row_um, col_um) tuples in micrometers.
    """
    vsrc_locations = []
    node_pattern = re.compile(r'n\d+_[mM]\d+_(\d+)_(\d+)')

    try:
        with open(netlist_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*') or line.startswith('.'):
                    continue
                if line.startswith('V') or line.startswith('v'):
                    match = node_pattern.search(line)
                    if match:
                        x_dbu = int(match.group(1))
                        y_dbu = int(match.group(2))
                        x_um = x_dbu / unit_micron
                        y_um = y_dbu / unit_micron
                        vsrc_locations.append((y_um, x_um))
    except (FileNotFoundError, IOError):
        pass

    return vsrc_locations


def stack_features(*feature_maps: np.ndarray) -> np.ndarray:
    """Stack multiple 2D feature maps into a multi-channel tensor [C, H, W]."""
    return np.stack([normalize_channel(f) for f in feature_maps], axis=0)
