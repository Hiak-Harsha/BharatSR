"""
Physics validation for Sentinel-2 spectral indices: NDVI, NDWI, EVI, SAVI, RVI, GCI.
"""

import numpy as np
from backend.app.services.preprocessing import compute_spectral_indices, BAND_INDEX


def test_ndvi_range_minus1_to_1():
    # Canonical band order: B2, B3, B4, B8
    img = np.array([
        [[0.1, 0.1]],  # B2 Blue
        [[0.1, 0.1]],  # B3 Green
        [[0.1, 0.6]],  # B4 Red
        [[0.8, 0.2]],  # B8 NIR
    ], dtype=np.float32)

    indices = compute_spectral_indices(img)
    ndvi = indices["ndvi"]
    assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)
    # High NIR / low Red should have high NDVI
    assert ndvi[0, 0] > 0.7
    # Low NIR / high Red should have negative NDVI
    assert ndvi[0, 1] < 0.0


def test_ndwi_range_minus1_to_1():
    img = np.array([
        [[0.1]],  # B2
        [[0.7]],  # B3 Green
        [[0.1]],  # B4
        [[0.1]],  # B8 NIR
    ], dtype=np.float32)

    indices = compute_spectral_indices(img)
    ndwi = indices["ndwi"]
    assert -1.0 <= ndwi[0, 0] <= 1.0
    # Water has high green and low NIR -> positive NDWI
    assert ndwi[0, 0] > 0.5


def test_evi_formula_matches_paper():
    # EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    blue, green, red, nir = 0.05, 0.10, 0.08, 0.50
    img = np.array([[[blue]], [[green]], [[red]], [[nir]]], dtype=np.float32)

    indices = compute_spectral_indices(img)
    expected_evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-7)
    assert np.isclose(indices["evi"][0, 0], expected_evi, atol=1e-5)


def test_indices_consistent_with_band_order():
    # Verify index mapping matches BAND_INDEX
    assert BAND_INDEX["B2"] == 0
    assert BAND_INDEX["B3"] == 1
    assert BAND_INDEX["B4"] == 2
    assert BAND_INDEX["B8"] == 3

    img = np.zeros((4, 8, 8), dtype=np.float32)
    indices = compute_spectral_indices(img)
    for key in ["ndvi", "ndwi", "evi", "savi", "rvi", "ndbi_approx", "gci"]:
        assert key in indices
        assert indices[key].shape == (8, 8)
