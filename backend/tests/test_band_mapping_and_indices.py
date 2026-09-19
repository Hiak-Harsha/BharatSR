"""
Tests for Canonical Band Mapping, Reflectance Normalization, and Spectral Indices
Verifies:
1. Canonical BAND_INDEX = {"B2": 0, "B3": 1, "B4": 2, "B8": 3}
2. Physical NDVI calculation with known synthetic values (e.g. B4=0.2, B8=0.6 -> NDVI=0.5)
3. True Color composition (R=B4, G=B3, B=B2)
4. Color Infrared (CIR) composition (R=B8, G=B4, B=B3)
5. Rejection of arbitrary 3-band RGB uploads without zero-padding ("Sentinel-2 model requires B2/B3/B4/B8.")
6. Explicit product-aware reflectance normalization modes:
   - sentinel2_l2a_dn
   - reflectance_float
   - uint8_rgb
"""

import io
import pytest
import numpy as np
from PIL import Image

from backend.app.services.preprocessing import (
    BAND_INDEX,
    BAND_NAMES,
    normalize_reflectance,
    load_image_from_bytes,
    generate_multi_spectral_views,
    numpy_to_png_bytes,
)
from evaluation.downstream_task import (
    segment_micro_canopy_rule_based,
    segment_built_up_rule_based,
)


def test_canonical_band_definitions():
    """Verify standard Sentinel-2 Level-2A band indices and names."""
    assert BAND_INDEX["B2"] == 0, "B2 (Blue) must be index 0"
    assert BAND_INDEX["B3"] == 1, "B3 (Green) must be index 1"
    assert BAND_INDEX["B4"] == 2, "B4 (Red) must be index 2"
    assert BAND_INDEX["B8"] == 3, "B8 (NIR) must be index 3"
    assert len(BAND_INDEX) == 4


def test_ndvi_with_known_synthetic_values():
    """
    Test NDVI with exact known values:
    B4 (Red) = 0.2
    B8 (NIR) = 0.6
    Expected NDVI = (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    """
    img = np.zeros((4, 16, 16), dtype=np.float32)
    img[BAND_INDEX["B2"]] = 0.1  # Blue
    img[BAND_INDEX["B3"]] = 0.15 # Green
    img[BAND_INDEX["B4"]] = 0.2  # Red
    img[BAND_INDEX["B8"]] = 0.6  # NIR

    nir = img[BAND_INDEX["B8"]]
    red = img[BAND_INDEX["B4"]]
    ndvi = (nir - red) / (nir + red + 1e-7)

    assert np.allclose(ndvi, 0.5, atol=1e-4), f"Expected NDVI 0.5, got {ndvi[0, 0]}"

    # Verify downstream rule-based canopy extraction
    # Since NDVI = 0.5 > 0.35, all pixels must be classified as vegetation canopy
    canopy_mask = segment_micro_canopy_rule_based(img, ndvi_threshold=0.35)
    assert (canopy_mask == 1).all()


def test_true_color_and_cir_composition():
    """
    Verify True Color uses (B4, B3, B2) and CIR uses (B8, B4, B3).
    """
    # Create an image where each band has a distinct value
    img = np.zeros((4, 8, 8), dtype=np.float32)
    img[BAND_INDEX["B2"]] = 0.1  # Blue -> 0.1
    img[BAND_INDEX["B3"]] = 0.5  # Green -> 0.5
    img[BAND_INDEX["B4"]] = 0.9  # Red -> 0.9
    img[BAND_INDEX["B8"]] = 0.7  # NIR -> 0.7

    views = generate_multi_spectral_views(img)
    assert "rgb" in views
    assert "cir" in views
    assert "ndvi" in views
    assert "b2" in views
    assert "b3" in views
    assert "b4" in views
    assert "b8" in views

    # Test numpy_to_png_bytes True Color mapping
    png_bytes = numpy_to_png_bytes(img)
    pil_img = Image.open(io.BytesIO(png_bytes))
    arr = np.array(pil_img)

    # R channel should reflect B4 (0.9 * 255 = 229)
    # G channel should reflect B3 (0.5 * 255 = 127)
    # B channel should reflect B2 (0.1 * 255 = 25)
    assert abs(arr[0, 0, 0] - int(0.9 * 255)) <= 2
    assert abs(arr[0, 0, 1] - int(0.5 * 255)) <= 2
    assert abs(arr[0, 0, 2] - int(0.1 * 255)) <= 2


def test_reject_arbitrary_rgb_without_padding():
    """
    Verify that arbitrary 3-band RGB imagery is strictly REJECTED
    instead of being silently padded with zero channels.
    """
    # Create 3-channel RGB image bytes
    rgb_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    pil_img = Image.fromarray(rgb_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    with pytest.raises(ValueError) as exc_info:
        load_image_from_bytes(png_bytes, expected_bands=4)

    assert "Sentinel-2 model requires B2/B3/B4/B8." in str(exc_info.value)


def test_explicit_input_modes_and_product_aware_scaling():
    """
    Verify explicit product-aware normalization modes:
    - sentinel2_l2a_dn: scaled by 10000.0
    - reflectance_float: values preserved, clouds > 1.0 kept
    - uint8_rgb: scaled by 255.0
    """
    # 1. sentinel2_l2a_dn
    dn_arr = np.array([1000.0, 5000.0, 10000.0, 12000.0], dtype=np.float32)
    norm_dn = normalize_reflectance(dn_arr, mode="sentinel2_l2a_dn")
    assert np.allclose(norm_dn, [0.1, 0.5, 1.0, 1.2])

    # 2. reflectance_float
    float_arr = np.array([0.05, 0.45, 0.85, 1.15], dtype=np.float32)
    norm_float = normalize_reflectance(float_arr, mode="reflectance_float")
    assert np.allclose(norm_float, float_arr)
    assert norm_float[3] > 1.0, "Bright targets > 1.0 must NOT be clipped!"

    # 3. uint8_rgb
    uint8_arr = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    norm_u8 = normalize_reflectance(uint8_arr, mode="uint8_rgb")
    assert np.allclose(norm_u8, [0.0, 0.5, 1.0])

    # 4. Unknown mode raises ValueError
    with pytest.raises(ValueError):
        normalize_reflectance(float_arr, mode="unknown_invalid_mode")
