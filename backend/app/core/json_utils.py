"""
BharatSR — JSON Sanitization Utilities
Ensures all numeric values conform strictly to standard JSON specifications (no NaN or Infinity literals).
"""

import math
from typing import Any
import numpy as np


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize data structures for standard JSON compliance.
    Replaces float('nan'), float('inf'), float('-inf') with None (JSON null).
    Converts numpy scalars to native Python types.
    """
    if obj is None:
        return None
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or np.isnan(obj) or math.isinf(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (str, bytes)):
        return obj if isinstance(obj, str) else obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj
