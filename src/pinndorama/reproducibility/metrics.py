"""Numerical metrics used by every publication comparison."""

from __future__ import annotations

import numpy as np


def volume_weighted_relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    volume_weights: np.ndarray,
) -> float:
    """Return ``sqrt(sum(w*(prediction-reference)^2) / sum(w*reference^2))``.

    The arrays are flattened deliberately: callers may provide a structured
    native grid or an explicitly weighted point cloud, but the metric and its
    validation are identical.
    """

    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    volume_weights = np.asarray(volume_weights, dtype=np.float64).reshape(-1)
    if not (prediction.shape == reference.shape == volume_weights.shape):
        raise ValueError(
            "prediction, reference, and volume_weights must have identical sizes"
        )
    if prediction.size == 0:
        raise ValueError("relative L2 requires at least one sample")
    for name, values in (
        ("prediction", prediction),
        ("reference", reference),
        ("volume_weights", volume_weights),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(volume_weights < 0.0):
        raise ValueError("volume_weights must be non-negative")
    if not np.any(volume_weights > 0.0):
        raise ValueError("at least one volume weight must be positive")

    numerator = np.sum(
        volume_weights * np.square(prediction - reference), dtype=np.float64
    )
    denominator = np.sum(volume_weights * np.square(reference), dtype=np.float64)
    if denominator <= 0.0:
        raise ValueError("weighted reference norm is zero")
    return float(np.sqrt(numerator / denominator))
