from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Quote:
    ratio: float
    block_size: int
    coefficient: np.ndarray


def margin(pairs: int) -> float:
    if pairs < 2:
        return 0.0
    return 1.0 / (1.0 + 2.0 * math.sqrt(2.0 / (pairs - 1)))


def regression_coefficient(surrogate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if surrogate.shape[0] < 2:
        return np.zeros(surrogate.shape[1])
    variance = surrogate.var(axis=0, ddof=1)
    covariance = ((surrogate - surrogate.mean(0)) * (reference - reference.mean(0))).sum(0) / (surrogate.shape[0] - 1)
    coefficient = np.divide(covariance, variance, out=np.zeros_like(variance), where=variance > 0)
    return np.clip(coefficient, 0.0, 2.0)


def block_variance(surrogate: np.ndarray, reference: np.ndarray, coefficient: np.ndarray, size: int) -> np.ndarray:
    if surrogate.shape[0] < 2:
        return np.full(surrogate.shape[1], np.inf)
    difference = (reference - coefficient * surrogate).var(axis=0, ddof=1)
    variance = surrogate.var(axis=0, ddof=1)
    covariance = ((surrogate - surrogate.mean(0)) * (reference - reference.mean(0))).sum(0) / (surrogate.shape[0] - 1)
    return difference + coefficient / size * (2.0 * covariance - coefficient * variance)


def price(
    surrogate: np.ndarray,
    reference: np.ndarray,
    price_ratio: float,
    sizes: range = range(2, 65),
) -> Quote:
    coefficient = regression_coefficient(surrogate, reference)
    if reference.shape[0] < 2:
        return Quote(math.inf, 1, coefficient)
    baseline = reference.var(axis=0, ddof=1)
    valid = baseline > 1e-12
    if not valid.any():
        return Quote(math.inf, 1, coefficient)
    best = Quote(math.inf, 1, coefficient)
    for size in sizes:
        variance = block_variance(surrogate, reference, coefficient, size)
        ratio = float(np.max((size * price_ratio + 1.0) * variance[valid] / baseline[valid]))
        if ratio < best.ratio:
            best = Quote(ratio, size, coefficient)
    return best
