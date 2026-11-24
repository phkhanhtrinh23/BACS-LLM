from __future__ import annotations

import math
from enum import Enum

import numpy as np


class Decision(str, Enum):
    PRESERVED = "preserved"
    NOT_PRESERVED = "not preserved"
    ABSTAIN = "abstain"


def decide(interval: tuple[float, float] | None, tolerance: float) -> Decision:
    if interval is None:
        return Decision.ABSTAIN
    low, high = interval
    if -tolerance <= low and high <= tolerance:
        return Decision.PRESERVED
    if high < -tolerance or low > tolerance:
        return Decision.NOT_PRESERVED
    return Decision.ABSTAIN


class BettingSequence:
    def __init__(self, level: float, low: float, high: float, grid: int = 4001, truncation: float = 0.5):
        self.level = level
        self.truncation = truncation
        self.means = np.linspace(low, high, grid)
        self.log_up = np.zeros(grid)
        self.log_down = np.zeros(grid)
        self.count = 0
        self.total = 0.0
        self.squares = 0.0
        self.low = low
        self.high = high
        self.frozen = False

    def bet(self, low: float, high: float) -> float:
        prior_variance = 0.25 * (high - low) ** 2
        variance = (prior_variance + self.squares) / (self.count + 1)
        t = self.count + 1
        return math.sqrt(2.0 * math.log(2.0 / self.level) / (max(variance, 1e-12) * t * math.log(t + 1.0)))

    def update(self, value: float, low: float, high: float) -> None:
        if self.frozen:
            return
        size = self.bet(low, high)
        m = self.means
        up = np.full_like(m, size)
        above = m > low
        up[above] = np.minimum(size, self.truncation / (m[above] - low))
        down = np.full_like(m, size)
        below = m < high
        down[below] = np.minimum(size, self.truncation / (high - m[below]))
        self.log_up += np.log1p(up * (value - m))
        self.log_down += np.log1p(-down * (value - m))
        prior_mean = 0.5 * (low + high)
        predicted = (prior_mean + self.total) / (self.count + 1)
        self.squares += (value - predicted) ** 2
        self.total += value
        self.count += 1
        wealth = np.logaddexp(self.log_up, self.log_down) - math.log(2.0)
        kept = self.means[wealth < math.log(1.0 / self.level)]
        if kept.size == 0:
            middle = 0.5 * (self.low + self.high)
            kept = np.array([min(max(middle, self.low), self.high)])
        self.low = max(self.low, float(kept[0]))
        self.high = min(self.high, float(kept[-1]))
        if self.low > self.high:
            self.low = self.high = 0.5 * (self.low + self.high)

    @property
    def interval(self) -> tuple[float, float] | None:
        if self.count == 0:
            return None
        return self.low, self.high


class AsymptoticSequence:
    def __init__(self, level: float, low: float, high: float, rho_squared: float = 0.1):
        self.level = level
        self.rho_squared = rho_squared
        self.values: list[float] = []
        self.low = low
        self.high = high
        self.frozen = False

    def update(self, value: float, low: float, high: float) -> None:
        if not self.frozen:
            self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def interval(self) -> tuple[float, float] | None:
        n = len(self.values)
        if n < 2:
            return None
        values = np.asarray(self.values)
        scale = n * self.rho_squared + 1.0
        width = values.std(ddof=1) * math.sqrt(
            2.0 * scale / (n**2 * self.rho_squared) * math.log(math.sqrt(scale) / self.level)
        )
        center = values.mean()
        return max(self.low, center - width), min(self.high, center + width)


def make_sequence(asymptotic: bool, level: float, low: float, high: float):
    if asymptotic:
        return AsymptoticSequence(level, low, high)
    return BettingSequence(level, low, high)
