from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import SocialGraph


@dataclass(frozen=True)
class Substrate:
    attention: np.ndarray
    decision: np.ndarray
    seed: int

    @classmethod
    def draw(cls, graph: SocialGraph, root_seed: int, index: int) -> "Substrate":
        rng = np.random.default_rng([root_seed, index])
        return cls(
            attention=rng.random(graph.num_edges),
            decision=rng.random((graph.num_agents, 2)),
            seed=int(rng.integers(2**62)),
        )
