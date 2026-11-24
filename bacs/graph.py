from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SocialGraph:
    num_agents: int
    sources: np.ndarray
    targets: np.ndarray
    offsets: np.ndarray
    communities: np.ndarray
    num_communities: int
    personas: tuple[str, ...]

    @classmethod
    def from_edges(
        cls,
        num_agents: int,
        edges: Sequence[Sequence[int]] | np.ndarray,
        communities: Sequence[int] | np.ndarray,
        personas: Sequence[str] | None = None,
        directed: bool = True,
    ) -> "SocialGraph":
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if edges.size and (edges.min() < 0 or edges.max() >= num_agents):
            raise ValueError("edge endpoints must lie in [0, num_agents)")
        if not directed:
            edges = np.concatenate([edges, edges[:, ::-1]])
        edges = edges[edges[:, 0] != edges[:, 1]]
        edges = np.unique(edges, axis=0)
        sources = np.ascontiguousarray(edges[:, 0])
        targets = np.ascontiguousarray(edges[:, 1])
        offsets = np.zeros(num_agents + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(np.bincount(sources, minlength=num_agents))
        communities = np.asarray(communities)
        if communities.shape != (num_agents,):
            raise ValueError("communities must hold one label per agent")
        labels, inverse = np.unique(communities, return_inverse=True)
        if personas is None:
            personas = [""] * num_agents
        if len(personas) != num_agents:
            raise ValueError("personas must hold one entry per agent")
        return cls(
            num_agents=num_agents,
            sources=sources,
            targets=targets,
            offsets=offsets,
            communities=inverse.astype(np.int64),
            num_communities=int(labels.size),
            personas=tuple(personas),
        )

    @property
    def num_edges(self) -> int:
        return int(self.sources.size)

    @property
    def out_degree(self) -> np.ndarray:
        return np.diff(self.offsets)

    def out_edges(self, agent: int) -> slice:
        return slice(int(self.offsets[agent]), int(self.offsets[agent + 1]))
