import numpy as np
import pytest

from bacs import COMPRESSED, FULL, ModelPair, Query, SocialGraph, TableModel


def planted_partition(
    num_agents: int = 300,
    num_communities: int = 6,
    inside: float = 0.08,
    outside: float = 0.004,
    seed: int = 0,
) -> SocialGraph:
    rng = np.random.default_rng(seed)
    communities = rng.integers(num_communities, size=num_agents)
    same = communities[:, None] == communities[None, :]
    chance = np.where(same, inside, outside)
    upper = np.triu(rng.random((num_agents, num_agents)) < chance, k=1)
    edges = np.argwhere(upper)
    return SocialGraph.from_edges(num_agents, edges, communities, directed=False)


def answer_tables(
    graph: SocialGraph,
    cheap_shift: float = 0.6,
    compressed_shift: float = -0.4,
    noise: float = 0.5,
    seed: int = 0,
    frontier_cost: float = 1.0,
    cheap_cost: float = 0.05,
) -> ModelPair:
    rng = np.random.default_rng(seed)
    n = graph.num_agents
    base = rng.normal(-2.2, 1.5, size=n)
    logits = np.column_stack([base, base + 0.8])
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    frontier = {
        FULL: sigmoid(logits),
        COMPRESSED: sigmoid(logits + compressed_shift),
    }
    drift = rng.normal(cheap_shift, noise, size=(n, 1))
    cheap = {
        FULL: sigmoid(0.7 * logits + drift),
        COMPRESSED: sigmoid(0.7 * logits + drift + compressed_shift),
    }
    return ModelPair(TableModel(cheap, cheap_cost), TableModel(frontier, frontier_cost))


@pytest.fixture(scope="module")
def world():
    graph = planted_partition(num_agents=200, seed=2)
    models = answer_tables(graph, seed=2)
    query = Query("post", (0, 1, 2), 6, 0.3, headline="short")
    return graph, models, query
