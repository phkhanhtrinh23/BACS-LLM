from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .graph import SocialGraph
from .models import ModelPair, Prompt
from .substrate import Substrate

if TYPE_CHECKING:
    from .configuration import Configuration

STATISTICS = ("reach", "community_reach", "depth")
ATTENTION = 0
DECISION = 1
FULL = "full"
COMPRESSED = "compressed"


@dataclass(frozen=True)
class Query:
    text: str
    seeds: tuple[int, ...]
    rounds: int
    attention: float
    headline: str | None = None

    def delivered(self, channel: str) -> str:
        if channel == FULL:
            return self.text
        if channel == COMPRESSED:
            if self.headline is None:
                raise ValueError("the compressed channel needs a headline for this query")
            return self.headline
        raise ValueError(f"unknown channel: {channel}")


@dataclass(frozen=True)
class TrialLog:
    kind: np.ndarray
    round: np.ndarray
    agent: np.ndarray
    source: np.ndarray
    outcome: np.ndarray
    probability: np.ndarray
    sharer_count: np.ndarray
    cost: np.ndarray

    def __len__(self) -> int:
        return int(self.kind.size)


@dataclass(frozen=True, eq=False)
class RunRecord:
    configuration: str
    statistics: np.ndarray
    cost: float
    trials: TrialLog


def simulate(
    graph: SocialGraph,
    query: Query,
    configuration: "Configuration",
    models: ModelPair,
    substrate: Substrate,
) -> RunRecord:
    n = graph.num_agents
    message = query.delivered(configuration.channel)
    seeds = np.unique(np.asarray(query.seeds, dtype=np.int64))
    shared = np.zeros(n, dtype=bool)
    shared[seeds] = True
    noticed = np.zeros(n, dtype=np.int64)
    active = seeds
    depth = 0
    total_cost = 0.0
    columns: dict[str, list[np.ndarray]] = {name: [] for name in TrialLog.__dataclass_fields__}

    def log(kind, round_index, agent, source, outcome, probability, sharer_count, cost):
        size = len(agent)
        columns["kind"].append(np.full(size, kind, dtype=np.int8))
        columns["round"].append(np.full(size, round_index, dtype=np.int64))
        columns["agent"].append(np.asarray(agent, dtype=np.int64))
        columns["source"].append(np.asarray(source, dtype=np.int64))
        columns["outcome"].append(np.asarray(outcome, dtype=bool))
        columns["probability"].append(np.asarray(probability, dtype=float))
        columns["sharer_count"].append(np.asarray(sharer_count, dtype=np.int64))
        columns["cost"].append(np.asarray(cost, dtype=float))

    for round_index in range(1, query.rounds + 1):
        if active.size == 0:
            break
        edge_ids = np.concatenate([np.arange(graph.offsets[u], graph.offsets[u + 1]) for u in active])
        receivers = graph.targets[edge_ids]
        open_ = ~shared[receivers] & (noticed[receivers] < 2)
        edge_ids, receivers = edge_ids[open_], receivers[open_]
        order = np.lexsort((graph.sources[edge_ids], receivers))
        edge_ids, receivers = edge_ids[order], receivers[order]
        hit = substrate.attention[edge_ids] < query.attention
        size = edge_ids.size
        log(
            ATTENTION,
            round_index,
            receivers,
            graph.sources[edge_ids],
            hit,
            np.full(size, query.attention),
            np.zeros(size),
            np.zeros(size),
        )
        gained = np.bincount(receivers[hit], minlength=n)
        deciders = np.flatnonzero(gained)
        outcomes, probabilities, counts, costs = [], [], [], []
        for agent in deciders:
            prior = int(noticed[agent])
            total = prior + int(gained[agent])
            sharer_count = 1 if total == 1 else 2
            prompt = Prompt(
                int(agent),
                graph.personas[agent],
                message,
                configuration.channel,
                sharer_count,
                seed=substrate.seed,
            )
            answer = configuration.tier.answer(prompt, models)
            if answer.probability is None:
                outcomes.append(bool(answer.shared))
                probabilities.append(np.nan)
            else:
                outcomes.append(substrate.decision[agent, prior] < answer.probability)
                probabilities.append(answer.probability)
            counts.append(sharer_count)
            costs.append(answer.cost)
            noticed[agent] = min(total, 2)
            total_cost += answer.cost
        log(DECISION, round_index, deciders, np.full(deciders.size, -1), outcomes, probabilities, counts, costs)
        active = deciders[np.asarray(outcomes, dtype=bool)] if deciders.size else deciders
        shared[active] = True
        if active.size:
            depth = round_index

    trials = TrialLog(**{name: _concatenate(parts, name) for name, parts in columns.items()})
    statistics = np.array(
        [
            shared.sum() / n,
            np.unique(graph.communities[shared]).size / graph.num_communities,
            depth / query.rounds,
        ]
    )
    return RunRecord(configuration.name, statistics, total_cost, trials)


def _concatenate(parts: list[np.ndarray], name: str) -> np.ndarray:
    if parts:
        return np.concatenate(parts)
    dtype = {"kind": np.int8, "outcome": bool, "probability": float, "cost": float}.get(name, np.int64)
    return np.zeros(0, dtype=dtype)
