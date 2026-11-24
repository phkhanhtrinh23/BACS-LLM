from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .graph import SocialGraph
from .models import Answer, ModelPair, Prompt
from .simulator import DECISION, FULL, RunRecord


class TierRule(Protocol):
    def answer(self, prompt: Prompt, models: ModelPair) -> Answer: ...


@dataclass(frozen=True)
class AllModel:
    tier: str

    def answer(self, prompt: Prompt, models: ModelPair) -> Answer:
        return getattr(models, self.tier).answer(prompt)


@dataclass(frozen=True)
class AgentRouting:
    escalated: frozenset[int]

    def answer(self, prompt: Prompt, models: ModelPair) -> Answer:
        model = models.frontier if prompt.agent in self.escalated else models.cheap
        return model.answer(prompt)


@dataclass(frozen=True)
class ConfidenceRouting:
    low: float = 0.2
    high: float = 0.8

    def answer(self, prompt: Prompt, models: ModelPair) -> Answer:
        cheap = models.cheap.answer(prompt)
        if cheap.probability is None or not self.low <= cheap.probability <= self.high:
            return cheap
        frontier = models.frontier.answer(prompt)
        return Answer(frontier.probability, cheap.cost + frontier.cost, frontier.shared)


@dataclass(frozen=True)
class Calibrated:
    calibrator: Callable[[float], float]

    def answer(self, prompt: Prompt, models: ModelPair) -> Answer:
        cheap = models.cheap.answer(prompt)
        if cheap.probability is None:
            return cheap
        return Answer(float(self.calibrator(cheap.probability)), cheap.cost)


@dataclass(frozen=True)
class Configuration:
    name: str
    tier: TierRule
    channel: str = FULL


def reference_configuration() -> Configuration:
    return Configuration("reference", AllModel("frontier"), FULL)


def all_cheap(channel: str = FULL) -> Configuration:
    return Configuration(f"all-cheap/{channel}", AllModel("cheap"), channel)


def hub_frontier(graph: SocialGraph, fraction: float = 0.1, channel: str = FULL) -> Configuration:
    count = max(1, int(round(fraction * graph.num_agents)))
    hubs = np.argsort(-graph.out_degree, kind="stable")[:count]
    return Configuration(f"hub-frontier/{channel}", AgentRouting(frozenset(int(v) for v in hubs)), channel)


def confidence_routing(low: float = 0.2, high: float = 0.8, channel: str = FULL) -> Configuration:
    return Configuration(f"confidence-routing/{channel}", ConfidenceRouting(low, high), channel)


class IsotonicMap:
    def __init__(self, knots: np.ndarray, values: np.ndarray):
        self.knots = knots
        self.values = values

    def __call__(self, probability: float) -> float:
        return float(np.interp(probability, self.knots, self.values))


def fit_isotonic(cheap: Sequence[float], frontier: Sequence[float]) -> IsotonicMap:
    cheap = np.asarray(cheap, dtype=float)
    frontier = np.asarray(frontier, dtype=float)
    order = np.argsort(cheap, kind="stable")
    x, y = cheap[order], frontier[order]
    means: list[float] = []
    weights: list[int] = []
    for value in y:
        means.append(float(value))
        weights.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            mean = (means[-2] * weights[-2] + means[-1] * weights[-1]) / weight
            means[-2:], weights[-2:] = [mean], [weight]
    fitted = np.repeat(means, weights)
    knots, first = np.unique(x, return_index=True)
    last = np.r_[first[1:], x.size] - 1
    values = np.array([fitted[a : b + 1].mean() for a, b in zip(first, last)])
    return IsotonicMap(knots, np.clip(values, 0.0, 1.0))


def calibrated(calibrator: Callable[[float], float], channel: str = FULL) -> Configuration:
    return Configuration(f"calibrated/{channel}", Calibrated(calibrator), channel)


@dataclass(frozen=True)
class ShadowStatistics:
    disagreement: np.ndarray
    extra_cost: np.ndarray
    cost: float


def shadow_calls(graph: SocialGraph, reference_runs: Sequence[RunRecord], models: ModelPair, message: str) -> ShadowStatistics:
    n = graph.num_agents
    disagreement = np.zeros(n)
    extra_cost = np.zeros(n)
    spent = 0.0
    for run in reference_runs:
        trials = run.trials
        for index in np.flatnonzero(trials.kind == DECISION):
            agent = int(trials.agent[index])
            prompt = Prompt(agent, graph.personas[agent], message, FULL, int(trials.sharer_count[index]))
            cheap = models.cheap.answer(prompt)
            spent += cheap.cost
            if cheap.probability is not None and not np.isnan(trials.probability[index]):
                disagreement[agent] += abs(cheap.probability - trials.probability[index])
            extra_cost[agent] += max(trials.cost[index] - cheap.cost, 0.0)
    runs = max(len(reference_runs), 1)
    return ShadowStatistics(disagreement / runs, extra_cost / runs, spent)


def disagreement_routing(
    shadow: ShadowStatistics,
    fractions: Sequence[float] = (0.1, 0.25, 0.5),
    channel: str = FULL,
) -> list[Configuration]:
    gamma = np.maximum(shadow.extra_cost, 1e-12)
    order = np.argsort(-shadow.disagreement / gamma, kind="stable")
    spend = np.cumsum(shadow.extra_cost[order])
    total = spend[-1] if spend.size else 0.0
    configurations = []
    for fraction in fractions:
        count = int(np.searchsorted(spend, fraction * total, side="right"))
        escalated = frozenset(int(v) for v in order[:count] if shadow.extra_cost[v] > 0)
        name = f"disagreement-{int(round(100 * fraction))}/{channel}"
        configurations.append(Configuration(name, AgentRouting(escalated), channel))
    return configurations


def standard_candidates(
    graph: SocialGraph,
    calibrator: Callable[[float], float] | None = None,
    channel: str = FULL,
    hub_fraction: float = 0.1,
    confidence_band: tuple[float, float] = (0.2, 0.8),
) -> list[Configuration]:
    candidates = [
        all_cheap(channel),
        hub_frontier(graph, hub_fraction, channel),
        confidence_routing(*confidence_band, channel=channel),
    ]
    if calibrator is not None:
        candidates.append(calibrated(calibrator, channel))
    return candidates
