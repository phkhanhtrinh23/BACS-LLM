from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .graph import SocialGraph
from .simulator import ATTENTION, DECISION, STATISTICS, Query, RunRecord

NUM_FEATURES = 5


@dataclass(frozen=True)
class ValueTables:
    attention: float
    mean_answer: np.ndarray
    reach: np.ndarray
    community: np.ndarray
    depth: np.ndarray
    depth_prefix: np.ndarray


def mean_answers(graph: SocialGraph, reference_runs: Sequence[RunRecord]) -> np.ndarray:
    sums = np.zeros((graph.num_agents, 2))
    counts = np.zeros((graph.num_agents, 2))
    for run in reference_runs:
        trials = run.trials
        index = np.flatnonzero(trials.kind == DECISION)
        probability = trials.probability[index]
        value = np.where(np.isnan(probability), trials.outcome[index].astype(float), probability)
        slot = trials.sharer_count[index] - 1
        np.add.at(sums, (trials.agent[index], slot), value)
        np.add.at(counts, (trials.agent[index], slot), 1.0)
    overall = sums.sum() / counts.sum() if counts.sum() else 0.0
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), overall)


def recurse(graph: SocialGraph, query: Query, mean_answer: np.ndarray, attention: float):
    n, rounds = graph.num_agents, query.rounds
    src, tgt = graph.sources, graph.targets
    edge = attention * mean_answer[tgt, 0]
    crossing = (graph.communities[src] != graph.communities[tgt]) / graph.num_communities
    reach = np.empty((rounds + 1, n))
    community = np.empty((rounds + 1, n))
    depth = np.empty((rounds + 1, n))
    reach[0], community[0], depth[0] = 1.0 / n, 0.0, 1.0
    for s in range(1, rounds + 1):
        reach[s] = 1.0 / n + np.bincount(src, edge * reach[s - 1][tgt], minlength=n)
        community[s] = np.bincount(src, edge * (crossing + community[s - 1][tgt]), minlength=n)
        miss = np.clip(1.0 - edge * depth[s - 1][tgt], 1e-300, 1.0)
        depth[s] = 1.0 - np.exp(np.bincount(src, np.log(miss), minlength=n))
    return reach, community, depth


def fit_value_tables(
    graph: SocialGraph,
    query: Query,
    reference_runs: Sequence[RunRecord],
    iterations: int = 40,
) -> ValueTables:
    mean_answer = mean_answers(graph, reference_runs)
    seeds = np.unique(np.asarray(query.seeds))
    target = np.mean([run.statistics[0] for run in reference_runs]) if reference_runs else 0.0

    def predicted(attention: float) -> float:
        reach, _, _ = recurse(graph, query, mean_answer, attention)
        return min(1.0, float(reach[query.rounds][seeds].sum()))

    low, high = 0.0, query.attention
    if predicted(high) > target:
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            if predicted(middle) > target:
                high = middle
            else:
                low = middle
    attention = high
    reach, community, depth = recurse(graph, query, mean_answer, attention)
    depth_prefix = np.vstack([np.zeros(graph.num_agents), np.cumsum(depth[1:], axis=0)])
    return ValueTables(attention, mean_answer, reach, community, depth, depth_prefix)


@dataclass(frozen=True)
class TrialFeatures:
    features: np.ndarray
    residual: np.ndarray
    step: np.ndarray


def trial_features(
    graph: SocialGraph,
    query: Query,
    tables: ValueTables,
    run: RunRecord,
    decision_trials: bool = True,
) -> TrialFeatures:
    n, rounds = graph.num_agents, query.rounds
    src, tgt, com = graph.sources, graph.targets, graph.communities
    trials = run.trials
    features = np.zeros((len(trials), NUM_FEATURES))
    residual = trials.outcome.astype(float) - np.nan_to_num(trials.probability, nan=0.0)
    step = np.maximum(trials.probability, 1.0 - trials.probability)
    unknown = np.isnan(trials.probability)
    residual[unknown] = 0.0
    step[unknown] = 0.0
    if not decision_trials:
        decision = trials.kind == DECISION
        residual[decision] = 0.0
        step[decision] = 0.0

    shared = np.zeros(n, dtype=bool)
    shared[np.asarray(query.seeds)] = True
    past = np.zeros(n, dtype=np.int64)
    reached = np.zeros(graph.num_communities, dtype=bool)
    reached[com[shared]] = True
    rows = np.arange(n)
    answer = tables.mean_answer

    for t in range(1, rounds + 1):
        in_round = np.flatnonzero(trials.round == t)
        if in_round.size == 0:
            break
        s = rounds - t
        open_ = ~shared & (past < 2)
        current = answer[rows, np.minimum(past, 1)]
        if s >= 1:
            edge = open_[tgt] * tables.attention * current[tgt]
            crossing = (~reached[com[tgt]] & (com[tgt] != com[src])) / graph.num_communities
            reach = 1.0 / n + np.bincount(src, edge * tables.reach[s - 1][tgt], minlength=n)
            community = np.bincount(src, edge * (crossing + tables.community[s - 1][tgt]), minlength=n)
        else:
            reach = np.full(n, 1.0 / n)
            community = np.zeros(n)
        depth = tables.depth_prefix[s] / rounds

        attention = in_round[trials.kind[in_round] == ATTENTION]
        receivers = trials.agent[attention]
        hits = trials.outcome[attention].astype(np.int64)
        starts = np.r_[True, receivers[1:] != receivers[:-1]] if receivers.size else np.zeros(0, dtype=bool)
        group = np.cumsum(starts) - 1
        cumulative = np.cumsum(hits)
        offset = np.r_[0, cumulative][np.flatnonzero(starts)][group] if receivers.size else hits
        pending = cumulative - hits - offset
        before = past[receivers]
        gain = np.zeros(receivers.size)
        first = (before == 0) & (pending == 0)
        second = (before == 0) & (pending == 1)
        late = (before == 1) & (pending == 0)
        gain[first] = answer[receivers[first], 0]
        gain[second] = answer[receivers[second], 1] - answer[receivers[second], 0]
        gain[late] = answer[receivers[late], 1]
        features[attention] = gain[:, None] * np.column_stack(
            [
                reach[receivers],
                community[receivers],
                depth[receivers],
                np.full(receivers.size, 1.0 / rounds),
                np.full(receivers.size, 1.0 / n),
            ]
        )

        decisions = in_round[trials.kind[in_round] == DECISION]
        deciders = trials.agent[decisions]
        outcomes = trials.outcome[decisions]
        nobody = np.r_[True, np.cumsum(outcomes)[:-1] == 0] if decisions.size else np.zeros(0, dtype=bool)
        features[decisions] = np.column_stack(
            [reach[deciders], community[deciders], depth[deciders], nobody / rounds, np.full(deciders.size, 1.0 / n)]
        )

        gained = np.bincount(receivers[hits.astype(bool)], minlength=n)
        past = np.minimum(past + gained, 2)
        shared[deciders[outcomes]] = True
        reached[com[shared]] = True

    return TrialFeatures(features, residual, step)


@dataclass(frozen=True)
class TrialControl:
    tables: ValueTables | None
    coefficients: np.ndarray
    cap: np.ndarray
    decision_trials: bool = True

    @property
    def active(self) -> np.ndarray:
        return np.any(self.coefficients != 0, axis=1) & (self.cap > 0)

    @property
    def bound(self) -> np.ndarray:
        return np.where(self.active, self.cap, 0.0)

    def summed_features(self, graph: SocialGraph, query: Query, run: RunRecord) -> np.ndarray:
        trial = trial_features(graph, query, self.tables, run, self.decision_trials)
        return trial.features.T @ trial.residual

    def __call__(self, graph: SocialGraph, query: Query, run: RunRecord) -> np.ndarray:
        if self.tables is None or not self.active.any():
            return np.zeros(len(STATISTICS))
        trial = trial_features(graph, query, self.tables, run, self.decision_trials)
        weights = trial.features @ self.coefficients.T
        increments = weights * trial.residual[:, None]
        before = np.vstack([np.zeros(len(STATISTICS)), np.cumsum(increments, axis=0)])
        reach = np.abs(before[:-1]) + trial.step[:, None] * np.abs(weights)
        control = np.empty(len(STATISTICS))
        for k in range(len(STATISTICS)):
            if not self.active[k]:
                control[k] = 0.0
                continue
            crossing = np.flatnonzero(reach[:, k] > self.cap[k])
            control[k] = before[crossing[0], k] if crossing.size else before[-1, k]
        return control


def no_control() -> TrialControl:
    k = len(STATISTICS)
    return TrialControl(None, np.zeros((k, NUM_FEATURES)), np.zeros(k))


def ridge(features: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    gram = features.T @ features
    scale = np.trace(gram) / gram.shape[0] if gram.size else 0.0
    return np.linalg.solve(gram + penalty * max(scale, 1e-12) * np.eye(gram.shape[0]), features.T @ target)


def cross_validated_gain(features: np.ndarray, target: np.ndarray, penalty: float, folds: int) -> float:
    n = target.size
    fold = np.arange(n) % folds
    with_control, without = 0.0, 0.0
    for f in range(folds):
        train, test = fold != f, fold == f
        if not test.any() or train.sum() < 2:
            continue
        mean = target[train].mean()
        beta = ridge(features[train], target[train] - mean, penalty)
        with_control += np.sum((target[test] - mean - features[test] @ beta) ** 2)
        without += np.sum((target[test] - mean) ** 2)
    if without <= 0:
        return 0.0
    return 1.0 - with_control / without


def fit_control(
    graph: SocialGraph,
    query: Query,
    reference_runs: Sequence[RunRecord],
    cap: np.ndarray,
    penalty: float = 1e-3,
    required_gain: float = 0.05,
    folds: int = 5,
    decision_trials: bool = True,
) -> TrialControl:
    k = len(STATISTICS)
    tables = fit_value_tables(graph, query, reference_runs)
    probe = TrialControl(tables, np.zeros((k, NUM_FEATURES)), np.asarray(cap, dtype=float), decision_trials)
    features = np.array([probe.summed_features(graph, query, run) for run in reference_runs])
    targets = np.array([run.statistics for run in reference_runs])
    coefficients = np.zeros((k, NUM_FEATURES))
    if len(reference_runs) >= folds:
        for index in range(k):
            target = targets[:, index]
            if cross_validated_gain(features, target, penalty, folds) >= required_gain:
                coefficients[index] = ridge(features, target - target.mean(), penalty)
    return TrialControl(tables, coefficients, probe.cap, decision_trials)
