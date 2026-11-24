from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from weakref import WeakKeyDictionary

import numpy as np

from .certificate import Decision, decide, make_sequence
from .configuration import (
    AllModel,
    Configuration,
    all_cheap,
    disagreement_routing,
    reference_configuration,
    shadow_calls,
)
from .control import TrialControl, fit_control, no_control
from .graph import SocialGraph
from .models import MemoizedModel, ModelPair
from .pricing import block_variance, margin, price, regression_coefficient
from .simulator import DECISION, FULL, STATISTICS, Query, RunRecord, simulate
from .substrate import Substrate


@dataclass
class Settings:
    budget: float
    tolerances: tuple[float, ...] = (0.02, 0.05, 0.10)
    level: float = 0.05
    explore_runs: int = 3
    commit_runs: int = 10
    recheck_runs: int = 30
    epoch_blocks: int = 10
    block_sizes: range = range(2, 65)
    cap_deviations: float = 3.0
    ridge_penalty: float = 1e-3
    required_gain: float = 0.05
    use_control: bool = True
    decision_trials: bool = True
    split_bias: bool = False
    asymptotic: bool = False
    tabulate: bool = False
    max_blocks: int = 100_000
    seed: int = 0


@dataclass(frozen=True)
class StatisticReport:
    name: str
    estimate: float
    reference_interval: tuple[float, float] | None
    bias_interval: tuple[float, float] | None
    tier_interval: tuple[float, float] | None
    channel_interval: tuple[float, float] | None
    decision: Decision | None


@dataclass(frozen=True)
class Report:
    statistics: tuple[StatisticReport, ...]
    committed: str | None
    block_size: int
    certified: str | None
    spent: float
    blocks: int
    reference_runs: int
    tabulated: bool

    def __getitem__(self, name: str) -> StatisticReport:
        for statistic in self.statistics:
            if statistic.name == name:
                return statistic
        raise KeyError(name)


@dataclass
class Block:
    value: np.ndarray
    low: np.ndarray
    high: np.ndarray
    weight: np.ndarray
    after_commit: bool


@dataclass
class Plan:
    surrogate: Configuration | None = None
    size: int = 1
    coefficient: np.ndarray = field(default_factory=lambda: np.zeros(len(STATISTICS)))


class BudgetExhausted(Exception):
    pass


class BudgetedSimulation:
    def __init__(
        self,
        graph: SocialGraph,
        query: Query,
        models: ModelPair,
        settings: Settings,
        candidates: Sequence[Configuration] | None = None,
        certify: Configuration | str = "all-cheap",
        routing_fractions: Sequence[float] = (),
    ):
        self.graph = graph
        self.query = query
        self.models = models
        self.settings = settings
        self.reference = reference_configuration()
        self.candidates = list(candidates) if candidates is not None else [all_cheap()]
        self.routing_fractions = tuple(routing_fractions)
        self.named: Configuration | None = None
        self.certify_committed = False
        if isinstance(certify, Configuration):
            self.named = certify
        elif certify == "all-cheap":
            cheap = [x for x in self.candidates if isinstance(x.tier, AllModel) and x.tier.tier == "cheap"]
            self.named = cheap[0] if cheap else all_cheap()
        elif certify == "committed":
            self.certify_committed = True
        elif certify != "none":
            raise ValueError(f"unknown certify option: {certify}")

        k = len(STATISTICS)
        certifying = self.named is not None or self.certify_committed
        streams = 1 + (2 if settings.split_bias else 1) * certifying
        level = settings.level / (streams * k)
        sequence = lambda low, high: [make_sequence(settings.asymptotic, level, low, high) for _ in range(k)]
        self.reference_sequences = sequence(0.0, 1.0)
        self.bias_sequences = sequence(-1.0, 1.0)
        self.channel_sequences = sequence(-1.0, 1.0) if settings.split_bias else None

        self.certified = self.named
        self.bias_open = self.named is not None
        self.next_substrate = 0
        self.spent = 0.0
        self.costs: dict[str, list[float]] = {}
        self.reference_runs: list[RunRecord] = []
        self.pairs: dict[str, list[tuple[RunRecord, RunRecord]]] = {}
        self.blocks: list[Block] = []
        self.plan = Plan()
        self.committed: Configuration | None = None
        self.control: TrialControl = no_control()
        self.cap = np.zeros(k)
        self.tabulated = False
        self.cache: WeakKeyDictionary[RunRecord, np.ndarray] = WeakKeyDictionary()

    def substrate(self, index: int) -> Substrate:
        return Substrate.draw(self.graph, self.settings.seed, index)

    def fresh_substrate(self) -> Substrate:
        substrate = self.substrate(self.next_substrate)
        self.next_substrate += 1
        return substrate

    def expected_cost(self, configuration: Configuration) -> float:
        history = self.costs.get(configuration.name) or self.costs.get(self.reference.name) or [0.0]
        return float(np.mean(history[-10:]))

    def execute(self, configuration: Configuration, substrate: Substrate) -> RunRecord:
        if self.spent + self.expected_cost(configuration) > self.settings.budget:
            raise BudgetExhausted
        run = simulate(self.graph, self.query, configuration, self.models, substrate)
        self.spent += run.cost
        self.costs.setdefault(configuration.name, []).append(run.cost)
        if self.spent > self.settings.budget:
            raise BudgetExhausted
        return run

    def controlled(self, run: RunRecord) -> np.ndarray:
        if run not in self.cache:
            self.cache[run] = run.statistics - self.control(self.graph, self.query, run)
        return self.cache[run]

    def paired(self, configuration: Configuration, controlled: bool) -> tuple[np.ndarray, np.ndarray, float]:
        pairs = self.pairs.get(configuration.name, [])
        if not pairs:
            k = len(STATISTICS)
            return np.zeros((0, k)), np.zeros((0, k)), 1.0
        read = self.controlled if controlled else (lambda run: run.statistics)
        surrogate = np.array([read(x) for x, _ in pairs])
        reference = np.array([read(y) for _, y in pairs])
        reference_cost = np.mean([y.cost for _, y in pairs])
        ratio = np.mean([x.cost for x, _ in pairs]) / reference_cost if reference_cost > 0 else np.inf
        return surrogate, reference, float(ratio)

    def add_reference_block(self, run: RunRecord, weight: np.ndarray) -> None:
        bound = self.control.bound
        self.add_block(self.controlled(run), -bound, 1.0 + bound, weight)

    def add_block(self, value: np.ndarray, low: np.ndarray, high: np.ndarray, weight: np.ndarray) -> None:
        for k, sequence in enumerate(self.reference_sequences):
            sequence.update(float(value[k]), float(low[k]), float(high[k]))
        self.blocks.append(Block(value, low, high, weight, self.committed is not None))

    def frontier_on_channel(self, configuration: Configuration) -> Configuration:
        return Configuration(f"frontier/{configuration.channel}", AllModel("frontier"), configuration.channel)

    def bias_pair(self, surrogate: RunRecord, reference: RunRecord, substrate: Substrate) -> None:
        if self.channel_sequences is None:
            for k, sequence in enumerate(self.bias_sequences):
                sequence.update(float(surrogate.statistics[k] - reference.statistics[k]), -1.0, 1.0)
            return
        middle = reference
        if self.certified.channel != FULL:
            middle = self.execute(self.frontier_on_channel(self.certified), substrate)
        for k in range(len(STATISTICS)):
            self.bias_sequences[k].update(float(surrogate.statistics[k] - middle.statistics[k]), -1.0, 1.0)
            self.channel_sequences[k].update(float(middle.statistics[k] - reference.statistics[k]), -1.0, 1.0)

    def tier_interval(self, k: int) -> tuple[float, float] | None:
        if self.certified is None or self.channel_sequences is None:
            return None
        return self.bias_sequences[k].interval

    def channel_interval(self, k: int) -> tuple[float, float] | None:
        if self.certified is None or self.channel_sequences is None:
            return None
        return self.channel_sequences[k].interval

    def bias_interval(self, k: int) -> tuple[float, float] | None:
        if self.certified is None:
            return None
        if self.channel_sequences is None:
            return self.bias_sequences[k].interval
        tier = self.tier_interval(k)
        channel = self.channel_interval(k)
        if tier is None or channel is None:
            return None
        return tier[0] + channel[0], tier[1] + channel[1]

    def decisions(self) -> list[Decision | None]:
        if self.certified is None:
            return [None] * len(STATISTICS)
        return [decide(self.bias_interval(k), tol) for k, tol in enumerate(self.settings.tolerances)]

    def all_decided(self) -> bool:
        decisions = self.decisions()
        return self.certified is not None and all(d not in (None, Decision.ABSTAIN) for d in decisions)

    def needs_named_pair(self) -> bool:
        return self.certified is not None and self.bias_open and not self.all_decided()

    def freeze_bias(self) -> None:
        self.bias_open = False
        for sequence in self.bias_sequences + (self.channel_sequences or []):
            sequence.frozen = True

    def run_pair(self, configuration: Configuration, substrate: Substrate, reference: RunRecord) -> RunRecord:
        run = self.execute(configuration, substrate)
        self.pairs.setdefault(configuration.name, []).append((run, reference))
        if self.certified is not None and configuration.name == self.certified.name and self.bias_open:
            self.bias_pair(run, reference, substrate)
        return run

    def explore(self) -> None:
        s = self.settings
        ones = np.ones(len(STATISTICS))
        substrates = [self.fresh_substrate() for _ in range(s.commit_runs)]
        references = []
        for substrate in substrates[: s.explore_runs]:
            run = self.execute(self.reference, substrate)
            self.reference_runs.append(run)
            references.append(run)
            self.add_reference_block(run, ones)
        if self.routing_fractions:
            shadow = shadow_calls(self.graph, references, self.models, self.query.text)
            self.spent += shadow.cost
            channel = self.candidates[0].channel if self.candidates else FULL
            self.candidates += disagreement_routing(shadow, self.routing_fractions, channel)
        extra = [self.named] if self.named is not None and self.named not in self.candidates else []
        for substrate, reference in zip(substrates, references):
            for configuration in self.candidates + extra:
                self.run_pair(configuration, substrate, reference)

        survivors = [x for x in self.candidates if price(*self.paired(x, False)).ratio < 1.0]
        paired = survivors + [x for x in [self.named] if x is not None and x not in survivors]
        for substrate in substrates[s.explore_runs :]:
            reference = self.execute(self.reference, substrate)
            self.reference_runs.append(reference)
            self.add_reference_block(reference, ones)
            for configuration in paired:
                self.run_pair(configuration, substrate, reference)

        quotes = [(price(*self.paired(x, False), s.block_sizes), x) for x in survivors]
        if quotes:
            quote, best = min(quotes, key=lambda item: item[0].ratio)
            if quote.ratio < margin(len(self.pairs[best.name])):
                self.committed = best
                self.plan = Plan(best, quote.block_size, quote.coefficient)
                if self.certify_committed:
                    self.certified = best
                    self.bias_open = True

        first = np.array([run.statistics for run in self.reference_runs[: s.commit_runs]])
        if first.shape[0] >= 2:
            self.cap = s.cap_deviations * first.std(axis=0, ddof=1)

        if s.tabulate:
            self.maybe_tabulate()

    def maybe_tabulate(self) -> None:
        costs = np.concatenate([run.trials.cost[run.trials.kind == DECISION] for run in self.reference_runs])
        table_cost = 2 * self.graph.num_agents * (costs.mean() if costs.size else 0.0)
        if table_cost < self.settings.budget - self.spent:
            self.models = ModelPair(self.models.cheap, MemoizedModel(self.models.frontier))
            self.tabulated = True
            self.plan = Plan()
            if self.certify_committed and self.certified is not None:
                self.freeze_bias()

    def refit(self) -> None:
        s = self.settings
        self.cache = WeakKeyDictionary()
        if s.use_control and len(self.reference_runs) >= s.commit_runs:
            self.control = fit_control(
                self.graph,
                self.query,
                self.reference_runs,
                self.cap,
                penalty=s.ridge_penalty,
                required_gain=s.required_gain,
                decision_trials=s.decision_trials,
            )

    def epoch_weight(self) -> np.ndarray:
        if self.plan.surrogate is not None:
            surrogate, reference, _ = self.paired(self.plan.surrogate, True)
            variance = block_variance(surrogate, reference, self.plan.coefficient, self.plan.size)
            n = surrogate.shape[0]
        else:
            values = np.array([self.controlled(run) for run in self.reference_runs])
            variance = values.var(axis=0, ddof=1) if values.shape[0] >= 2 else np.full(len(STATISTICS), 0.25)
            n = values.shape[0]
        return (n + 1) / (0.25 + n * np.maximum(variance, 0.0))

    def replan(self, rechecked: bool) -> bool:
        s = self.settings
        if self.plan.surrogate is None:
            return rechecked
        surrogate, reference, ratio = self.paired(self.plan.surrogate, True)
        self.plan.coefficient = regression_coefficient(surrogate, reference)
        if not rechecked and surrogate.shape[0] >= s.recheck_runs:
            rechecked = True
            if price(surrogate, reference, ratio, s.block_sizes).ratio >= margin(surrogate.shape[0]):
                self.plan = Plan()
                if self.certify_committed:
                    self.freeze_bias()
        return rechecked

    def surrogate_block(self, weight: np.ndarray) -> None:
        plan = self.plan
        substrates = [self.fresh_substrate() for _ in range(plan.size)]
        reference = self.execute(self.reference, substrates[0])
        runs = [self.run_pair(plan.surrogate, substrates[0], reference)]
        runs += [self.execute(plan.surrogate, substrate) for substrate in substrates[1:]]
        if self.needs_named_pair() and self.certified.name != plan.surrogate.name:
            self.run_pair(self.certified, substrates[0], reference)
        self.reference_runs.append(reference)
        surrogate = np.array([self.controlled(run) for run in runs])
        lam = plan.coefficient
        value = lam * surrogate.mean(axis=0) + self.controlled(reference) - lam * surrogate[0]
        bound = self.control.bound
        spread = bound + lam * (1.0 - 1.0 / plan.size) * (1.0 + 2.0 * bound)
        self.add_block(value, -spread, 1.0 + spread, weight)

    def reference_block(self, weight: np.ndarray) -> None:
        substrate = self.fresh_substrate()
        reference = self.execute(self.reference, substrate)
        if self.needs_named_pair():
            self.run_pair(self.certified, substrate, reference)
        self.reference_runs.append(reference)
        self.add_reference_block(reference, weight)

    def run(self) -> Report:
        s = self.settings
        try:
            self.explore()
            rechecked = False
            while len(self.blocks) < s.max_blocks and not self.all_decided():
                self.refit()
                rechecked = self.replan(rechecked)
                weight = self.epoch_weight()
                for _ in range(s.epoch_blocks):
                    if self.plan.surrogate is not None:
                        self.surrogate_block(weight)
                    else:
                        self.reference_block(weight)
                    if self.all_decided() or len(self.blocks) >= s.max_blocks:
                        break
        except BudgetExhausted:
            pass
        return self.report()

    def estimate(self) -> np.ndarray:
        if not self.blocks:
            return np.full(len(STATISTICS), np.nan)
        later = [block for block in self.blocks if block.after_commit]
        if not later:
            return np.mean([block.value for block in self.blocks], axis=0)
        values = np.array([block.value for block in later])
        weights = np.array([block.weight for block in later])
        return (values * weights).sum(axis=0) / weights.sum(axis=0)

    def report(self) -> Report:
        estimate = self.estimate()
        decisions = self.decisions()
        statistics = tuple(
            StatisticReport(
                name=name,
                estimate=float(estimate[k]),
                reference_interval=self.reference_sequences[k].interval,
                bias_interval=self.bias_interval(k),
                tier_interval=self.tier_interval(k),
                channel_interval=self.channel_interval(k),
                decision=decisions[k],
            )
            for k, name in enumerate(STATISTICS)
        )
        return Report(
            statistics=statistics,
            committed=self.committed.name if self.committed is not None else None,
            block_size=self.plan.size,
            certified=self.certified.name if self.certified is not None else None,
            spent=self.spent,
            blocks=len(self.blocks),
            reference_runs=len(self.reference_runs),
            tabulated=self.tabulated,
        )


def run_query(
    graph: SocialGraph,
    query: Query,
    models: ModelPair,
    settings: Settings,
    candidates: Sequence[Configuration] | None = None,
    certify: Configuration | str = "all-cheap",
    routing_fractions: Sequence[float] = (),
) -> Report:
    return BudgetedSimulation(graph, query, models, settings, candidates, certify, routing_fractions).run()
