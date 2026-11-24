import json
import math

import numpy as np
import pytest

from bacs import (
    BettingSequence,
    BudgetedSimulation,
    Decision,
    ModelPair,
    Query,
    SampledTableModel,
    Settings,
    Substrate,
    TableModel,
    all_cheap,
    block_variance,
    decide,
    fit_control,
    fit_isotonic,
    margin,
    reference_configuration,
    run_query,
    share_probability,
    simulate,
)
from bacs.cli import main
from bacs.simulator import ATTENTION, DECISION


def test_share_probability_follows_temperature():
    assert share_probability(math.log(0.75), math.log(0.25)) == pytest.approx(0.75)
    sharper = share_probability(math.log(0.75), math.log(0.25), temperature=0.5)
    assert sharper == pytest.approx(0.9)


def test_runs_are_deterministic_and_coupled(world):
    graph, models, query = world
    substrate = Substrate.draw(graph, 0, 4)
    first = simulate(graph, query, reference_configuration(), models, substrate)
    again = simulate(graph, query, reference_configuration(), models, substrate)
    assert np.array_equal(first.statistics, again.statistics)
    twin = ModelPair(TableModel(models.frontier.probabilities, 0.0), models.frontier)
    cheap = simulate(graph, query, all_cheap(), twin, substrate)
    assert np.array_equal(first.statistics, cheap.statistics)
    assert np.all((first.statistics >= 0) & (first.statistics <= 1))


def test_each_uniform_is_read_at_most_once(world):
    graph, models, query = world
    for index in range(20):
        run = simulate(graph, query, reference_configuration(), models, Substrate.draw(graph, 1, index))
        trials = run.trials
        attention = trials.kind == ATTENTION
        edges = list(zip(trials.source[attention], trials.agent[attention]))
        assert len(edges) == len(set(edges))
        decisions = trials.agent[trials.kind == DECISION]
        assert np.bincount(decisions, minlength=graph.num_agents).max() <= 2


def test_control_has_mean_zero_and_respects_cap(world):
    graph, models, query = world
    reference = reference_configuration()
    runs = [simulate(graph, query, reference, models, Substrate.draw(graph, 3, i)) for i in range(400)]
    statistics = np.array([run.statistics for run in runs])
    cap = 3 * statistics[:10].std(axis=0, ddof=1)
    control = fit_control(graph, query, runs[:100], cap)
    values = np.array([control(graph, query, run) for run in runs[100:]])
    assert np.all(np.abs(values) <= cap + 1e-12)
    spread = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    assert np.all(np.abs(values.mean(axis=0)) <= 4 * spread + 1e-12)
    plain = statistics[100:].var(axis=0)
    controlled = (statistics[100:] - values).var(axis=0)
    assert controlled[0] < plain[0]


def test_block_variance_matches_simulation():
    rng = np.random.default_rng(0)
    surrogate = rng.normal(size=(20000, 1))
    reference = 0.8 * surrogate + 0.3 * rng.normal(size=(20000, 1))
    coefficient = np.array([0.8])
    size = 4
    predicted = block_variance(surrogate, reference, coefficient, size)[0]
    x = rng.normal(size=(5000, size))
    y = 0.8 * x[:, 0] + 0.3 * rng.normal(size=5000)
    blocks = 0.8 * x.mean(axis=1) + y - 0.8 * x[:, 0]
    assert predicted == pytest.approx(blocks.var(), rel=0.1)


def test_margin_matches_heuristic():
    assert margin(10) == pytest.approx(0.515, abs=0.005)
    assert margin(30) == pytest.approx(0.656, abs=0.005)


def test_betting_sequence_covers_and_shrinks():
    rng = np.random.default_rng(1)
    misses = 0
    for _ in range(40):
        sequence = BettingSequence(0.05, 0.0, 1.0)
        for value in rng.random(300) < 0.3:
            sequence.update(float(value), 0.0, 1.0)
            low, high = sequence.interval
            if not low <= 0.3 <= high:
                misses += 1
                break
    assert misses <= 6
    assert sequence.interval[1] - sequence.interval[0] < 0.3


def test_decisions_follow_tolerance():
    assert decide((-0.01, 0.01), 0.02) is Decision.PRESERVED
    assert decide((0.05, 0.08), 0.02) is Decision.NOT_PRESERVED
    assert decide((0.0, 0.05), 0.02) is Decision.ABSTAIN
    assert decide(None, 0.02) is Decision.ABSTAIN


def test_isotonic_map_is_monotone():
    rng = np.random.default_rng(3)
    cheap = rng.random(200)
    frontier = np.clip(cheap**2 + 0.1 * rng.normal(size=200), 0, 1)
    calibrator = fit_isotonic(cheap, frontier)
    grid = np.linspace(0, 1, 50)
    values = np.array([calibrator(p) for p in grid])
    assert np.all(np.diff(values) >= -1e-12)


def test_driver_respects_budget_and_reports(world):
    graph, models, query = world
    report = run_query(graph, query, models, Settings(budget=300.0, seed=5))
    assert report.spent <= 320.0
    assert len(report.statistics) == 3
    for statistic in report.statistics:
        low, high = statistic.reference_interval
        assert 0.0 <= low <= high <= 1.0
        assert statistic.decision is not None


def test_identical_surrogate_is_committed_and_preserved(world):
    graph, models, query = world
    twin = ModelPair(TableModel(models.frontier.probabilities, 0.02), models.frontier)
    query = Query("post", (0, 1, 2, 3, 4), 6, 0.8)
    simulation = BudgetedSimulation(graph, query, twin, Settings(budget=6000.0, seed=6))
    report = simulation.run()
    assert report.committed == all_cheap().name
    assert report.block_size > 1
    assert all(s.bias_interval[0] <= 0.0 <= s.bias_interval[1] for s in report.statistics)
    report = run_query(graph, query, twin, Settings(budget=1e6, seed=6), candidates=[], certify=all_cheap())
    assert all(s.decision is Decision.PRESERVED for s in report.statistics)
    assert report.spent < 1e6


def test_split_bias_and_sampled_answers_run(world):
    graph, models, query = world
    cheap = all_cheap("compressed")
    report = run_query(
        graph, query, models, Settings(budget=200.0, split_bias=True, seed=7), [cheap], certify=cheap
    )
    assert report.certified == cheap.name
    for statistic in report.statistics:
        tier, channel = statistic.tier_interval, statistic.channel_interval
        assert statistic.bias_interval == pytest.approx((tier[0] + channel[0], tier[1] + channel[1]))
    sampled = ModelPair(
        SampledTableModel(models.cheap.probabilities, 0.05),
        SampledTableModel(models.frontier.probabilities, 1.0),
    )
    report = run_query(graph, query, sampled, Settings(budget=200.0, seed=8))
    assert np.isfinite([s.estimate for s in report.statistics]).all()


def test_cli_reads_graph_and_tables(world, tmp_path, capsys):
    graph, models, _ = world
    edges = tmp_path / "edges.csv"
    communities = tmp_path / "communities.txt"
    tables = tmp_path / "answers.npz"
    np.savetxt(edges, np.column_stack([graph.sources, graph.targets]), fmt="%d", delimiter=",")
    np.savetxt(communities, graph.communities, fmt="%d")
    np.savez(
        tables,
        cheap_full=models.cheap.probabilities["full"],
        frontier_full=models.frontier.probabilities["full"],
        cheap_compressed=models.cheap.probabilities["compressed"],
        frontier_compressed=models.frontier.probabilities["compressed"],
    )
    main(
        [
            "--edges", str(edges),
            "--communities", str(communities),
            "--tables", str(tables),
            "--channel", "compressed",
            "--split-bias",
            "--budget", "200",
            "--seed", "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert all("tier_interval" in s and "channel_interval" in s for s in output["statistics"])
