from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .configuration import all_cheap, confidence_routing, hub_frontier
from .estimator import Settings, run_query
from .graph import SocialGraph
from .models import ModelPair, TableModel
from .simulator import COMPRESSED, FULL, Query


def load_graph(edges: Path, communities: Path, directed: bool) -> SocialGraph:
    pairs = np.loadtxt(edges, dtype=np.int64, delimiter=",", ndmin=2)
    labels = np.loadtxt(communities, dtype=str, ndmin=1)
    return SocialGraph.from_edges(labels.size, pairs, labels, directed=directed)


def load_models(tables: Path, cheap_cost: float, frontier_cost: float) -> ModelPair:
    data = np.load(tables)
    cheap = {FULL: data["cheap_full"]}
    frontier = {FULL: data["frontier_full"]}
    if "cheap_compressed" in data and "frontier_compressed" in data:
        cheap[COMPRESSED] = data["cheap_compressed"]
        frontier[COMPRESSED] = data["frontier_compressed"]
    return ModelPair(TableModel(cheap, cheap_cost), TableModel(frontier, frontier_cost))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m bacs", description="Budgeted, certified estimation of cascade statistics.")
    data = parser.add_argument_group("data")
    data.add_argument("--edges", type=Path, required=True, help="CSV file with one follower edge per line: source,target")
    data.add_argument("--communities", type=Path, required=True, help="text file with one community label per agent")
    data.add_argument("--tables", type=Path, required=True, help="npz file with cheap_full and frontier_full answer tables")
    data.add_argument("--undirected", action="store_true", help="add the reverse of every edge")
    query = parser.add_argument_group("query")
    query.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    query.add_argument("--rounds", type=int, default=6)
    query.add_argument("--attention", type=float, default=0.3)
    query.add_argument("--channel", choices=[FULL, COMPRESSED], default=FULL)
    study = parser.add_argument_group("study")
    study.add_argument("--budget", type=float, required=True, help="total spend, in the same unit as the call costs")
    study.add_argument("--cheap-cost", type=float, default=0.05)
    study.add_argument("--frontier-cost", type=float, default=1.0)
    study.add_argument("--tolerances", type=float, nargs=3, default=[0.02, 0.05, 0.10])
    study.add_argument("--level", type=float, default=0.05)
    study.add_argument(
        "--candidates",
        nargs="+",
        choices=["all-cheap", "hub-frontier", "confidence-routing"],
        default=["all-cheap"],
    )
    study.add_argument("--routing-fractions", type=float, nargs="*", default=[])
    study.add_argument("--certify", choices=["all-cheap", "committed", "none"], default="all-cheap")
    study.add_argument("--no-control", action="store_true")
    study.add_argument("--attention-only", action="store_true", help="drop decision trials from the control")
    study.add_argument("--split-bias", action="store_true", help="certify tier and channel bias separately")
    study.add_argument("--asymptotic", action="store_true", help="use asymptotic instead of betting sequences")
    study.add_argument("--tabulate", action="store_true", help="allow caching all reference answers when cheaper")
    study.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    graph = load_graph(args.edges, args.communities, directed=not args.undirected)
    models = load_models(args.tables, args.cheap_cost, args.frontier_cost)
    headline = "compressed" if args.channel == COMPRESSED else None
    query = Query("message", tuple(args.seeds), args.rounds, args.attention, headline=headline)
    builders = {
        "all-cheap": lambda: all_cheap(args.channel),
        "hub-frontier": lambda: hub_frontier(graph, channel=args.channel),
        "confidence-routing": lambda: confidence_routing(channel=args.channel),
    }
    settings = Settings(
        budget=args.budget,
        tolerances=tuple(args.tolerances),
        level=args.level,
        use_control=not args.no_control,
        decision_trials=not args.attention_only,
        split_bias=args.split_bias,
        asymptotic=args.asymptotic,
        tabulate=args.tabulate,
        seed=args.seed,
    )
    candidates = [builders[name]() for name in args.candidates]
    report = run_query(graph, query, models, settings, candidates, args.certify, args.routing_fractions)
    output = {
        "committed": report.committed,
        "block_size": report.block_size,
        "certified": report.certified,
        "spent": report.spent,
        "blocks": report.blocks,
        "reference_runs": report.reference_runs,
        "tabulated": report.tabulated,
        "statistics": [
            {
                "name": s.name,
                "estimate": s.estimate,
                "reference_interval": s.reference_interval,
                "bias_interval": s.bias_interval,
                "tier_interval": s.tier_interval,
                "channel_interval": s.channel_interval,
                "decision": s.decision.value if s.decision is not None else None,
            }
            for s in report.statistics
        ],
    }
    print(json.dumps(output, indent=2))
