# BACS: Budget-Adaptive Community Simulation with LLM Agents

BACS estimates how far a message spreads through an online community when every sharing decision is made by a language model. It spends a fixed budget on one query and reports three cascade statistics: reach, community reach and depth. Each statistic comes with an interval for what an all-frontier simulation would report, valid at whatever point the study stops.

The frontier model is expensive and each run is random, so a precise answer normally needs many frontier runs. A cheaper model saves money but can shift the answer by an unknown amount. BACS runs every configuration on the same random draws, removes luck from each run using the chances it already knows, and uses a cheaper surrogate only when the query's own early runs show that it pays. It also certifies whether the surrogate's bias stays within a tolerance you choose.

## Architecture

```text
Social graph, personas, message, seed accounts, budget
        │
        ▼
Shared random draws (one set per run, reused by every configuration)
        │
        ▼
Cascade simulator
        ├── Reference: frontier model, full post
        └── Surrogates: cheap model, routing rules, headline-only post
        │
        ▼
Trial log: every exposure and sharing decision with its known chance
        │
        ▼
Zero-call control (value tables over the graph + fitted weights)
        │
        ▼
Surrogate pricing (pilot runs → commit to a block plan or stay frontier-only)
        │
        ▼
Confidence sequences
        ├── Frontier value of each statistic
        └── Surrogate bias → preserved, not preserved or abstain
        │
        ▼
Report: estimate, intervals, bias decisions, spend
```

A run is fully determined by its configuration and its random draws. A cheap run therefore follows its frontier run step by step until the two models first disagree. Every exposure and decision is a coin flip with a chance the simulator already has, so the control subtracts each flip's surprise without making another model call.


## Installation

Use **Python 3.12** for the environment below. The code requires **Python 3.10 or newer**.

We encourage you to use conda or miniconda:

```sh
conda create -n bacs python=3.12 pip -y
conda activate bacs
```

Clone the repository and install the dependencies from its root:

```sh
git clone git@github.com:phkhanhtrinh23/BACS-LLM.git
cd BACS-LLM
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) lists NumPy, the OpenAI client for live agents.

### API keys


Create a local `.env` file using [.env.example](.env.example) as a reference:

```sh
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
BACS_CHEAP_MODEL=gpt-4o-mini
BACS_FRONTIER_MODEL=gpt-4.1
```

Replace the placeholders with your own key, endpoint and model IDs. `OPENAI_BASE_URL` can point to any compatible provider. `load_env()` reads the file without overriding variables already set in your shell. The `.env` file is git-ignored, so keep your key there and never in the code.

## Data Sources

See [data/README.md](data/README.md) for how to prepare a study from the sources below.

| Source | What it provides |
|---|---|
| [Bluesky posts](https://huggingface.co/datasets/alpindale/two-million-bluesky-posts) | Public posts for building a reply network, personas and messages |
| [Reed98 Facebook100 graph](https://networkrepository.com/socfb-Reed98.php) | A university friendship graph |
| [Digg social news data](https://www.isi.edu/~lerman/downloads/digg2009.html) | A social-news follower graph |

### Input format for logged answers

- **Edges:** a CSV file with one `source,target` follower pair per line, using agent indices from zero. Pass `--undirected` to add the reverse of every edge.
- **Communities:** a text file with one community label per agent, in agent order.
- **Answer tables:** an `.npz` file with `cheap_full` and `frontier_full`, plus optional `cheap_compressed` and `frontier_compressed` for the headline channel. Each table has one row per agent and two columns, for one sharer noticed and several sharers noticed.


## Running a Query

Replay a study from logged answers. The budget uses the same unit as the call costs, here one frontier call:

```sh
python -m bacs --edges edges.csv --communities communities.txt --tables answers.npz --seeds 12 40 77 --attention 0.3 --budget 400 --cheap-cost 0.075 --frontier-cost 1.0
```

The command prints a JSON report with these fields:

- the committed surrogate and its block size
- the certified surrogate
- the amount spent
- for each statistic: the estimate, the frontier interval, the bias interval and the bias decision

To price more surrogates and certify whichever one is committed:

```sh
python -m bacs --edges edges.csv --communities communities.txt --tables answers.npz --budget 400 --candidates all-cheap hub-frontier confidence-routing --routing-fractions 0.1 0.25 0.5 --certify committed
```

The routing fractions offer disagreement routing at 10, 25 and 50 percent of the total extra cost of escalating every agent.

To certify a headline-only surrogate and report its model bias and format bias separately:

```sh
python -m bacs --edges edges.csv --communities communities.txt --tables answers.npz --budget 400 --channel compressed --split-bias
```

Other flags:

- `--tabulate`: buy each frontier answer once and reuse it. This is valid when prompts depend only on the agent, message, format and sharer count.
- `--asymptotic`: use asymptotic intervals instead of the betting intervals.
- `--no-control`: switch off the zero-call control.
- `--attention-only`: keep exposure flips in the control and leave out decision flips.

Live agents run from Python:

```python
import os

from openai import OpenAI
from bacs import ChatLogprobModel, ModelPair, Query, Settings, SocialGraph, load_env, run_query

load_env()
graph = SocialGraph.from_edges(num_agents, edges, communities, personas)
client = OpenAI()
models = ModelPair(
    cheap=ChatLogprobModel(client, os.environ["BACS_CHEAP_MODEL"], input_price=0.15e-6, output_price=0.6e-6),
    frontier=ChatLogprobModel(client, os.environ["BACS_FRONTIER_MODEL"], input_price=2e-6, output_price=8e-6),
)
query = Query(post_text, seeds=(12, 40, 77), rounds=6, attention=0.3)
report = run_query(graph, query, models, Settings(budget=5.0))
print(report["reach"])
```

The prices above are placeholders. [`Settings`](bacs/estimator.py) also holds the tolerances, the error level, the pilot sizes and the epoch length.

A block's cost is only known after it runs, so a study can overspend the budget by up to one run. The report records the exact spend. When disagreement routing is priced, the cheap calls that measure disagreement are charged without a budget check.

## Project Structure

```text
BACS-LLM/
├── bacs/
│   ├── graph.py           social graph with communities and personas
│   ├── substrate.py       shared random draws for one run
│   ├── models.py          prompts, answer tables and the chat API wrapper
│   ├── simulator.py       cascade simulation with a full trial log
│   ├── configuration.py   reference and surrogate configurations
│   ├── control.py         value tables and the zero-call control
│   ├── pricing.py         block variance and the surrogate price
│   ├── certificate.py     betting and asymptotic confidence sequences
│   ├── estimator.py       the budgeted per-query procedure and its report
│   ├── cli.py             command line entry point
│   └── __main__.py        runs the command line with python -m bacs
├── data/                  guide to preparing study data
├── requirements.txt       Python dependencies
├── .env.example           template for API keys and model IDs
└── LICENSE                MIT license
```

## License

MIT. See [LICENSE](LICENSE). External datasets and model APIs keep their own terms.
