# BACS: Budget-Adaptive Community Simulation with LLM Agents

Simulates how a message spreads through a social graph where every "do I share this?" decision is made by an LLM. Given a fixed budget, BACS reports three cascade statistics — reach, community reach, depth — each with a confidence interval that stays valid no matter when you stop the run.

The problem it solves: the frontier model gives the answer you want but is expensive, and a cheap model may be biased by an unknown amount. BACS runs every configuration on the same random draws, uses the simulator's own known flip probabilities as a zero-call control variate, and only switches to a cheap surrogate if that query's pilot runs say it pays off. It also certifies whether the surrogate's bias stays inside a tolerance you pick.

## Install

Python 3.10+ (3.12 recommended).

```sh
conda create -n bacs python=3.12 pip -y
conda activate bacs

git clone git@github.com:phkhanhtrinh23/BACS-LLM.git
cd BACS-LLM
python -m pip install -r requirements.txt
```

Then copy [.env.example](.env.example) to `.env` and fill it in:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
BACS_CHEAP_MODEL=gpt-4o-mini
BACS_FRONTIER_MODEL=gpt-4.1
```

`OPENAI_BASE_URL` works with any OpenAI-compatible provider.

## Usage

Replay a study from logged answers. Budget is in units of one frontier call:

```sh
python -m bacs \
  --edges edges.csv --communities communities.txt --tables answers.npz \
  --seeds 12 40 77 --attention 0.3 \
  --budget 400 --cheap-cost 0.075 --frontier-cost 1.0
```

You get back a JSON report: which surrogate was committed and at what block size, which one was certified, what was spent, and per statistic the estimate, the frontier interval, the bias interval and the bias decision (preserved / not preserved / abstain).

Price several surrogates and certify the one that gets committed:

```sh
python -m bacs ... --candidates all-cheap hub-frontier confidence-routing \
  --routing-fractions 0.1 0.25 0.5 --certify committed
```

Routing fractions are expressed as a share of the total cost of escalating every agent.

Certify a headline-only surrogate, splitting model bias from format bias:

```sh
python -m bacs ... --channel compressed --split-bias
```

Other flags:

- `--tabulate` — buy each frontier answer once and reuse it. Only valid when prompts depend solely on agent, message, format and sharer count.
- `--asymptotic` — asymptotic intervals instead of betting intervals.
- `--no-control` — turn off the zero-call control.
- `--attention-only` — control on exposure flips only, not decision flips.

Tolerances, error level, pilot sizes and epoch length live in [`Settings`](bacs/estimator.py).

Two caveats worth knowing: a block's cost is only known once it has run, so a study can overshoot the budget by up to one run (the report records actual spend), and the cheap calls used to measure disagreement for routing are charged without a budget check.

## Input format

- **Edges** — CSV, one `source,target` follower pair per line, zero-based agent indices. `--undirected` adds the reverse of every edge.
- **Communities** — text file, one community label per line, in agent order.
- **Answer tables** — `.npz` with `cheap_full` and `frontier_full`, optionally `cheap_compressed` and `frontier_compressed` for the headline channel. One row per agent, two columns: one sharer noticed, several sharers noticed.

## Data

[data/README.md](data/README.md) covers preparing a study. Sources used:

| Source | What it gives you |
|---|---|
| [Bluesky posts](https://huggingface.co/datasets/alpindale/two-million-bluesky-posts) | Reply network, personas, messages |
| [Reed98 Facebook100](https://networkrepository.com/socfb-Reed98.php) | University friendship graph |
| [Digg 2009](https://www.isi.edu/~lerman/downloads/digg2009.html) | Social-news follower graph |

## Layout

```text
bacs/
├── graph.py          social graph, communities, personas
├── substrate.py      shared random draws for a run
├── models.py         prompts, answer tables, chat API wrapper
├── simulator.py      cascade simulation + trial log
├── configuration.py  reference and surrogate configurations
├── control.py        value tables, zero-call control
├── pricing.py        block variance, surrogate price
├── certificate.py    betting and asymptotic confidence sequences
├── estimator.py      the budgeted per-query procedure and its report
└── cli.py            entry point (python -m bacs)
```

## License

MIT, see [LICENSE](LICENSE). Datasets and model APIs keep their own terms.
