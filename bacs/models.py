from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np


def load_env(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Prompt:
    agent: int
    persona: str
    message: str
    channel: str
    sharer_count: int
    seed: int = field(default=0, compare=False)


@dataclass(frozen=True)
class Answer:
    probability: float | None
    cost: float
    shared: bool | None = None


class DecisionModel(Protocol):
    def answer(self, prompt: Prompt) -> Answer: ...


@dataclass(frozen=True)
class ModelPair:
    cheap: DecisionModel
    frontier: DecisionModel


def share_probability(logp_yes: float, logp_no: float, temperature: float = 1.0) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logp_yes == -math.inf and logp_no == -math.inf:
        raise ValueError("the answer carries no Yes or No mass")
    gap = (logp_no - logp_yes) / temperature
    if gap > 700:
        return 0.0
    return 1.0 / (1.0 + math.exp(gap))


class TableModel:
    def __init__(self, probabilities: Mapping[str, np.ndarray], cost_per_call: float | Mapping[str, float]):
        self.probabilities = {name: np.asarray(table, dtype=float) for name, table in probabilities.items()}
        for table in self.probabilities.values():
            if table.ndim != 2 or table.shape[1] != 2:
                raise ValueError("each table needs one row per agent and one column per sharer count")
        self.cost_per_call = cost_per_call

    def answer(self, prompt: Prompt) -> Answer:
        probability = self.probabilities[prompt.channel][prompt.agent, prompt.sharer_count - 1]
        cost = self.cost_per_call
        if not isinstance(cost, (int, float)):
            cost = cost[prompt.channel]
        return Answer(float(probability), float(cost))


class SampledTableModel(TableModel):
    def answer(self, prompt: Prompt) -> Answer:
        answer = super().answer(prompt)
        rng = np.random.default_rng([prompt.seed, prompt.agent, prompt.sharer_count])
        return Answer(None, answer.cost, bool(rng.random() < answer.probability))


class MemoizedModel:
    def __init__(self, model: DecisionModel):
        self.model = model
        self.answers: dict[Prompt, Answer] = {}

    def answer(self, prompt: Prompt) -> Answer:
        if prompt in self.answers:
            stored = self.answers[prompt]
            return Answer(stored.probability, 0.0, stored.shared)
        answer = self.model.answer(prompt)
        if answer.probability is not None:
            self.answers[prompt] = answer
        return answer


class ChatLogprobModel:
    instruction = (
        "You are a user of a social network. Decide whether you would reshare the post below. "
        "Answer with exactly one word: Yes or No."
    )

    def __init__(
        self,
        client: Any,
        model: str,
        input_price: float,
        output_price: float,
        temperature: float = 1.0,
        top_logprobs: int = 20,
    ):
        self.client = client
        self.model = model
        self.input_price = input_price
        self.output_price = output_price
        self.temperature = temperature
        self.top_logprobs = top_logprobs

    def messages(self, prompt: Prompt) -> list[dict[str, str]]:
        system = self.instruction
        if prompt.persona:
            system = f"{system}\n\nYour recent posts:\n{prompt.persona}"
        seen = "one account you follow" if prompt.sharer_count == 1 else "several accounts you follow"
        user = f"Post:\n{prompt.message}\n\nYou noticed it shared by {seen}. Would you reshare it?"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def answer(self, prompt: Prompt) -> Answer:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages(prompt),
            max_tokens=1,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        mass = {"yes": 0.0, "no": 0.0}
        for candidate in response.choices[0].logprobs.content[0].top_logprobs:
            word = candidate.token.strip().lower()
            if word in mass:
                mass[word] += math.exp(candidate.logprob)
        logp_yes = math.log(mass["yes"]) if mass["yes"] > 0 else -math.inf
        logp_no = math.log(mass["no"]) if mass["no"] > 0 else -math.inf
        probability = share_probability(logp_yes, logp_no, self.temperature)
        usage = response.usage
        cost = usage.prompt_tokens * self.input_price + usage.completion_tokens * self.output_price
        return Answer(probability, cost)
