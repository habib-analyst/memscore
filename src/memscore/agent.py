"""Simulated agents and a synthetic QA task environment.

The environment is fully deterministic (seeded) so every reported number is
reproducible. Three task domains -- math, geo, medical -- each with a large
pool of (question, answer) pairs. An agent answers a query by retrieving its
top-1 memory; a retrieved memory whose answer matches is correct, otherwise
wrong; no retrieval ("unknown") is wrong. After each task the environment
reveals the ground truth, so the agent learns: it stores the correct memory
if it answered wrong, and applies bandit reward/penalty to the memory it used.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .memory import MemoryEntry, MemoryStore

Task = Tuple[str, str, str]  # (question, answer, task_type)


class TaskEnvironment:
    DOMAINS = ("math", "geo", "medical")

    def __init__(self, seed: int = 0, pool_per_domain: int = 120):
        self.rng = random.Random(seed)
        self.pools: Dict[str, List[Tuple[str, str]]] = {
            d: self._build_pool(d) for d in self.DOMAINS
        }

    def _build_pool(self, domain: str) -> List[Tuple[str, str]]:
        r = self.rng
        pool: List[Tuple[str, str]] = []
        if domain == "math":
            for _ in range(120):
                a, b = r.randint(2, 99), r.randint(2, 99)
                op = r.choice(["+", "-", "*"])
                q = f"what is {a} {op} {b}?"
                ans = str(eval(f"{a}{op}{b}"))
                pool.append((q, ans))
        elif domain == "geo":
            countries = [
                ("france", "paris"), ("japan", "tokyo"), ("brazil", "brasilia"),
                ("egypt", "cairo"), ("canada", "ottawa"), ("australia", "canberra"),
                ("india", "new delhi"), ("germany", "berlin"), ("italy", "rome"),
                ("spain", "madrid"), ("mexico", "mexico city"), ("turkey", "ankara"),
                ("norway", "oslo"), ("kenya", "nairobi"), ("thailand", "bangkok"),
                ("argentina", "buenos aires"), ("chile", "santiago"),
                ("poland", "warsaw"), ("greece", "athens"), ("portugal", "lisbon"),
            ]
            for _ in range(120):
                c, cap = r.choice(countries)
                fmt = r.choice([0, 1])
                if fmt == 0:
                    pool.append((f"what is the capital of {c}?", cap))
                else:
                    pool.append((f"which country has capital {cap}?", c))
        else:  # medical
            facts = [
                ("what does insulin treat", "diabetes"),
                ("what does metformin treat", "diabetes"),
                ("what does atorvastatin treat", "cholesterol"),
                ("what does lisinopril treat", "hypertension"),
                ("what does amoxicillin treat", "infection"),
                ("what does ibuprofen treat", "pain"),
                ("what does albuterol treat", "asthma"),
                ("what does omeprazole treat", "reflux"),
                ("what does levothyroxine treat", "hypothyroidism"),
                ("what does warfarin treat", "clots"),
                ("what does aspirin prevent", "clots"),
                ("what does furosemide treat", "edema"),
                ("what imaging shows bones best", "xray"),
                ("what imaging shows soft tissue best", "mri"),
                ("what measures blood sugar", "glucometer"),
                ("what measures blood pressure", "sphygmomanometer"),
                ("normal resting heart rate range", "60 to 100"),
                ("what vitamin comes from sunlight", "vitamin d"),
                ("what mineral carries oxygen in blood", "iron"),
                ("what organ filters blood", "kidney"),
            ]
            for _ in range(120):
                q, a = r.choice(facts)
                pool.append((q + "?", a))
        # shuffle deterministically
        r.shuffle(pool)
        return pool

    def sample(self, distribution: Dict[str, float]) -> Task:
        r = self.rng.random()
        cum = 0.0
        for d in self.DOMAINS:
            cum += distribution.get(d, 0.0)
            if r <= cum:
                q, a = self.rng.choice(self.pools[d])
                return (q, a, d)
        d = self.DOMAINS[-1]
        q, a = self.rng.choice(self.pools[d])
        return (q, a, d)


class Agent:
    """A memory-using agent in the simulated environment."""

    def __init__(self, agent_id: str, distribution: Dict[str, float], alpha: float = 0.6):
        self.agent_id = agent_id
        self.distribution = dict(distribution)
        self.store = MemoryStore(alpha=alpha)
        self.n_tasks = 0
        self.n_correct = 0

    def answer(self, query: str) -> Tuple[str, str | None]:
        hits = self.store.retrieve(query, k=1)
        if not hits:
            return "unknown", None
        e, _ = hits[0]
        return e.answer, e.entry_id

    def step(self, task: Task, lr: float = 0.15) -> bool:
        """Run one task with environment feedback. Returns correctness."""
        query, answer, ttype = task
        pred, used_id = self.answer(query)
        correct = pred == answer
        self.n_tasks += 1
        if correct:
            self.n_correct += 1
            if used_id:
                self.store.reward(used_id, True, lr=lr)
        else:
            if used_id:
                self.store.reward(used_id, False, lr=lr)
            # learn: store the ground truth for next time
            self.store.add(
                MemoryEntry(
                    content=query,
                    answer=answer,
                    task_types=[ttype],
                    agent_id=self.agent_id,
                    utility=0.5,
                )
            )
        return correct

    def accuracy(self) -> float:
        return self.n_correct / self.n_tasks if self.n_tasks else 0.0

    def probe_tasks(self, env: TaskEnvironment, n: int) -> List[Task]:
        return [env.sample(self.distribution) for _ in range(n)]
