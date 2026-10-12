"""Utility-scored memory entries and the memory store.

A MemoryEntry carries its utility score WITH it (portable across agents),
plus provenance: which agent created it and which task-type tags it serves.

Retrieval blends content similarity with utility:
    score = alpha * cosine_similarity + (1 - alpha) * utility
so a donor agent's high-utility-but-irrelevant memory can outrank a
relevant one -- the exact failure mode naive import suffers from, and
the reason recalibration exists (see portable.py).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embeddings import cosine, embed


@dataclass
class MemoryEntry:
    content: str
    answer: str
    task_types: List[str]
    agent_id: str
    utility: float = 0.5
    embedding: np.ndarray = field(default=None, repr=False)  # type: ignore
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    retrieves: int = 0
    successes: int = 0

    def __post_init__(self) -> None:
        if self.embedding is None:
            self.embedding = embed(self.content + " " + " ".join(self.task_types))
        self.utility = float(min(1.0, max(0.0, self.utility)))

    def to_dict(self) -> Dict:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "answer": self.answer,
            "task_types": list(self.task_types),
            "agent_id": self.agent_id,
            "utility": self.utility,
            "embedding": self.embedding.tolist(),
            "created_at": self.created_at,
            "retrieves": self.retrieves,
            "successes": self.successes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MemoryEntry":
        e = cls(
            content=d["content"],
            answer=d["answer"],
            task_types=list(d.get("task_types", [])),
            agent_id=d.get("agent_id", "unknown"),
            utility=float(d.get("utility", 0.5)),
            entry_id=d.get("entry_id", uuid.uuid4().hex[:12]),
            created_at=float(d.get("created_at", time.time())),
        )
        if "embedding" in d and d["embedding"] is not None:
            e.embedding = np.asarray(d["embedding"], dtype=np.float64)
        e.retrieves = int(d.get("retrieves", 0))
        e.successes = int(d.get("successes", 0))
        return e


class MemoryStore:
    """Utility-scored store with bandit-style learning and utility pruning."""

    def __init__(self, alpha: float = 0.6, sim_threshold: float = 0.15):
        self.alpha = alpha  # weight on similarity vs utility in retrieval
        self.sim_threshold = sim_threshold
        self.entries: Dict[str, MemoryEntry] = {}

    def add(self, entry: MemoryEntry) -> str:
        self.entries[entry.entry_id] = entry
        return entry.entry_id

    def __len__(self) -> int:
        return len(self.entries)

    def retrieve(self, query: str, k: int = 1) -> List[Tuple[MemoryEntry, float]]:
        if not self.entries:
            return []
        q = embed(query)
        scored = []
        for e in self.entries.values():
            sim = cosine(q, e.embedding)
            if sim < self.sim_threshold:
                continue
            score = self.alpha * sim + (1.0 - self.alpha) * e.utility
            scored.append((e, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        for e, _ in scored[:k]:
            e.retrieves += 1
        return scored[:k]

    # ---- bandit-style utility learning ----
    def reward(self, entry_id: str, success: bool, lr: float = 0.15) -> None:
        e = self.entries.get(entry_id)
        if e is None:
            return
        if success:
            e.successes += 1
            e.utility = e.utility + lr * (1.0 - e.utility)
        else:
            e.utility = e.utility - lr * e.utility

    def prune(self, min_utility: float = 0.15, min_retrieves: int = 3) -> int:
        """Drop entries proven useless by experience. Never prune the
        untested (retrieves < min_retrieves)."""
        doomed = [
            eid
            for eid, e in self.entries.items()
            if e.retrieves >= min_retrieves and e.utility < min_utility
        ]
        for eid in doomed:
            del self.entries[eid]
        return len(doomed)

    def mean_utility(self) -> float:
        if not self.entries:
            return 0.0
        return float(np.mean([e.utility for e in self.entries.values()]))
