"""Deterministic, dependency-free text embeddings for memscore.

Uses hashed bag-of-words with seeded random projection vectors.
Deterministic across runs (fixed seed); no model download, no network.
Similarity between same-domain texts is higher than cross-domain, which is
all the memory experiments need.
"""

import hashlib
import math
import re
from typing import List

import numpy as np

_DIM = 128
_SEED = 20261012


def _token_vector(token: str) -> np.ndarray:
    h = hashlib.sha256(f"{_SEED}:{token}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.standard_normal(_DIM)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def embed(text: str) -> np.ndarray:
    toks = tokenize(text)
    if not toks:
        return np.zeros(_DIM, dtype=np.float64)
    vecs = [_token_vector(t) for t in toks]
    v = np.sum(vecs, axis=0)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
