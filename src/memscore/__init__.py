"""memscore — portable cross-agent utility memory.

Memories carry their utility scores WITH them across agents and sessions,
and importing agents recalibrate (trust-but-verify) instead of trusting
donor scores blindly.
"""

from .memory import MemoryEntry, MemoryStore
from .portable import (
    PACK_VERSION,
    RecalibrationReport,
    export_pack,
    naive_import,
    recalibrated_import,
)
from .agent import Agent, TaskEnvironment
from .embeddings import embed, cosine

__version__ = "0.1.0"
__all__ = [
    "MemoryEntry",
    "MemoryStore",
    "PACK_VERSION",
    "RecalibrationReport",
    "export_pack",
    "naive_import",
    "recalibrated_import",
    "Agent",
    "TaskEnvironment",
    "embed",
    "cosine",
]
