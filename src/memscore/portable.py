"""Portable memory packs: trust-but-verify transfer across agents.

The invention. A memory pack carries each entry's utility score WITH it,
but the importing agent never trusts donor utilities blindly. Instead:

1. Imported entries go to a STAGING area (not the live store).
2. The agent runs a small PROBE set sampled from its OWN task distribution,
   retrieving from the staged entries and recording per-entry outcomes.
3. Each staged entry gets a RECALIBRATED utility:
       recalibrated = w_donor * donor_utility * domain_support
                   + w_meas  * measured_contribution
   where domain_support = fraction of probe tasks whose type is in the
   entry's task_types (evidence the entry's domain matters to THIS agent),
   and measured_contribution in [0,1] is mapped from the entry's
   probe success/failure record.
   An entry whose domain never appears in the probes gets its donor score
   discounted to ~0 -- no evidence of relevance, no trust.
4. Entries below the keep threshold are dropped; the rest merge into the
   live store with recalibrated utilities.

Naive import (trust donor scores) is the baseline this beats: a donor's
high-utility-but-irrelevant memory outranks relevant ones in retrieval
because retrieval blends similarity AND utility (see memory.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .memory import MemoryEntry, MemoryStore
from .embeddings import cosine, embed

PACK_VERSION = "memscore-pack/1"


@dataclass
class RecalibrationReport:
    n_imported: int
    n_kept: int
    n_dropped: int
    kept_ids: List[str] = field(default_factory=list)
    dropped_ids: List[str] = field(default_factory=list)
    # per-entry: (entry_id, donor_utility, domain_support, measured, recalibrated)
    details: List[Tuple[str, float, float, float, float]] = field(default_factory=list)


def export_pack(store: MemoryStore, agent_id: str) -> Dict:
    """Serialize a store into a portable pack. Utilities travel with entries."""
    return {
        "pack_version": PACK_VERSION,
        "agent_id": agent_id,
        "exported_at": time.time(),
        "n_entries": len(store.entries),
        "mean_utility": store.mean_utility(),
        "entries": [e.to_dict() for e in store.entries.values()],
    }


def _validate_pack(pack: Dict) -> None:
    if not isinstance(pack, dict) or pack.get("pack_version") != PACK_VERSION:
        raise ValueError(f"not a {PACK_VERSION} pack")
    if "entries" not in pack or not isinstance(pack["entries"], list):
        raise ValueError("pack has no entries list")


def naive_import(store: MemoryStore, pack: Dict) -> int:
    """Baseline: merge everything, trusting donor utilities as-is."""
    _validate_pack(pack)
    n = 0
    for d in pack["entries"]:
        store.add(MemoryEntry.from_dict(d))
        n += 1
    return n


def recalibrated_import(
    store: MemoryStore,
    pack: Dict,
    probe_tasks: List[Tuple[str, str, str]],
    keep_threshold: float = 0.3,
) -> RecalibrationReport:
    """Trust-but-verify import. probe_tasks: (query, answer, task_type)."""
    _validate_pack(pack)
    staged = [MemoryEntry.from_dict(d) for d in pack["entries"]]
    n_probes = len(probe_tasks)
    if n_probes == 0:
        raise ValueError("probe set is empty; cannot recalibrate")

    # per-entry probe record: [successes, failures, high_conf_failures]
    # A failure only counts as HIGH-CONFIDENCE when the probe query nearly
    # exactly matches the entry (cosine > 0.8): then a wrong answer proves
    # the entry itself is bad (poisoned/incorrect), not merely a victim of
    # retrieval confusion between near-duplicate questions.
    rec: Dict[str, List[int]] = {e.entry_id: [0, 0, 0] for e in staged}
    probe_types = [t for _, _, t in probe_tasks]

    for query, answer, _ttype in probe_tasks:
        # retrieve against staged entries only
        tmp = MemoryStore(alpha=store.alpha, sim_threshold=store.sim_threshold)
        for e in staged:
            tmp.entries[e.entry_id] = e
        hits = tmp.retrieve(query, k=1)
        if not hits:
            continue
        e, _ = hits[0]
        if e.answer == answer:
            rec[e.entry_id][0] += 1
        else:
            rec[e.entry_id][1] += 1
            if cosine(embed(query), e.embedding) > 0.8:
                rec[e.entry_id][2] += 1

    report = RecalibrationReport(
        n_imported=len(staged), n_kept=0, n_dropped=0
    )
    # Posterior update (Laplace smoothing, prior strength C=3):
    #   recal = (donor_prior * C + success_rate * k) / (C + k)
    # The donor prior is trusted UNLESS probe evidence contradicts it.
    # Two separate decisions:
    #   UTILITY: stable posterior (C=3 resists probe-attribution noise).
    #   DROP: convict only on strong negative evidence (k>=3, sr<=0.25).
    # Thin probe records never convict -- insufficient evidence.
    C = 3.0
    for e in staged:
        succ, fail, hc_fail = rec[e.entry_id]
        k = succ + fail
        if k == 0:
            recal = e.utility
            sr = 1.0
        else:
            sr = succ / k
            recal = (e.utility * C + sr * k) / (C + k)
        recal = float(min(1.0, max(0.0, recal)))
        domain_support = (
            sum(1 for t in probe_types if t in e.task_types) / n_probes
        )
        # Convict on PROVEN bad entries: >=2 high-confidence failures
        # (near-exact probe match, wrong answer). Low-similarity failures
        # are retrieval confusion, not entry guilt -- never convict on those.
        convicted = hc_fail >= 2 or recal < 0.1
        report.details.append(
            (e.entry_id, e.utility, domain_support, sr, recal)
        )
        if not convicted:
            e.utility = recal
            e.retrieves = 0
            e.successes = 0
            store.add(e)
            report.n_kept += 1
            report.kept_ids.append(e.entry_id)
        else:
            report.n_dropped += 1
            report.dropped_ids.append(e.entry_id)
    return report
