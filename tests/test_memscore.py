import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memscore import (
    Agent,
    MemoryEntry,
    MemoryStore,
    TaskEnvironment,
    cosine,
    embed,
    export_pack,
    naive_import,
    recalibrated_import,
)


def test_embed_deterministic():
    a = embed("what is the capital of france?")
    b = embed("what is the capital of france?")
    assert np.allclose(a, b)
    assert abs(cosine(a, a) - 1.0) < 1e-9


def test_embed_domain_separation():
    m1 = embed("what does insulin treat diabetes")
    m2 = embed("what does metformin treat diabetes")
    g1 = embed("what is the capital of france paris")
    assert cosine(m1, m2) > cosine(m1, g1)


def test_entry_roundtrip():
    e = MemoryEntry(content="q?", answer="a", task_types=["medical"],
                    agent_id="a1", utility=0.8)
    e2 = MemoryEntry.from_dict(e.to_dict())
    assert e2.answer == "a" and abs(e2.utility - 0.8) < 1e-9
    assert np.allclose(e2.embedding, e.embedding)


def test_retrieve_prefers_similar_and_useful():
    s = MemoryStore(alpha=0.6)
    s.add(MemoryEntry(content="what is the capital of france?",
                      answer="paris", task_types=["geo"], agent_id="a",
                      utility=0.9))
    s.add(MemoryEntry(content="what is two plus two?",
                      answer="4", task_types=["math"], agent_id="a",
                      utility=0.9))
    hits = s.retrieve("capital of france?", k=1)
    assert hits and hits[0][0].answer == "paris"


def test_bandit_reward_moves_utility():
    s = MemoryStore()
    e = MemoryEntry(content="q", answer="a", task_types=["t"], agent_id="a",
                    utility=0.5)
    eid = s.add(e)
    s.reward(eid, True)
    assert s.entries[eid].utility > 0.5
    s.reward(eid, False)
    s.reward(eid, False)
    assert s.entries[eid].utility < 0.5


def test_prune_drops_proven_useless_keeps_untested():
    s = MemoryStore()
    bad = MemoryEntry(content="q1", answer="a", task_types=["t"],
                      agent_id="a", utility=0.05)
    new = MemoryEntry(content="q2", answer="a", task_types=["t"],
                      agent_id="a", utility=0.05)
    bid, nid = s.add(bad), s.add(new)
    for _ in range(5):
        s.retrieve("q1", k=2)
    assert s.prune(min_utility=0.15, min_retrieves=3) == 1
    assert bid not in s.entries and nid in s.entries


def test_pack_roundtrip_and_naive_import():
    s = MemoryStore()
    s.add(MemoryEntry(content="q", answer="a", task_types=["medical"],
                      agent_id="donor", utility=0.9))
    pack = export_pack(s, "donor")
    assert pack["pack_version"] == "memscore-pack/1"
    s2 = MemoryStore()
    assert naive_import(s2, pack) == 1
    assert abs(list(s2.entries.values())[0].utility - 0.9) < 1e-9


def test_recalibrated_import_trusts_prior_without_counterevidence():
    # Evidence-driven verification: unprobed entries keep their donor prior;
    # only probe failures convict.
    donor = MemoryStore()
    for i in range(6):  # math memories, high donor utility
        donor.add(MemoryEntry(content=f"what is {i} plus {i}?",
                              answer=str(2 * i), task_types=["math"],
                              agent_id="donor", utility=0.95))
    for i in range(6):  # medical memories, high donor utility
        donor.add(MemoryEntry(content="what does insulin treat?",
                              answer="diabetes", task_types=["medical"],
                              agent_id="donor", utility=0.95))
    pack = export_pack(donor, "donor")
    importer = MemoryStore()
    probes = [("what does insulin treat?", "diabetes", "medical")] * 30 + \
             [("what is 3 plus 3?", "6", "math")] * 5
    rep = recalibrated_import(importer, pack, probes)
    # useful medical memories verified by probes: kept with high utility
    med_utils = [importer.entries[eid].utility for eid in rep.kept_ids
                 if "medical" in importer.entries[eid].task_types]
    assert med_utils and min(med_utils) > 0.8
    # unprobed math memories: prior trusted (no counter-evidence)
    math_utils = [importer.entries[eid].utility for eid in rep.kept_ids
                  if "math" in importer.entries[eid].task_types]
    assert len(math_utils) >= 5  # tail preserved


def test_recalibrated_import_convicts_poison():
    # A confident-but-wrong donor entry fails probes -> dropped.
    donor = MemoryStore()
    donor.add(MemoryEntry(content="what does insulin treat?",
                          answer="hypertension",  # WRONG
                          task_types=["medical"], agent_id="donor",
                          utility=0.95))
    donor.add(MemoryEntry(content="what does metformin treat?",
                          answer="diabetes",  # right
                          task_types=["medical"], agent_id="donor",
                          utility=0.95))
    pack = export_pack(donor, "donor")
    importer = MemoryStore()
    probes = ([("what does insulin treat?", "diabetes", "medical")] * 20 +
              [("what does metformin treat?", "diabetes", "medical")] * 20)
    rep = recalibrated_import(importer, pack, probes)
    assert rep.n_dropped == 1 and rep.n_kept == 1
    kept = importer.entries[rep.kept_ids[0]]
    assert kept.answer == "diabetes"


def test_recalibrated_import_rejects_bad_pack():
    with pytest.raises(ValueError):
        recalibrated_import(MemoryStore(), {"nope": 1}, [("q", "a", "t")])


def test_recalibrated_import_rejects_empty_probes():
    s = MemoryStore()
    pack = export_pack(s, "x")
    with pytest.raises(ValueError):
        recalibrated_import(MemoryStore(), pack, [])


def test_agent_learns_from_feedback():
    env = TaskEnvironment(seed=11)
    a = Agent("a", {"medical": 1.0})
    acc_first = sum(a.step(env.sample(a.distribution)) for _ in range(30)) / 30
    acc_last = sum(a.step(env.sample(a.distribution)) for _ in range(30)) / 30
    assert acc_last > acc_first  # learning happens
