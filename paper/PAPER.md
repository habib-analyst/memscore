# memscore: Portable Cross-Agent Utility Memory with Trust-but-Verify Transfer

**Habib Ur Rehman** · 2026-10-12 · `github.com/habib-analyst/memscore`

## 1. The problem

Agent memories don't travel. When a new agent starts — or an existing agent
moves to a new task distribution — it rebuilds its memory from zero, even
when a sibling agent already learned the same facts. The obvious fix (copy
the memory store) has a trust problem: the donor's utility scores were earned
in *the donor's* world. A memory the donor found invaluable can be useless or
actively harmful to the importer, and blindly trusting donor scores lets a
buggy or malicious donor poison the importer with confident wrong answers.

## 2. Gap analysis

Utility/bandit-scored agent memory is covered prior art (U-MEM, Memory
Intelligence Agent, and the broader "memory with feedback" literature score
memories by usefulness *within one agent*). What nobody ships is **portable
utility**: memories that carry their scores *across* agents and sessions, plus
a **verification protocol** for the import — the importer re-measures each
entry's usefulness on its own probe tasks instead of trusting donor scores.
That combination is the invention here.

## 3. Method

**Memory entry.** `{content, embedding, utility, provenance}` where provenance
= `(agent_id, task_types, created_at, retrieves, successes)`. Utility is a
first-class portable attribute.

**Retrieval.** `score = 0.6 * cosine_similarity + 0.4 * utility`. Because
retrieval blends similarity *and* utility, a donor's high-utility-but-wrong
entry can outrank the right one — the failure mode naive import suffers.

**Utility learning (within an agent).** Bandit update: success pulls utility
toward 1, failure toward 0 (`lr=0.15`). Pruning drops entries proven useless;
untested entries are never pruned.

**Portability protocol (the invention).**
1. `export_pack` serializes the store; utilities travel with entries
   (`memscore-pack/1` JSON).
2. On import, entries go to **staging**, not the live store.
3. The importer runs a **probe set** sampled from its own task distribution,
   retrieving from staged entries and recording per-entry outcomes.
4. **Recalibration** (posterior, Laplace smoothing, prior strength C=3):

   `recal = (donor_utility * 3 + probe_success_rate * k) / (3 + k)`

   The donor prior is trusted unless probe evidence contradicts it.
5. **Conviction** (drop) only on *proven* bad entries: ≥2 **high-confidence**
   probe failures, where high-confidence means the probe query nearly exactly
   matched the entry (cosine > 0.8) yet the answer was wrong. Low-similarity
   failures are retrieval confusion, not entry guilt — never convicted on
   those. Thin probe records (k<2) never convict.

The design principle: **verification is evidence-driven, not
prejudice-driven.** No evidence → keep the donor prior. Counter-evidence →
downweight or drop.

## 4. Experiments

Deterministic synthetic QA environment (seeded; all numbers reproducible):
three domains (math / geo / medical), 120 questions each. Agent answers by
top-1 retrieval; environment reveals ground truth after each task (learning).
5 seeds.

- **Donor A**: math-heavy (70/20/10), 400 tasks → pack of ~155 entries.
- **Importer B**: medical-heavy (10/20/70). Conditions: from-scratch,
  naive import (trust donor scores), recalibrated import (120 probes).
- **Poisoned donor**: 25% of on-domain (medical) entries corrupted with wrong
  answers at utility 0.95 — the untrustworthy-donor scenario.
- **Importer C**: geo-heavy (15/70/15) — generality check, high-ambiguity
  domain (near-duplicate question pairs: "capital of X?" vs "country with
  capital Y?").

**Kill-gate** (pre-registered): recalibrated must beat naive under the
poisoned donor AND beat from-scratch benign — else the claim dies.

## 5. Results

| Importer B (medical-heavy), success rate | mean (5 seeds) |
|---|---|
| from scratch | 0.651 |
| naive import | 0.831 |
| **recalibrated import** | **0.795** |

| Poisoned donor (25% entries corrupted @ 0.95 utility) | mean |
|---|---|
| naive import | 0.725 |
| **recalibrated import** | **0.788** (+6.3pp) |

- **Poison recall: 0.90** — verification catches 90% of corrupted entries.
- Importer C (geo-heavy, benign): scratch 0.529 / naive 0.743 /
  **recalibrated 0.653**.
- 12/12 unit tests green; experiment CPU-only, ~10 s.

**Kill-gate: PASS.**

## 6. Honest limitations

1. **The insurance premium is real.** In benign settings recalibration costs
   accuracy vs naive import (−3.6pp on B, −9pp on C): verification
   conservatism discounts entries it cannot verify. Trust-but-verify is
   insurance — it costs a premium and pays out under adversity (+6.3pp
   poisoned). We report both, not just the win.
2. **High-ambiguity domains degrade verification.** On C (geo), probe
   attribution is noisy even with the high-confidence rule: near-duplicate
   questions make probe failures ambiguous, so verification neither convicts
   correctly nor helps ranking. Probe-based verification assumes probe
   outcomes are attributable — in confusable domains they aren't.
3. **Synthetic environment.** All numbers are from the simulator (hashed
   bag-of-words embeddings, synthetic QA). The mechanism is domain-agnostic,
   but production-agent validation is future work.
4. **Probe cost.** 120 probes per import is the verification overhead; the
   cost/accuracy tradeoff of probe budget is not yet characterized.
5. **Donor was mediocre.** Donor accuracy was only 0.60 (math
   near-duplicate confusion) — the pack it exported was imperfect, which is
   realistic but bounds the head-start import can provide.

## 7. What this proves

Cross-agent memory transfer needs a verification step, and the verification
can be lightweight and evidence-driven: trust the donor prior, convict only
on proven failures. It catches 90% of poisoned entries and turns a −10.6pp
poisoning loss (naive: 0.831→0.725) into a −0.7pp nick (recalibrated:
0.795→0.788). The premium (−3.6pp benign on B) is the honest price.
