[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/memscore.svg)](https://github.com/habib-analyst/memscore/stargazers)

# memscore — portable memory with trust-but-verify transfer

**Your agent's memories are assets. They should travel with their track record — and no agent should trust another agent's track record blindly.**

`memscore` scores every memory by *measured usefulness* (bandit-style: memories that help successful tasks get upweighted, the useless get pruned), and makes memories **portable across agents**: export a memory pack, import it elsewhere — but on import, a probe set measures whether each imported memory *actually helps the new agent* and recalibrates its score. Trust, but verify.

## Quickstart

```bash
pip install memscore
```

```python
from memscore import Agent, TaskEnvironment, export_pack, recalibrated_import

# Donor agent learns the medical domain
env = TaskEnvironment(seed=0)
donor = Agent("donor", {"medical": 0.8, "geo": 0.2})
for _ in range(200):
    donor.step(env.sample(donor.distribution))
print("donor accuracy:", round(donor.accuracy(), 3))   # ~0.78

pack = export_pack(donor.store, donor.agent_id)

# New agent imports -- but verifies first
newbie = Agent("newbie", {"medical": 0.8, "geo": 0.2})
probes = newbie.probe_tasks(env, 40)
report = recalibrated_import(newbie.store, pack, probes)
print(f"kept {report.n_kept}, dropped {report.n_dropped}")
```

Expected output:
```
donor accuracy: 0.775
kept 45, dropped 0
```

## Measured results

3 agents, 5 seeds, synthetic QA environment (math/geo/medical), CPU-only.
Donor A is math-heavy; importer B is medical-heavy.

| Importer B (medical-heavy) | success rate (mean, 5 seeds) |
|---|---|
| from scratch | 0.651 |
| naive import (trust donor scores) | 0.831 |
| **recalibrated import (memscore)** | **0.795** |

| Poisoned donor (25% of on-domain entries corrupted @ 0.95 utility) | |
|---|---|
| naive import | 0.725 |
| **recalibrated import** | **0.788** (+6.3pp) |

- **Poison recall: 0.90** — verification catches 90% of corrupted entries.
- Honest cost: in benign settings verification is conservative (−3.6pp vs
  naive on B) — the insurance premium. It pays out under adversity.
- 12/12 tests green.

Full numbers: [`paper/PAPER.md`](paper/PAPER.md), [`results/summary.json`](results/summary.json).

> Honest note: this ships with a synthetic evaluation environment. The mechanism (utility scoring, probe recalibration) is domain-agnostic, but the numbers above are from the simulator, not production agents.

## How it works

```
memory entry = { content, embedding, utility, provenance }
                                        |
retrieval:  score = 0.6 * similarity + 0.4 * utility
                                        |
import:     pack --> STAGING --> probe tasks --> per-entry measured
            contribution --> recalibrated utility --> keep/drop --> live store
```

- **Utility learning**: bandit update — success pulls utility toward 1, failure toward 0. Pruning removes entries proven useless (never the untested).
- **Recalibration**: `recalibrated = 0.4 * donor_utility * domain_support + 0.6 * measured`. An entry whose domain never appears in the importer's probes gets its donor score discounted to ~0 — no evidence of relevance, no trust.
- **Why not naive import?** Retrieval blends similarity *and* utility, so a donor's high-utility-but-irrelevant memory can outrank the right one. Recalibration fixes the ranking.

## Citation

```bibtex
@software{rehman2026memscore,
  title  = {memscore: portable cross-agent utility memory with trust-but-verify transfer},
  author = {Habib Ur Rehman},
  year   = {2026},
  url    = {https://github.com/habib-analyst/memscore}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
