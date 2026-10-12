"""Quickstart: portable memory transfer between two agents."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memscore import (
    Agent, TaskEnvironment, export_pack, naive_import, recalibrated_import,
)

env = TaskEnvironment(seed=0)

# 1. Donor learns the math domain
donor = Agent("donor", {"medical": 0.8, "geo": 0.2})
for _ in range(200):
    donor.step(env.sample(donor.distribution))
print(f"donor accuracy: {donor.accuracy():.3f} "
      f"({len(donor.store)} memories)")

pack = export_pack(donor.store, donor.agent_id)

# 2. New agent: naive import (trusts donor scores blindly)
env2 = TaskEnvironment(seed=1)
n1 = Agent("naive", {"medical": 0.8, "geo": 0.2})
naive_import(n1.store, pack)
for _ in range(60):
    n1.step(env2.sample(n1.distribution))
print(f"naive-import accuracy after 60 tasks: {n1.accuracy():.3f}")

# 3. New agent: recalibrated import (trust-but-verify)
env3 = TaskEnvironment(seed=1)
n2 = Agent("recal", {"medical": 0.8, "geo": 0.2})
report = recalibrated_import(n2.store, pack, n2.probe_tasks(env3, 40))
for _ in range(60):
    n2.step(env3.sample(n2.distribution))
print(f"recalibrated: kept {report.n_kept}, dropped {report.n_dropped}")
print(f"recalibrated accuracy after 60 tasks: {n2.accuracy():.3f}")
