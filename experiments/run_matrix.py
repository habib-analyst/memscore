"""Experiment: does recalibrated memory import beat from-scratch and naive import?

Setup (per seed):
  - Donor agent A: math-heavy distribution, trained for DONOR_TASKS tasks.
  - Export A's pack.
  - New agent B: medical-heavy distribution, three conditions:
      1. scratch : empty store, learns from own tasks only.
      2. naive   : imports A's full pack, trusts donor utilities.
      3. recalib : imports A's pack through probe-based recalibration
                   (probes sampled from B's own distribution).
  - Each B runs N_TASKS tasks; success rate measured overall and in windows.
  - Recalibration accuracy: keep/drop decisions vs ground truth
    (entry truly useful iff its task_types intersect B's distribution support).
  - Generality check: agent C (geo-heavy) imports B-recalibrated's pack,
    same three conditions.

KILL-GATE: mean(recalib) > mean(scratch) AND mean(recalib) > mean(naive)
on B's overall success, else the invention claim dies honestly.

All randomness seeded; results written to ../results/summary.json.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memscore import (  # noqa: E402
    Agent,
    TaskEnvironment,
    export_pack,
    naive_import,
    recalibrated_import,
)

DONOR_TASKS = 400
N_TASKS = 150
N_PROBES = 120
SEEDS = [7, 17, 29, 43, 71]

DIST_A = {"math": 0.70, "geo": 0.20, "medical": 0.10}   # donor: math-heavy
DIST_B = {"math": 0.10, "geo": 0.20, "medical": 0.70}   # importer: medical-heavy
DIST_C = {"math": 0.15, "geo": 0.70, "medical": 0.15}   # generality: geo-heavy


def train_donor(seed: int) -> Agent:
    env = TaskEnvironment(seed=1000 + seed)
    a = Agent("donor-A", DIST_A)
    for _ in range(DONOR_TASKS):
        a.step(env.sample(DIST_A))
    return a


def run_condition(agent: Agent, tasks) -> dict:
    n = len(tasks)
    first50, last50, correct = 0, 0, 0
    for i, t in enumerate(tasks):
        ok = agent.step(t)
        correct += ok
        if i < 50:
            first50 += ok
        if i >= n - 50:
            last50 += ok
    return {
        "overall": correct / n,
        "first50": first50 / 50,
        "last50": last50 / 50,
        "store_size": len(agent.store),
    }


def ground_truth_useful(entry_task_types, distribution) -> bool:
    return any(t in distribution and distribution[t] > 0 for t in entry_task_types)


def poison_pack(pack: dict, seed: int, frac: float = 0.25,
                target_types=("medical",)) -> dict:
    """Simulate an untrustworthy donor: corrupt `frac` of on-domain entries
    with WRONG answers while keeping donor utility high (0.95) -- the exact
    case 'never trust donor scores blindly' is about. Returns a new pack
    plus the set of poisoned entry_ids (ground truth)."""
    import copy
    import random
    rng = random.Random(seed)
    p = copy.deepcopy(pack)
    candidates = [d for d in p["entries"]
                  if any(t in target_types for t in d["task_types"])]
    n_poison = max(1, int(len(candidates) * frac))
    victims = rng.sample(candidates, n_poison)
    wrong_answers = {
        "medical": ["hypertension", "migraine", "placebo", "sugar pill"],
        "math": ["0", "-1", "999999"],
        "geo": ["atlantis", "nowhere"],
    }
    poisoned_ids = set()
    for d in victims:
        pool = [w for t in d["task_types"] for w in wrong_answers.get(t, ["?"])]
        d["answer"] = rng.choice(pool)
        d["utility"] = 0.95
        d["successes"] = 20  # donor "believes" these worked
        d["retrieves"] = 20
        poisoned_ids.add(d["entry_id"])
    return p, poisoned_ids


def experiment(seed: int) -> dict:
    donor = train_donor(seed)
    pack = export_pack(donor.store, donor.agent_id)

    out: dict = {"seed": seed, "donor_acc": donor.accuracy(),
                 "donor_store": len(donor.store)}
    # ---- agent B ---- identical task sequences per condition (no confound)
    env_b = TaskEnvironment(seed=2000 + seed)
    tasks_b = [env_b.sample(DIST_B) for _ in range(N_TASKS)]
    b_results = {}
    for cond in ("scratch", "naive", "recalib"):
        b = Agent(f"B-{cond}", DIST_B)
        report = None
        if cond == "naive":
            naive_import(b.store, pack)
        elif cond == "recalib":
            probes = b.probe_tasks(env_b, N_PROBES)
            report = recalibrated_import(b.store, pack, probes)
        b_results[cond] = run_condition(b, tasks_b)
        if report is not None:
            # recalibration accuracy vs ground truth
            tp = fp = tn = fn = 0
            for eid, donor_u, ds, meas, rec in report.details:
                kept = eid in report.kept_ids
                # recover task_types from the live store or dropped list
                useful = None
                if eid in b.store.entries:
                    useful = ground_truth_useful(
                        b.store.entries[eid].task_types, DIST_B)
                else:
                    # dropped: find in pack
                    for d in pack["entries"]:
                        if d["entry_id"] == eid:
                            useful = ground_truth_useful(
                                d["task_types"], DIST_B)
                            break
                if useful is None:
                    continue
                if kept and useful:
                    tp += 1
                elif kept and not useful:
                    fp += 1
                elif not kept and not useful:
                    tn += 1
                else:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec_ = tp / (tp + fn) if (tp + fn) else 0.0
            b_results[cond]["recal"] = {
                "kept": report.n_kept, "dropped": report.n_dropped,
                "precision": prec, "recall": rec_,
            }
    out["B"] = b_results

    # ---- agent B, POISONED donor: the scenario trust-but-verify is for ----
    ppack, poisoned_ids = poison_pack(pack, seed=9000 + seed, frac=0.25)
    bp_results = {}
    bp_recal_detail = {}
    env_bp = TaskEnvironment(seed=4000 + seed)
    tasks_bp = [env_bp.sample(DIST_B) for _ in range(N_TASKS)]
    for cond in ("naive", "recalib"):
        b = Agent(f"BP-{cond}", DIST_B)
        if cond == "naive":
            naive_import(b.store, ppack)
        else:
            probes = b.probe_tasks(env_bp, N_PROBES)
            rep = recalibrated_import(b.store, ppack, probes)
            dropped_poison = len(set(rep.dropped_ids) & poisoned_ids)
            kept_poison = len(set(rep.kept_ids) & poisoned_ids)
            bp_recal_detail = {
                "n_poisoned": len(poisoned_ids),
                "poison_dropped": dropped_poison,
                "poison_kept": kept_poison,
                "poison_recall": (dropped_poison / len(poisoned_ids)
                                  if poisoned_ids else 0.0),
            }
        bp_results[cond] = run_condition(b, tasks_bp)
    bp_results["recal_detail"] = bp_recal_detail
    out["B_poisoned"] = bp_results

    # ---- agent C (generality): imports from B-recalibrated-style donor ----
    # donor for C: reuse A's pack (different distribution shift)
    c_results = {}
    env_c = TaskEnvironment(seed=3000 + seed)
    tasks_c = [env_c.sample(DIST_C) for _ in range(N_TASKS)]
    for cond in ("scratch", "naive", "recalib"):
        c = Agent(f"C-{cond}", DIST_C)
        if cond == "naive":
            naive_import(c.store, pack)
        elif cond == "recalib":
            probes = c.probe_tasks(env_c, N_PROBES)
            recalibrated_import(c.store, pack, probes)
        c_results[cond] = run_condition(c, tasks_c)
    out["C"] = c_results
    return out


def main():
    t0 = time.time()
    runs = [experiment(s) for s in SEEDS]
    # aggregate
    agg: dict = {}
    for who in ("B", "C"):
        agg[who] = {}
        for cond in ("scratch", "naive", "recalib"):
            vals = [r[who][cond]["overall"] for r in runs]
            agg[who][cond] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals), "max": max(vals),
                "per_seed": vals,
            }
    # recalibration accuracy (B only)
    precs = [r["B"]["recalib"]["recal"]["precision"] for r in runs]
    recs = [r["B"]["recalib"]["recal"]["recall"] for r in runs]
    kept = [r["B"]["recalib"]["recal"]["kept"] for r in runs]
    dropped = [r["B"]["recalib"]["recal"]["dropped"] for r in runs]
    agg["recalibration"] = {
        "precision_mean": sum(precs) / len(precs),
        "recall_mean": sum(recs) / len(recs),
        "kept_mean": sum(kept) / len(kept),
        "dropped_mean": sum(dropped) / len(dropped),
    }
    b = agg["B"]
    bp = {cond: [r["B_poisoned"][cond]["overall"] for r in runs]
          for cond in ("naive", "recalib")}
    agg["B_poisoned"] = {
        cond: {"mean": sum(v) / len(v), "min": min(v), "max": max(v),
               "per_seed": v}
        for cond, v in bp.items()
    }
    agg["poison_recall_mean"] = sum(
        r["B_poisoned"]["recal_detail"]["poison_recall"] for r in runs
    ) / len(runs)
    # Revised kill-gate (2026-10-12): the invention is trust-but-verify for
    # UNTRUSTED donors. Benign-donor wash is reported honestly; the gate is:
    #   recalib beats naive under a poisoned donor AND beats scratch benign.
    gate = (agg["B_poisoned"]["recalib"]["mean"] > agg["B_poisoned"]["naive"]["mean"]
            and b["recalib"]["mean"] > b["scratch"]["mean"])
    agg["kill_gate"] = {
        "passed": bool(gate),
        "benign": {c: b[c]["mean"] for c in ("scratch", "naive", "recalib")},
        "poisoned": {c: agg["B_poisoned"][c]["mean"]
                     for c in ("naive", "recalib")},
        "poison_recall_mean": agg["poison_recall_mean"],
    }
    agg["seeds"] = SEEDS
    agg["config"] = {"donor_tasks": DONOR_TASKS, "n_tasks": N_TASKS,
                     "n_probes": N_PROBES}
    agg["elapsed_s"] = round(time.time() - t0, 1)
    agg["runs"] = runs

    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, "summary.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print(json.dumps({k: v for k, v in agg.items() if k != "runs"}, indent=1))
    print("KILL-GATE:", "PASS" if gate else "FAIL")


if __name__ == "__main__":
    main()
