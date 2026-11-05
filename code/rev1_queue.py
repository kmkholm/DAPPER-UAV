"""Revision 1 experiment grid (fixed; resumable: a run whose result json exists is skipped).

Operational-state task (op3), both fleets, rounds 20 x steps 50 as in the original grid. In priority order:
  cv-dapper   : DAPPER, sortie-grouped 5-fold CV, transition windows scored            10 runs
  companion   : DAPPER on traffic without Remote-ID + log upload (nocomp), and also
                with a humanised command cadence (nocomp_human), 5-fold CV              20 runs
  lodo        : DAPPER, leave-one-drone-out (drone 1, 2, 3 of every site)                6 runs
  lomt        : DAPPER, leave-one-mission-type-out (A: delivery/survey/inspection;
                B: patrol/survey/inspection)                                             6 runs
  onboarding  : DAPPER with held-out site A3 / B3, prompts freshly initialised at the
                new site (corrected Algorithm 1), seeds 1-2, original split              4 runs
  cv-methods  : local, central, fedavg, fedprox, fedbn under the same 5-fold CV         50 runs
                                                                                total   96 runs
  python rev1_queue.py                 (one GPU; ~2-3 min per run on an RTX A4000, faster on a 4090)
  python rev1_queue.py --only cv-dapper,companion
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "rev1", "runs")
PY = sys.executable
ROUNDS, STEPS = 20, 50
NETS = ("A", "B")
MTYPES = {"A": ["delivery", "survey", "inspection"], "B": ["patrol", "survey", "inspection"]}
HOLD = {"A": "A3", "B": "B3"}


def jobs():
    for net in NETS:
        for f in range(5):
            yield "cv-dapper", dict(net=net, method="dapper", seed=1, split="cv5", fold=f)
    for d in ("nocomp", "nocomp_human"):
        for net in NETS:
            for f in range(5):
                yield "companion", dict(net=net, method="dapper", seed=1, split="cv5", fold=f, defense=d)
    for net in NETS:
        for f in (1, 2, 3):
            yield "lodo", dict(net=net, method="dapper", seed=1, split="lodo", fold=f)
    for net in NETS:
        for t in MTYPES[net]:
            yield "lomt", dict(net=net, method="dapper", seed=1, split="lomt", mtype=t)
    for net in NETS:
        for s in (1, 2):
            yield "onboarding", dict(net=net, method="dapper", seed=s, split="fixed", holdout=HOLD[net])
    for m in ("fedbn", "fedavg", "central", "local", "fedprox"):
        for net in NETS:
            for f in range(5):
                yield "cv-methods", dict(net=net, method=m, seed=1, split="cv5", fold=f)


def name(j):
    tag = ("_def" + j["defense"]) if j.get("defense") else ""
    if j["split"] != "fixed":
        tag += "_" + j["split"] + (j["mtype"] if j["split"] == "lomt" else str(j["fold"]))
    return "%st_%s_s%d%s%s" % (j["net"], j["method"], j["seed"], ("_ho" + j["holdout"]) if j.get("holdout") else "", tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated groups")
    ap.add_argument("--dry", action="store_true")
    W = ap.parse_args()
    groups = set(W.only.split(",")) if W.only else None
    allj = [(g, j) for g, j in jobs() if groups is None or g in groups]
    os.makedirs(OUT, exist_ok=True)
    print("revision grid:", len(allj), "runs ->", os.path.abspath(OUT), flush=True)
    t0, done = time.time(), 0
    for i, (g, j) in enumerate(allj, 1):
        out = os.path.join(OUT, name(j) + ".json")
        if os.path.exists(out):
            print("[%d/%d] skip %s" % (i, len(allj), name(j))); continue
        cmd = [PY, os.path.join(HERE, "train.py"), "--net", j["net"], "--method", j["method"], "--seed", str(j["seed"]),
               "--rounds", str(ROUNDS), "--steps", str(STEPS), "--task", "op3", "--save_preds", "--out", OUT,
               "--split", j["split"]]
        if j["split"] != "fixed":
            cmd += ["--all_windows", "--fold", str(j.get("fold", 0))]
        if j.get("mtype"):
            cmd += ["--mtype", j["mtype"]]
        if j.get("defense"):
            cmd += ["--defense", j["defense"]]
        if j.get("holdout"):
            cmd += ["--holdout", j["holdout"]]
        eta = ""
        if done:
            per = (time.time() - t0) / done
            eta = "  ETA %.0f min" % (per * (len(allj) - i + 1) / 60)
        print("[%d/%d] %-10s %s%s" % (i, len(allj), g, name(j), eta), flush=True)
        if W.dry:
            print("   ", " ".join(cmd[1:])); continue
        r = subprocess.run(cmd)
        done += 1
        if r.returncode != 0:
            print("  FAILED rc", r.returncode, flush=True)
    print("queue done in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
