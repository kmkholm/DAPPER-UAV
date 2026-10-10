"""Fixed experiment grid for the DAPPER paper. Resumable: a run whose result json
already exists is skipped. One ETA, no open-ended additions.

Grid (per network A = multirotor fleet, B = VTOL fleet):
  main        : local, central, fedavg, fedprox, fedbn, dapper          x seeds 1-3
  ablation    : dapper_softmax, dapper_cal, dapper_prompt               x seeds 1-3
  onboarding  : dapper, fedavg  with one held-out site (A3 / B3)        x seeds 1-3
"""
import argparse, itertools, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "runs")
PY = sys.executable
SEEDS = [1, 2, 3]
MAIN = ["local", "central", "fedavg", "fedprox", "fedbn", "dapper"]
ABL = ["dapper_softmax", "dapper_nogate", "dapper_cal", "dapper_prompt", "dapper_nost", "dapper_nobins", "dapper_pk"]
DEF = ["pad256", "pad512", "jitter200", "const"]
OP4 = {"A": ["dapper", "central", "fedavg", "local"], "B": ["dapper", "central", "fedavg", "local"]}
FINE = {"A": ["dapper", "central", "fedavg", "local"], "B": ["dapper", "central", "fedavg", "local"]}
HOLD = {"A": "A3", "B": "B3"}
ROUNDS, STEPS = 20, 50


def jobs():
    for net in ("A", "B"):
        for m, s in itertools.product(MAIN, SEEDS):
            yield dict(net=net, method=m, seed=s, task="op3")
    for net in ("A", "B"):
        for m, s in itertools.product(ABL, SEEDS[:2]):
            yield dict(net=net, method=m, seed=s, task="op3")
    for net in ("A", "B"):
        for m, s in itertools.product(["dapper", "fedavg"], SEEDS[:2]):
            yield dict(net=net, method=m, seed=s, task="op3", holdout=HOLD[net])
        for d in DEF:
            yield dict(net=net, method="dapper", seed=1, task="op3", defense=d)
    for net in ("A", "B"):
        for m, s in itertools.product(OP4[net], SEEDS):
            yield dict(net=net, method=m, seed=s, task="op4")
    for net in ("A", "B"):
        for m, s in itertools.product(FINE[net], SEEDS):
            yield dict(net=net, method=m, seed=s, task="fine")


def name(j):
    return "%s%s_%s_s%d%s%s" % (j["net"], {"fine": "", "op4": "c", "op3": "t"}[j["task"]], j["method"], j["seed"],
                                ("_ho" + j["holdout"]) if j.get("holdout") else "", ("_def" + j["defense"]) if j.get("defense") else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=1); ap.add_argument("--worker", type=int, default=0)
    W = ap.parse_args()
    allj = list(jobs())
    print("grid:", len(allj), "runs; worker %d/%d" % (W.worker, W.workers))
    t0 = time.time()
    for i, j in enumerate(allj, 1):
        if (i - 1) % W.workers != W.worker:
            continue
        out = os.path.join(OUT, name(j) + ".json")
        if os.path.exists(out):
            print("[%d/%d] skip %s" % (i, len(allj), name(j))); continue
        cmd = [PY, os.path.join(HERE, "train.py"), "--net", j["net"], "--method", j["method"],
               "--seed", str(j["seed"]), "--rounds", str(ROUNDS), "--steps", str(STEPS), "--save_preds", "--task", j["task"]]
        if j["method"] == "dapper" and not j.get("holdout") and not j.get("defense"):
            cmd += ["--save_model"]
        if j.get("holdout"):
            cmd += ["--holdout", j["holdout"]]
        if j.get("defense"):
            cmd += ["--defense", j["defense"]]
        print("[%d/%d] %s  (elapsed %.1f min)" % (i, len(allj), name(j), (time.time() - t0) / 60), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("  FAILED rc", r.returncode, flush=True)
    print("queue done in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
