"""Prior encrypted-traffic approaches re-implemented on OUR data (same windows, same splits).

The published UAV traffic-analysis studies (Sciancalepore et al. 2019; PiNcH, 2020; Alipour-Fanid
et al. 2020) classify fixed-length captures with hand-crafted statistics of packet sizes and
inter-arrival times fed to classical learners (random forest, SVM). We rebuild that pipeline from
the raw packets of every held-out window and train it (i) per site, as those single-vantage-point
studies did, and (ii) on the pooled sites. Metrics: fine phases and the operational taxonomy.

  python prior_baselines.py --net A
Writes results/runs/prior_<net>.json
"""
import argparse, glob, json, os, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_network, T_DEF
from process import read_pcaps, drone_ip, BIN

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="A"); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--coarse", action="store_true", help="alias for --task op4")
ap.add_argument("--task", default="", help="fine | op4 | op3 (see train.py)")
ap.add_argument("--T", type=int, default=T_DEF)
ap.add_argument("--proc", default=os.path.join(os.path.dirname(__file__), "..", "data", "proc"))
ap.add_argument("--raw", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "runs"))
A = ap.parse_args()
A.task = A.task or ("op4" if A.coarse else "fine")
A.coarse = A.task != "fine"
TASK_MAP = {"op4": {"A": [0, 2, 1, 2, 1, 3, 2], "B": [0, 2, 2, 2, 1, 3]}, "op3": {"A": [0, 1, 1, 1, 1, 2, 1], "B": [0, 1, 1, 1, 1, 2]}}
COARSE = np.array(TASK_MAP["op4" if A.task == "fine" else A.task][A.net])
NCO = 3 if A.task == "op3" else 4
NC = 7 if A.net == "A" else 6
WIN = A.T * BIN


def stats(v):
    if len(v) == 0:
        return [0.0] * 8
    return [len(v), float(np.mean(v)), float(np.std(v)), float(np.median(v)), float(np.percentile(v, 10)),
            float(np.percentile(v, 90)), float(np.max(v)), float(np.min(v))]


def window_features(ts, ln, up, t_start):
    i0, i1 = np.searchsorted(ts, t_start), np.searchsorted(ts, t_start + WIN)
    f = []
    for d in (True, False):
        m = up[i0:i1] == d
        sizes = ln[i0:i1][m]; t = ts[i0:i1][m]
        iat = np.diff(t) if len(t) > 1 else np.zeros(0)
        f += stats(sizes) + stats(iat) + [float(sizes.sum()), float((sizes < 128).mean() if len(sizes) else 0.0),
                                          float((sizes > 1000).mean() if len(sizes) else 0.0)]
    nu, nd = f[0], f[19]
    f += [nu / max(nd, 1.0), (nu + nd) / WIN]
    return f


def main():
    X, clients, _ = load_network(A.proc, A.net, T=A.T, label_map=COARSE if A.coarse else None)
    # map every window back to absolute time via the per-file t0 and bin offsets
    files = [f for f in sorted(glob.glob(os.path.join(A.proc, "*.npz"))) if not f.endswith("_pk.npz")]
    meta = []
    off = 0
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        if str(z["net"]) != A.net:
            continue
        n = len(z["y"]); meta.append((off, off + n, float(z["t0"]), str(z["site"]), int(z["sysid"]))); off += n
    cache = {}

    def packets(site, sysid):
        key = (site, sysid)
        if key not in cache:
            caps = glob.glob(os.path.join(A.raw, "cell-%s" % site, "data", "cap_%s_%d_*.pcap" % (site, sysid)))
            cache[key] = read_pcaps(caps, drone_ip(A.net, site, sysid))
        return cache[key]

    feats, ys, splits, sites = [], [], [], []
    for site in sorted(clients):
        d = clients[site]
        for st, y, sp in zip(d["start"], d["y"], d["split"]):
            for (a, b, t0, s_, sysid) in meta:
                if a <= st < b:
                    ts, ln, up = packets(s_, sysid)
                    feats.append(window_features(ts, ln, up, t0 + (st - a) * BIN)); break
            ys.append(int(y)); splits.append(int(sp)); sites.append(site)
    F = np.nan_to_num(np.asarray(feats, np.float32)); y = np.asarray(ys); sp = np.asarray(splits); site = np.asarray(sites)
    print("windows", len(y), "features", F.shape[1])

    def fit_eval(model_fn, train_mask, test_mask, ylab):
        m = model_fn(); m.fit(F[train_mask], ylab[train_mask]); p = m.predict(F[test_mask])
        return p

    results = {}
    for name, mk in (("Random forest (size + IAT statistics)", lambda: RandomForestClassifier(300, random_state=A.seed, n_jobs=8, class_weight="balanced")),
                     ("SVM-RBF (size + IAT statistics)", lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")))):
        for setting in ("per site", "pooled"):
            yp_f = np.full(len(y), -1); yp_c = np.full(len(y), -1)
            yc = y if A.coarse else COARSE[y]
            if setting == "pooled":
                tr, te = sp == 0, sp == 2
                yp_c[te] = fit_eval(mk, tr, te, yc); yp_f[te] = yp_c[te] if A.coarse else fit_eval(mk, tr, te, y)
            else:
                for s_ in np.unique(site):
                    tr, te = (sp == 0) & (site == s_), (sp == 2) & (site == s_)
                    if tr.sum() and te.sum():
                        yp_c[te] = fit_eval(mk, tr, te, yc); yp_f[te] = yp_c[te] if A.coarse else fit_eval(mk, tr, te, y)
            te = sp == 2
            r = dict(fine_f1=float(f1_score(y[te], yp_f[te], average="macro", labels=range(NC), zero_division=0)),
                     fine_acc=float(accuracy_score(y[te], yp_f[te])),
                     op_f1=float(f1_score(yc[te], yp_c[te], average="macro", labels=range(NCO), zero_division=0)),
                     op_acc=float(accuracy_score(yc[te], yp_c[te])), op_bacc=float(balanced_accuracy_score(yc[te], yp_c[te])),
                     bin_acc=float(accuracy_score(yc[te] == 0, yp_c[te] == 0)))
            results["%s | %s" % (name, setting)] = r
            print("%-45s %-9s fine F1 %.3f | op F1 %.3f acc %.3f | airborne-vs-ground acc %.3f" % (name, setting, r["fine_f1"], r["op_f1"], r["op_acc"], r["bin_acc"]), flush=True)
    os.makedirs(A.out, exist_ok=True)
    json.dump(results, open(os.path.join(A.out, "prior_%s%s%s.json" % (A.net, {"fine": "", "op4": "c", "op3": "t"}[A.task], "" if A.T == T_DEF else "_T%d" % A.T)), "w"), indent=1)


if __name__ == "__main__":
    main()
