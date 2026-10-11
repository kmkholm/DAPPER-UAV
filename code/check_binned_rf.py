"""Diagnostic: how much of the prior-work RF accuracy survives when the RF only sees window
aggregates (mean / s.d. / max / min over the bins) of OUR binned features? Tells whether the
binned representation (not the deep model) is what loses information.

  python check_binned_rf.py --net A [--proc ../data/proc_v1] [--T 128]
"""
import argparse, os, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_network

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="A"); ap.add_argument("--T", type=int, default=128)
ap.add_argument("--proc", default=os.path.join(os.path.dirname(__file__), "..", "data", "proc"))
A = ap.parse_args()
COARSE = {"A": np.array([0, 2, 1, 2, 1, 3, 2]), "B": np.array([0, 2, 2, 2, 1, 3])}[A.net]
NC = 7 if A.net == "A" else 6

X, clients = load_network(A.proc, A.net, T=A.T)
X = X.float().numpy()
ar = np.arange(A.T)
feats, ys, sps, sites = [], [], [], []
for s in sorted(clients):
    d = clients[s]
    for st, y, sp in zip(d["start"], d["y"], d["split"]):
        w = X[st + ar]
        feats.append(np.concatenate([w.mean(0), w.std(0), w.max(0), w.min(0)]))
        ys.append(int(y)); sps.append(int(sp)); sites.append(s)
F = np.asarray(feats, np.float32); y = np.asarray(ys); sp = np.asarray(sps); site = np.asarray(sites)
print("windows", len(y), "features", F.shape[1], "proc", os.path.basename(os.path.abspath(A.proc)), "T", A.T)
for setting in ("per site", "pooled"):
    yp_f = np.full(len(y), -1); yp_c = np.full(len(y), -1); yc = COARSE[y]
    groups = [None] if setting == "pooled" else np.unique(site)
    for g in groups:
        tr = (sp == 0) if g is None else (sp == 0) & (site == g)
        te = (sp == 2) if g is None else (sp == 2) & (site == g)
        if not tr.sum() or not te.sum():
            continue
        for lab, out in ((y, yp_f), (yc, yp_c)):
            m = RandomForestClassifier(300, random_state=1, n_jobs=8, class_weight="balanced").fit(F[tr], lab[tr])
            out[te] = m.predict(F[te])
    te = sp == 2
    print("%-9s fine F1 %.3f acc %.3f | op F1 %.3f acc %.3f" % (
        setting, f1_score(y[te], yp_f[te], average="macro", labels=range(NC), zero_division=0), accuracy_score(y[te], yp_f[te]),
        f1_score(yc[te], yp_c[te], average="macro", labels=range(4), zero_division=0), accuracy_score(yc[te], yp_c[te])), flush=True)
