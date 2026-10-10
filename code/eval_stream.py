"""Stream-level evaluation of saved models: per-window posteriors on the held-out windows of
each site, ordered in time within each (drone, mission); then (i) the fine phase metrics,
(ii) the operational 4-phase metrics derived from the same posteriors, and (iii) both after
temporal smoothing of the posteriors over k consecutive windows (stride 2 s) of the same flight.

  python eval_stream.py --net A --method dapper --seeds 1 2 3 --k 1 5 9
"""
import argparse, json, os, sys
import numpy as np, torch
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_network, WindowStore
from model import DAPPER

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="A"); ap.add_argument("--method", default="dapper")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 9])
ap.add_argument("--proc", default=os.path.join(os.path.dirname(__file__), "..", "data", "proc"))
ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "results", "models"))
ap.add_argument("--coarse", action="store_true", help="alias for --task op4")
ap.add_argument("--task", default="", help="fine | op4 | op3: which checkpoints (A_/Ac_/At_) to evaluate")
ap.add_argument("--json", default="", help="write the summary to this file")
A = ap.parse_args()
A.task = A.task or ("op4" if A.coarse else "fine")
TT = {"fine": "", "op4": "c", "op3": "t"}[A.task]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# fine -> operational phases: 0 ground, 1 stationary (hover/orbit/loiter), 2 transit (ascent/cruise/descent), 3 tele-operation
COARSE = {"A": np.array([0, 2, 1, 2, 1, 3, 2]), "B": np.array([0, 2, 2, 2, 1, 3])}[A.net]
TASK_MAP = {"op4": {"A": [0, 2, 1, 2, 1, 3, 2], "B": [0, 2, 2, 2, 1, 3]}, "op3": {"A": [0, 1, 1, 1, 1, 2, 1], "B": [0, 1, 1, 1, 1, 2]}}
X, clients, VIEWS = load_network(A.proc, A.net, T=128, label_map=None if A.task == "fine" else np.array(TASK_MAP[A.task][A.net]))
store = WindowStore(X, VIEWS, T=128, device=DEV)


def smooth_group(P, k):
    if k <= 1 or len(P) < 2:
        return P
    out = np.empty_like(P); h = k // 2
    for i in range(len(P)):
        out[i] = P[max(0, i - h):i + h + 1].mean(0)
    return out


def run(seed):
    ck = torch.load(os.path.join(A.models, "%s_%s_s%d.pt" % (A.net + TT, A.method, seed)), map_location=DEV, weights_only=False)
    res = {k: {"y": [], "p": []} for k in A.k}
    for site in ck["sites"]:
        if site not in clients:
            continue
        m = DAPPER(**ck["cfg"]).to(DEV); m.load_state_dict(ck["sites"][site]); m.eval()
        d = clients[site]; sel = d["split"] == 2
        st, y, dr, mi, rows = d["start"][sel], d["y"][sel], d["drone"][sel], d["mission"][sel], d["row"][sel]
        order = np.lexsort((st, mi, dr)); st, y, dr, mi, rows = st[order], y[order], dr[order], mi[order], rows[order]
        P = []
        with torch.no_grad():
            for i in range(0, len(st), 512):
                P.append(torch.softmax(m(*store.get(rows[i:i + 512])), -1).cpu().numpy())
        P = np.concatenate(P) if P else np.zeros((0, len(COARSE)))
        key = dr * 100000 + mi
        for k in A.k:
            Ps = P.copy()
            for g in np.unique(key):
                idx = np.where(key == g)[0]; Ps[idx] = smooth_group(P[idx], k)
            res[k]["y"].append(y); res[k]["p"].append(Ps)
    out = {}
    for k in A.k:
        y = np.concatenate(res[k]["y"]); P = np.concatenate(res[k]["p"]); yp = P.argmax(1)
        if A.task == "fine":
            yc = COARSE[y]; Pc = np.zeros((len(P), 4))
            for c in range(len(COARSE)):
                Pc[:, COARSE[c]] += P[:, c]
        else:
            yc, Pc = y, P
        ypc = Pc.argmax(1)
        out[k] = dict(fine_f1=f1_score(y, yp, average="macro"), fine_acc=accuracy_score(y, yp),
                      op_f1=f1_score(yc, ypc, average="macro"), op_acc=accuracy_score(yc, ypc), op_bacc=balanced_accuracy_score(yc, ypc))
    return out


allr = [run(s) for s in A.seeds]
summary = {}
for k in A.k:
    agg = {m: float(np.mean([r[k][m] for r in allr])) for m in allr[0][k]}
    summary["k%d" % k] = agg
    print("k=%d windows (%2d s context): fine macro-F1 %.3f acc %.3f | operational 4-phase macro-F1 %.3f acc %.3f bacc %.3f"
          % (k, 12.8 + 2 * (k - 1), agg["fine_f1"], agg["fine_acc"], agg["op_f1"], agg["op_acc"], agg["op_bacc"]))
if A.json:
    json.dump(summary, open(A.json, "w"), indent=1); print("wrote", A.json)
