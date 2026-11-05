"""Static check (numpy only, no torch): run train.py's assign_splits() on the real windows and report
sorties / windows per split and state for every protocol of the revision grid."""
import ast, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rev1_counts import OP3, T, STRIDE, PROC

src = open(os.path.join(HERE, "train.py"), encoding="utf-8").read()
tree = ast.parse(src)
fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_h", "assign_splits")]
code = compile(ast.Module(body=fns, type_ignores=[]), "train_fns", "exec")


def build_clients(net, purity_min):
    clients = {}
    for f in sorted(glob.glob(os.path.join(PROC, "%s*_*.npz" % net))):
        if f.endswith("_pk.npz"):
            continue
        z = np.load(f, allow_pickle=True)
        sysid = int(z["sysid"]); site = str(z["site"])
        y, mi = z["y"].astype(np.int64), z["mission"].astype(np.int64)
        y = np.where(y >= 0, np.asarray(OP3[net])[np.clip(y, 0, None)], -1)
        n = len(y); starts = np.arange(0, n - T, STRIDE); ends = starts + T
        oh = np.zeros((n + 1, 8), np.int32); v = y >= 0
        oh[1:][np.arange(n)[v], y[v]] = 1
        cs = oh.cumsum(0); cnt = cs[ends] - cs[starts]
        lab = cnt.argmax(1); pur = cnt.max(1) / T; wmi = mi[starts + T // 2]
        ok = (pur >= purity_min) & (cnt.sum(1) == T) & (mi[starts] == mi[ends - 1]) & (wmi > 0)
        h = ((sysid * 1_000_003 + wmi) * 2_654_435_761) % 100
        split = np.where(h < 70, 0, np.where(h < 85, 1, 2))
        d = clients.setdefault(site, {k: [] for k in ("y", "split", "drone", "mission", "pur")})
        d["y"].append(lab[ok]); d["split"].append(split[ok]); d["drone"].append(np.full(int(ok.sum()), sysid))
        d["mission"].append(wmi[ok]); d["pur"].append(pur[ok])
    for d in clients.values():
        for k in d:
            d[k] = np.concatenate(d[k])
    return clients


class Args:
    pass


def check(net, split, fold=0, mtype=""):
    A = Args(); A.split, A.fold, A.mtype, A.net, A.proc = split, fold, mtype, net, PROC
    g = dict(np=np, json=json, os=os, A=A, CLASSES=["Ground", "Autonomous flight", "Tele-operation"], PURITY=0.7,
             clients=build_clients(net, 0.0))
    exec(code, g)
    n = g["assign_splits"]()
    cl = g["clients"]
    out = {}
    for sp in (0, 1, 2):
        w = np.zeros(3, int); so = [set(), set(), set()]
        for s, d in cl.items():
            m = d["split"] == sp
            for c in range(3):
                mc = m & (d["y"] == c); w[c] += mc.sum()
                so[c] |= set(zip(d["drone"][mc].tolist(), d["mission"][mc].tolist()))
        out[sp] = (w.tolist(), [len(x) for x in so])
    trans = sum(int(((d["split_all"] == 2) & (d["split"] == -1)).sum()) for d in cl.values())
    return out, trans, n


if __name__ == "__main__":
    rows = []
    for net in ("A", "B"):
        for split, folds in (("cv5", range(5)), ("lodo", (1, 2, 3)),
                             ("lomt", ("delivery", "survey", "inspection") if net == "A" else ("patrol", "survey", "inspection"))):
            for f in folds:
                out, trans, n = check(net, split, f if split != "lomt" else 0, f if split == "lomt" else "")
                print("%s %-4s %-10s test windows %s sorties %s | train sorties-with-state %s | test transition windows %d"
                      % (net, split, f, out[2][0], out[2][1], out[0][1], trans))
                rows.append(dict(net=net, split=split, fold=str(f), test_windows=out[2][0], test_sorties=out[2][1],
                                 val_windows=out[1][0], train_windows=out[0][0], train_sorties=out[0][1], test_transition=trans))
    os.makedirs(os.path.join(HERE, "..", "results", "rev1"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "..", "results", "rev1", "split_check.json"), "w"), indent=1)
