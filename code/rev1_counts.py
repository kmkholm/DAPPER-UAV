"""Revision 1: windows and independent sorties per drone, operational state and split (op3 task).

A sortie is one (drone, mission) pair; every window of a sortie falls in the same split, so the number
of sorties is the effective number of independent units behind each test score.

  python rev1_counts.py            -> results/rev1/sortie_counts.json + printed table
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "proc")
OUT = os.path.join(HERE, "..", "results", "rev1")
T, STRIDE, PURITY = 128, 20, 0.7
OP3 = {"A": [0, 1, 1, 1, 1, 2, 1], "B": [0, 1, 1, 1, 1, 2]}
SPLITS = ["train", "val", "test"]


def windows(f, net):
    z = np.load(f, allow_pickle=True)
    sysid = int(z["sysid"])
    y, mi = z["y"].astype(np.int64), z["mission"].astype(np.int64)
    y = np.where(y >= 0, np.asarray(OP3[net])[np.clip(y, 0, None)], -1)
    n = len(y)
    starts = np.arange(0, n - T, STRIDE); ends = starts + T
    oh = np.zeros((n + 1, 8), np.int32); v = y >= 0
    oh[1:][np.arange(n)[v], y[v]] = 1
    cs = oh.cumsum(0); cnt = cs[ends] - cs[starts]
    lab = cnt.argmax(1); pur = cnt.max(1) / T
    wmi = mi[starts + T // 2]
    ok = (pur >= PURITY) & (cnt.sum(1) == T) & (mi[starts] == mi[ends - 1]) & (wmi > 0)
    h = ((sysid * 1_000_003 + wmi) * 2_654_435_761) % 100
    split = np.where(h < 70, 0, np.where(h < 85, 1, 2))
    # windows dropped by the purity rule only (transition windows) inside missions
    trans = (pur < PURITY) & (cnt.sum(1) == T) & (mi[starts] == mi[ends - 1]) & (wmi > 0)
    return str(z["site"]), sysid, lab[ok], split[ok], wmi[ok], int(trans.sum()), int(ok.sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, summary = [], {}
    for net in ("A", "B"):
        tot = {"windows": np.zeros((3, 3), int), "sorties": np.zeros((3, 3), int), "transition_windows": 0, "kept_windows": 0,
               "sorties_any": np.zeros(3, int)}
        for f in sorted(glob.glob(os.path.join(PROC, "%s*_*.npz" % net))):
            if f.endswith("_pk.npz"):
                continue
            site, sysid, lab, split, mis, ntr, nok = windows(f, net)
            tot["transition_windows"] += ntr; tot["kept_windows"] += nok
            for s in range(3):
                m = split == s
                tot["sorties_any"][s] += len(np.unique(mis[m]))
                for c in range(3):
                    mc = m & (lab == c)
                    nw, ns = int(mc.sum()), len(np.unique(mis[mc]))
                    tot["windows"][s, c] += nw; tot["sorties"][s, c] += ns
                    rows.append(dict(net=net, site=site, drone=sysid, split=SPLITS[s], state=c, windows=nw, sorties=ns))
        summary[net] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in tot.items()}
    json.dump(dict(rows=rows, summary=summary, states=["Ground", "Autonomous flight", "Tele-operation"]),
              open(os.path.join(OUT, "sortie_counts.json"), "w"), indent=1)
    for net, s in summary.items():
        print(net, "kept windows", s["kept_windows"], "transition windows dropped", s["transition_windows"])
        for i, sp in enumerate(SPLITS):
            print("  %-5s sorties(any) %3d | windows %s | sorties with state %s" % (sp, s["sorties_any"][i], s["windows"][i], s["sorties"][i]))


if __name__ == "__main__":
    main()
