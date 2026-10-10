"""Windows, labels, flight-disjoint splits and the three views of every window.

A window is T consecutive 100 ms bins (default 12.8 s) with a 2 s stride. Its label is the majority
flight phase; windows whose majority covers < 70 % of the bins (phase transitions) or that touch a
ground-truth gap are not used, and a window must lie inside one mission. Every (drone, mission) is
hashed to train / validation / test (70 / 15 / 15), so no mission contributes windows to two splits.

Each window is exposed to the encoder in three views, all computed from the outer (encrypted)
packets the cell-site router sees:
  bins  (T, NFEAT)   100 ms bins: per-direction size histogram, bytes, inter-arrival statistics
  pk    (NPK, 4)     up to NPK packet tokens: log size, direction, log inter-arrival (ms), position
  st    (40,)        window statistics of sizes / inter-arrivals per direction (the features of the
                     classical traffic-analysis pipelines)
The three views are cached per (net, T, stride, transform) under data/cache.
"""
import glob, hashlib, os
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process import bin_traffic, BIN, NFEAT

T_DEF, STRIDE, PURITY = 128, 20, 0.7
NPK, PK_F, ST_F = 256, 4, 40


def _stats8(v):
    if len(v) == 0:
        return [0.0] * 8
    return [len(v), float(np.mean(v)), float(np.std(v)), float(np.median(v)), float(np.percentile(v, 10)),
            float(np.percentile(v, 90)), float(np.max(v)), float(np.min(v))]


def window_stats(ts, ln, up, i0, i1, win):
    """The 40 size / inter-arrival statistics of the prior-work pipelines for packets [i0, i1)."""
    f = []
    for d in (True, False):
        m = up[i0:i1] == d
        sizes = ln[i0:i1][m].astype(np.float64); t = ts[i0:i1][m]
        iat = np.diff(t) if len(t) > 1 else np.zeros(0)
        f += _stats8(sizes) + _stats8(iat) + [float(sizes.sum()), float((sizes < 128).mean() if len(sizes) else 0.0),
                                               float((sizes > 1000).mean() if len(sizes) else 0.0)]
    nu, nd = f[0], f[19]
    f += [nu / max(nd, 1.0), (nu + nd) / win]
    return f


def packet_tokens(ts, ln, up, i0, i1, t_start, win):
    """Up to NPK packet tokens for packets [i0, i1); longer windows are sub-sampled uniformly."""
    n = i1 - i0
    tok = np.zeros((NPK, PK_F), np.float32); mask = np.zeros(NPK, bool)
    if n == 0:
        return tok, mask
    sel = np.arange(i0, i1) if n <= NPK else i0 + np.unique(np.linspace(0, n - 1, NPK).astype(np.int64))
    t = ts[sel]
    iat = np.diff(t, prepend=t_start) * 1000.0
    k = len(sel)
    tok[:k, 0] = np.log1p(ln[sel]); tok[:k, 1] = np.where(up[sel], 1.0, -1.0)
    tok[:k, 2] = np.log1p(np.clip(iat, 0, None)); tok[:k, 3] = (t - t_start) / win
    mask[:k] = True
    return tok, mask


def cut_at_idle(mission, q):
    i = int(len(mission) * q)
    while i < len(mission) and mission[i] != 0:
        i += 1
    return min(i, len(mission))


def load_network(proc_dir, net, T=T_DEF, stride=STRIDE, transform=None, cache=True, label_map=None):
    """Returns (bins tensor over all drones of the net, clients dict, views dict).
    transform(ts, ln, up) -> (ts, ln, up) applies a packet-level defence before every view is built."""
    files = sorted(glob.glob(os.path.join(proc_dir, "*.npz")))
    files = [f for f in files if not f.endswith("_pk.npz")]
    tname = getattr(transform, "__name__", "none") if transform is not None else "none"
    key = hashlib.md5(("|".join(os.path.basename(f) + str(os.path.getmtime(f)) for f in files) +
                       "%s_%d_%d_%s_%d_%s" % (net, T, stride, tname, NPK, "" if label_map is None else "".join(map(str, label_map)))).encode()).hexdigest()[:10]
    cdir = os.path.join(os.path.dirname(proc_dir.rstrip("/\\")), "cache"); os.makedirs(cdir, exist_ok=True)
    cfile = os.path.join(cdir, "%s_T%d_%s_%s.npz" % (net, T, tname, key))
    Xs, clients, offset = [], {}, 0
    PK, MK, ST = [], [], []
    bytes0, bytes1 = 0.0, 0.0
    cached = np.load(cfile, allow_pickle=True) if (cache and os.path.exists(cfile)) else None
    win = T * BIN
    for f in files:
        z = np.load(f, allow_pickle=True)
        if str(z["net"]) != net:
            continue
        site = str(z["site"]); sysid = int(z["sysid"]); t0 = float(z["t0"])
        y, mi = z["y"].astype(np.int64), z["mission"].astype(np.int64)
        if label_map is not None:                      # task labels before windowing: purity is judged on the task
            y = np.where(y >= 0, np.asarray(label_map)[np.clip(y, 0, None)], -1)
        n = len(y)
        pk = np.load(f.replace(".npz", "_pk.npz"))
        ts, ln, up = pk["ts"].astype(np.float64), pk["ln"].astype(np.int32), pk["up"].astype(bool)
        bytes0 += float(ln.sum())
        if transform is not None:
            ts, ln, up = transform(ts, ln, up)
        bytes1 += float(ln.sum())
        Xraw = bin_traffic(ts, ln, up, t0, n) if (transform is not None) else z["X"]
        X = np.log1p(np.nan_to_num(Xraw.astype(np.float32), nan=0.0, posinf=6.5e4, neginf=0.0))
        starts = np.arange(0, n - T, stride)
        ends = starts + T
        oh = np.zeros((n + 1, 8), dtype=np.int32)
        valid = y >= 0
        oh[1:][np.arange(n)[valid], y[valid]] = 1
        cs = oh.cumsum(0)
        cnt = cs[ends] - cs[starts]
        lab = cnt.argmax(1)
        pur = cnt.max(1) / T
        wmi = mi[starts + T // 2]
        same = mi[starts] == mi[ends - 1]
        ok = (pur >= PURITY) & (cnt.sum(1) == T) & same & (wmi > 0)
        h = ((sysid * 1_000_003 + wmi) * 2_654_435_761) % 100
        split = np.where(h < 70, 0, np.where(h < 85, 1, 2))
        d = clients.setdefault(site, {"start": [], "y": [], "split": [], "drone": [], "mission": [], "row": []})
        st_ok = starts[ok]
        d["start"].append(st_ok + offset); d["y"].append(lab[ok]); d["split"].append(split[ok])
        d["drone"].append(np.full(int(ok.sum()), sysid)); d["mission"].append(wmi[ok])
        row0 = sum(len(p) for p in PK)
        d["row"].append(np.arange(row0, row0 + len(st_ok)))
        key_f = "%s_%d" % (site, sysid)
        if cached is not None and (key_f + "_pk") in cached.files:
            PK.append(cached[key_f + "_pk"]); MK.append(cached[key_f + "_mk"]); ST.append(cached[key_f + "_st"])
        else:
            tks = np.zeros((len(st_ok), NPK, PK_F), np.float16); mks = np.zeros((len(st_ok), NPK), bool)
            sts = np.zeros((len(st_ok), ST_F), np.float32)
            for j, s in enumerate(st_ok):
                t_start = t0 + s * BIN
                i0, i1 = np.searchsorted(ts, t_start), np.searchsorted(ts, t_start + win)
                tk, mk = packet_tokens(ts, ln, up, i0, i1, t_start, win)
                tks[j] = tk; mks[j] = mk; sts[j] = window_stats(ts, ln, up, i0, i1, win)
            PK.append(tks); MK.append(mks); ST.append(sts)
        Xs.append(X); offset += n
    X = torch.from_numpy(np.concatenate(Xs)).half()
    for s, d in clients.items():
        for k in d:
            d[k] = np.concatenate(d[k])
    views = dict(pk=np.concatenate(PK), mask=np.concatenate(MK), st=np.concatenate(ST), overhead=bytes1 / max(bytes0, 1.0) - 1.0)
    if cache and cached is None:
        pay = {}
        r = 0
        for f in files:
            z = np.load(f, allow_pickle=True)
            if str(z["net"]) != net:
                continue
            key_f = "%s_%d" % (str(z["site"]), int(z["sysid"]))
            m = clients[str(z["site"])]
            nrow = int((m["drone"] == int(z["sysid"])).sum())
            pay[key_f + "_pk"] = views["pk"][r:r + nrow]; pay[key_f + "_mk"] = views["mask"][r:r + nrow]
            pay[key_f + "_st"] = views["st"][r:r + nrow]; r += nrow
        np.savez_compressed(cfile, **pay)
    views["start"] = np.concatenate([clients[s]["start"] for s in sorted(clients)])
    views["row"] = np.concatenate([clients[s]["row"] for s in sorted(clients)])
    return X, clients, views


class WindowStore:
    """Gathers the three views of a set of windows (everything kept on the GPU)."""
    def __init__(self, X, views, T=T_DEF, device="cuda"):
        self.X = X.to(device)
        self.T = T
        self.ar = torch.arange(T, device=device)
        self.starts = torch.zeros(int(views["row"].max()) + 1, dtype=torch.long, device=device)
        self.starts[torch.as_tensor(views["row"], device=device)] = torch.as_tensor(views["start"], device=device)
        self.pk = torch.from_numpy(views["pk"]).to(device)
        self.mask = torch.from_numpy(views["mask"]).to(device)
        self.st = torch.from_numpy(np.log1p(np.clip(views["st"], 0, None)).astype(np.float32)).to(device)

    def get(self, rows):
        r = torch.as_tensor(np.asarray(rows), device=self.X.device, dtype=torch.long)
        s = self.starts[r]
        return (self.X[s[:, None] + self.ar[None, :]].float(), self.pk[r].float(), self.mask[r], self.st[r])


def stats(store, rows, n=4000, rng=None):
    """Site statistics (mean, s.d.) of each view for the personal calibration."""
    rng = rng or np.random.default_rng(0)
    pick = rng.choice(rows, size=min(n, len(rows)), replace=False)
    xb, pk, mk, st = store.get(pick)
    x = xb.reshape(-1, xb.shape[-1])
    p = pk[mk]
    return dict(bins=(x.mean(0), x.std(0).clamp_min(1e-3)), pk=(p.mean(0), p.std(0).clamp_min(1e-3)),
                st=(st.mean(0), st.std(0).clamp_min(1e-3)))
