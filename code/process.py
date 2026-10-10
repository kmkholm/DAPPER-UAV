"""Turn the router captures + drone ground truth into per-drone binned traffic tensors.

For every drone: packets captured inside its cell-site router (outer, encrypted tunnel
packets only) -> 100 ms bins x [uplink size histogram (40), downlink size histogram (40),
uplink bytes, downlink bytes]; per-bin flight-phase label from the 5 Hz ground truth;
per-bin mission index from the GCS event log.

  python process.py <raw_dir> <out_dir>
raw_dir layout: <raw>/<node>/data/... as downloaded by fetch_data.py
"""
import glob, json, os, sys
import dpkt
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phases import load_gt, label

BIN = 0.1
EDGES = np.concatenate([np.arange(64, 577, 16), np.arange(704, 1601, 128), [100000]])   # 33 + 8 edges -> 40 buckets
NB = len(EDGES) - 1
NFEAT = 2 * NB + 2 + 6      # + per-bin inter-arrival statistics (mean, s.d., max; uplink and downlink), in ms


def drone_ip(net, site, sysid):
    o = 1 if net == "A" else 2
    k, j = sysid // 10, sysid % 10
    return "172.%d.%d.%d" % (15 + o, k, 4 * j + 2)


def read_pcaps(files, dip):
    dip_b = bytes(int(x) for x in dip.split("."))
    ts, ln, up = [], [], []
    for f in sorted(files):
        try:
            with open(f, "rb") as fh:
                for t, buf in dpkt.pcap.Reader(fh):
                    if len(buf) < 34 or buf[12:14] != b"\x08\x00":
                        continue                                   # IPv4 only (drops ARP)
                    ip = buf[14:34]
                    proto = ip[9]
                    if proto not in (17, 50):                      # UDP (WireGuard) or ESP
                        continue
                    ts.append(t); ln.append(int.from_bytes(ip[2:4], "big")); up.append(ip[12:16] == dip_b)
        except (dpkt.dpkt.NeedData, ValueError) as e:
            print("  truncated pcap", os.path.basename(f), e)
    o = np.argsort(np.asarray(ts, dtype=np.float64), kind="stable")
    return (np.asarray(ts, dtype=np.float64)[o], np.asarray(ln, dtype=np.int32)[o], np.asarray(up, dtype=bool)[o])


def bin_traffic(ts, ln, up, t0, nbin):
    X = np.zeros((nbin, NFEAT), dtype=np.float32)
    idx = ((ts - t0) / BIN).astype(np.int64)
    ok = (idx >= 0) & (idx < nbin)
    idx, ln, up = idx[ok], ln[ok], up[ok]
    b = np.clip(np.searchsorted(EDGES, ln, side="right") - 1, 0, NB - 1)
    col = np.where(up, b, NB + b)
    np.add.at(X, (idx, col), 1)
    np.add.at(X, (idx, np.where(up, 2 * NB, 2 * NB + 1)), ln)
    # per-bin inter-arrival time statistics per direction (the timing information that packet-level
    # analyses use): the IAT of each packet is assigned to the bin of that packet; stored in ms
    tsk = ts[ok]
    for d, base in ((True, 2 * NB + 2), (False, 2 * NB + 5)):
        m = up == d
        t = tsk[m]; bi = idx[m]
        if len(t) < 2:
            continue
        iat = np.diff(t) * 1000.0; bi = bi[1:]
        cnt = np.zeros(nbin); s1 = np.zeros(nbin); s2 = np.zeros(nbin); mx = np.zeros(nbin)
        np.add.at(cnt, bi, 1); np.add.at(s1, bi, iat); np.add.at(s2, bi, iat * iat); np.maximum.at(mx, bi, iat)
        mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
        var = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mean ** 2, 0.0)
        X[:, base] = mean; X[:, base + 1] = np.sqrt(np.clip(var, 0, None)); X[:, base + 2] = mx
    return X


def mission_index(events, sysid, tb):
    """Mission number active in each bin (0 = between missions)."""
    e = events[events.sysid == sysid].sort_values("t")
    mi = np.zeros(len(tb), dtype=np.int32)
    starts = e[e.event == "mission_start"]
    for _, r in starts.iterrows():
        n = int(str(r.mission_id).split("-")[-1])
        end = e[(e.t > r.t) & (e.event == "mission_end")]
        te = end.t.iloc[0] if len(end) else tb[-1]
        mi[(tb >= r.t) & (tb <= te + 30)] = n          # +30 s covers the post-landing log upload
    return mi


def process(raw, out, meta, only_net=None):
    os.makedirs(out, exist_ok=True)
    old = json.load(open(os.path.join(out, "summary.json"))) if (only_net and os.path.exists(os.path.join(out, "summary.json"))) else []
    summary = [r for r in old if r["net"] != only_net]
    for d in meta["drones"]:
        if only_net and d["net"] != only_net:
            continue
        net, site, sysid, veh = d["net"], d["site"], d["sysid"], d["vehicle"]
        node = "%s-d%d" % (site, sysid % 10)
        gtf = os.path.join(raw, node, "data", "gt_%d.csv" % sysid)
        caps = glob.glob(os.path.join(raw, "cell-%s" % site, "data", "cap_%s_%d_*.pcap" % (site, sysid)))
        if not os.path.exists(gtf) or not caps:
            print("missing", node); continue
        g = load_gt(gtf)
        lab_gt = label(g, veh)
        ts, ln, up = read_pcaps(caps, drone_ip(net, site, sysid))
        t0 = max(g.t.iloc[0], ts[0]); t1 = min(g.t.iloc[-1], ts[-1])
        nbin = int((t1 - t0) / BIN)
        X = bin_traffic(ts, ln, up, t0, nbin)
        tb = t0 + (np.arange(nbin) + 0.5) * BIN
        j = np.searchsorted(g.t.values, tb, side="right") - 1
        stale = (j < 0) | (tb - g.t.values[np.clip(j, 0, None)] > 1.0)       # gt gap (e.g. SITL restart)
        y = np.where(stale, -1, lab_gt[np.clip(j, 0, None)]).astype(np.int8)
        ev = pd.read_csv(os.path.join(raw, "gcs-%s" % net, "data", "gcs_events.csv"))
        mi = mission_index(ev, sysid, tb)
        np.savez_compressed(os.path.join(out, "%s_%d.npz" % (site, sysid)), X=X.astype(np.float32), y=y, mission=mi,
                            t0=t0, site=site, net=net, sysid=sysid, vehicle=veh)
        cnt = np.bincount(y[y >= 0], minlength=7)
        summary.append(dict(node=node, net=net, site=site, sysid=sysid, hours=nbin * BIN / 3600, packets=int(len(ts)),
                            pkt_per_s=float(len(ts) / (ts[-1] - ts[0])), missions=int(mi.max()),
                            label_counts=cnt.tolist()))
        print(summary[-1], flush=True)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    raw, out = sys.argv[1], sys.argv[2]
    meta = json.load(open(os.path.join(raw, "meta.json")))
    process(raw, out, meta, only_net=(sys.argv[3] if len(sys.argv) > 3 else None))
