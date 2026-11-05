"""Revision 1: C2 delivery delay under the timing defences, from the real packet traces (numpy only).

  jitter200   every packet held for U(0, 200) ms. Two variants: as evaluated (independent delays, packets
              may be reordered) and order-preserving (a FIFO cannot release a packet before its predecessor).
  const-R     a constant-rate shaper: R cells/s of 512 B in each direction; the real packets (inner bytes =
              outer size - 60 B tunnel overhead) wait in a FIFO byte queue and are fragmented across cells
              (452 B of payload per cell). Delay = departure of the last fragment - arrival. R = 20 is the
              setting of the leakage experiment; higher R shows the bandwidth needed for prompt delivery.
Delays are reported for all packets and for the ground-to-drone packets sent during tele-operation
(the pilot's command stream), per fleet.

  python rev1_latency.py   -> results/rev1/latency.json
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "proc")
OUT = os.path.join(HERE, "..", "results", "rev1")
BIN, CELL, OVH = 0.1, 512, 60
TELE = 5                                     # fine-phase index of tele-operation in both fleets
RATES = [20, 30, 40, 60, 100]


def fifo_cells(t, size, rate, t_end):
    """Departure time of every packet through a constant-rate cell shaper (FIFO, fragmentation)."""
    cap = CELL - OVH
    slot = 1.0 / rate
    k0 = np.floor(t / slot).astype(np.int64) + 1             # first slot after arrival
    base = int(k0.min()); nk = int(np.ceil(t_end / slot)) - base + 2
    arr = np.zeros(nk); np.add.at(arr, k0 - base, size)
    S = np.cumsum(arr - cap)                                    # Lindley recursion in closed form:
    B = S - np.minimum(np.minimum.accumulate(S), 0.0)           # backlog after each slot
    served = np.concatenate([[0.0], B[:-1]]) + arr - B
    backlog = float(B[-1])
    cum_s = np.cumsum(served)
    cum_a = np.cumsum(size)                                     # FIFO in arrival order
    j = np.searchsorted(cum_s, cum_a - 1e-9)
    j = np.minimum(j, nk - 1)
    return (j + base) * slot, backlog


def summarize(d):
    d = np.asarray(d) * 1000.0
    if len(d) == 0:
        return {}
    return dict(n=int(len(d)), mean=float(d.mean()), median=float(np.median(d)), p95=float(np.percentile(d, 95)),
                p99=float(np.percentile(d, 99)), max=float(d.max()), frac_over_100ms=float((d > 100).mean()),
                frac_over_1s=float((d > 1000).mean()))


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for net in ("A", "B"):
        acc = {k: [] for k in ["jit_reorder", "jit_fifo", "jit_fifo_tele"] + ["c%d" % r for r in RATES] + ["c%d_tele" % r for r in RATES] + ["c%d_c2" % r for r in RATES]}
        real_bytes, dur, unstable = 0.0, 0.0, {r: 0 for r in RATES}
        cover_bytes = {r: 0.0 for r in RATES}
        for f in sorted(glob.glob(os.path.join(PROC, "%s*_pk.npz" % net))):
            pk = np.load(f); z = np.load(f.replace("_pk.npz", ".npz"), allow_pickle=True)
            ts, ln, up = pk["ts"].astype(np.float64), pk["ln"].astype(np.float64), pk["up"].astype(bool)
            t0 = float(z["t0"]); y = z["y"].astype(np.int64)
            b = np.clip(((ts - t0) / BIN).astype(np.int64), 0, len(y) - 1)
            tele_dn = (~up) & (y[b] == TELE)
            real_bytes += ln.sum(); T0, T1 = ts[0], ts[-1]; dur += T1 - T0
            rng = np.random.default_rng(int(T0) % 2 ** 31)
            u = rng.uniform(0, 0.2, len(ts))
            acc["jit_reorder"].append(u)
            for dmask in (up, ~up):                              # order is kept per direction
                tt = ts[dmask] + u[dmask]
                dep = np.maximum.accumulate(tt)
                dl = dep - ts[dmask]
                acc["jit_fifo"].append(dl)
                acc["jit_fifo_tele"].append(dl[tele_dn[dmask]])
            for r in RATES:
                cover_bytes[r] += 2 * CELL * r * (T1 - T0)
                for dmask in (up, ~up):
                    tt = ts[dmask] - T0; sz = np.maximum(ln[dmask] - OVH, 1.0)
                    dep, left = fifo_cells(tt, sz, r, T1 - T0)
                    acc["c%d" % r].append(dep - tt)
                    acc["c%d_tele" % r].append((dep - tt)[tele_dn[dmask]])
                    acc["c%d_c2" % r].append((dep - tt)[ln[dmask] < 1400])     # MAVLink-sized packets (bulk log upload excluded)
                    if left > 50 * (CELL - OVH):
                        unstable[r] += 1
        out = {k: summarize(np.concatenate(v)) for k, v in acc.items()}
        out["real_rate_kbit_s_per_drone"] = float(real_bytes * 8 / dur / 1000.0)
        out["cover_overhead"] = {str(r): float(cover_bytes[r] / real_bytes - 1.0) for r in RATES}
        out["queues_not_drained"] = {str(r): unstable[r] for r in RATES}
        res[net] = out
        print(net, "real %.1f kbit/s per drone" % out["real_rate_kbit_s_per_drone"])
        for k in ["jit_reorder", "jit_fifo", "jit_fifo_tele"] + ["c%d" % r for r in RATES] + ["c%d_c2" % r for r in RATES] + ["c%d_tele" % r for r in RATES]:
            s = out[k]
            if s:
                print("  %-14s median %8.1f  p95 %9.1f  p99 %9.1f  max %10.1f ms  >100ms %.3f" % (k, s["median"], s["p95"], s["p99"], s["max"], s["frac_over_100ms"]))
        print("  overhead", {r: round(v, 2) for r, v in out["cover_overhead"].items()}, "undrained", out["queues_not_drained"])
    json.dump(res, open(os.path.join(OUT, "latency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
