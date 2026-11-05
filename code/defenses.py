"""Packet-level traffic-shaping defences, applied to the outer (encrypted) packet stream of every drone
before any view (bins, packet tokens, window statistics) is built, so the adversary is retrained on
the defended traffic.  Each transform maps (timestamps, IP lengths, is_uplink) to the same triple.

  pad256 / pad512 : every packet is padded up to the next multiple of 256 / 512 bytes
  jitter200       : every packet is delayed by an independent uniform 0-200 ms (timing obfuscation,
                    no byte overhead)
  const           : constant-rate cover traffic, 20 packets/s of 512 bytes in each direction, real
                    packets absorbed into the stream (sizes and timing both hidden); an idealised
                    offline transformation (C2 delivery and delay are analysed in rev1_latency.py)

Revision 1 (companion-traffic controls, not defences): label-blind removal of the companion flows
whose presence depends on the flight state, applied to every packet of every window.
  nocomp          : network Remote-ID reports (1 Hz while armed; outer sizes 268 B over WireGuard,
                    256-268 B over ESP) and the post-landing log upload (TCP segments >= 1400 B,
                    never produced by the MAVLink stack, plus the acknowledgements returned during
                    each upload burst) are removed
  nocomp_human    : nocomp + a humanised command cadence: the ground-to-drone stream is thinned
                    per 1 s block with a random keep rate U(0.35, 1), 20 % of the blocks are pauses
                    (no packet), and every kept packet is shifted by N(0, 40 ms)
"""
import numpy as np


def _pad(k):
    def f(ts, ln, up):
        return ts, (np.ceil(np.asarray(ln, np.float64) / k) * k).astype(np.int32), up
    f.__name__ = "pad%d" % k
    return f


pad256, pad512 = _pad(256), _pad(512)


def jitter200(ts, ln, up):
    rng = np.random.default_rng(0)
    t = ts + rng.uniform(0.0, 0.2, len(ts))
    o = np.argsort(t, kind="stable")
    return t[o], ln[o], up[o]


def const(ts, ln, up):
    grid = np.arange(ts[0], ts[-1], 0.05)
    t = np.concatenate([grid, grid + 0.025])
    d = np.concatenate([np.ones(len(grid), bool), np.zeros(len(grid), bool)])
    o = np.argsort(t, kind="stable")
    return t[o], np.full(len(t), 512, np.int32), d[o]


RID_SIZES = np.array([256, 260, 264, 268])        # outer sizes of the Remote-ID report (WireGuard 268; ESP 256-268)
BULK_MIN = 1400                                   # largest MAVLink-stack packet in the captures: 1356 B


def _seed(ts):
    return int(abs(float(ts[0])) * 1000) % (2 ** 31) if len(ts) else 0


def nocomp(ts, ln, up):
    ts, ln, up = np.asarray(ts), np.asarray(ln), np.asarray(up, bool)
    drop = up & np.isin(ln, RID_SIZES)                         # Remote-ID
    bulk = up & (ln >= BULK_MIN)                               # log-upload segments
    drop |= bulk
    tb = ts[bulk]
    if len(tb):
        cuts = np.where(np.diff(tb) > 2.0)[0] + 1
        lo = np.array([g[0] for g in np.split(tb, cuts)]) - 0.2
        hi = np.array([g[-1] for g in np.split(tb, cuts)]) + 0.5
        k = np.searchsorted(lo, ts, side="right") - 1
        inside = (k >= 0) & (ts <= hi[np.clip(k, 0, None)])
        dn_in = inside & ~up
        if dn_in.any():
            ack = np.bincount(ln[dn_in]).argmax()              # acknowledgement size of this drone's uploads
            drop |= dn_in & (ln == ack)
    keep = ~drop
    return ts[keep], ln[keep], up[keep]


def nocomp_human(ts, ln, up):
    ts, ln, up = nocomp(ts, ln, up)
    rng = np.random.default_rng(_seed(ts))
    if len(ts) == 0:
        return ts, ln, up
    blk = ((ts - ts[0]) // 1.0).astype(np.int64)
    nb = int(blk.max()) + 1
    rate = rng.uniform(0.35, 1.0, nb) * (rng.random(nb) >= 0.2)
    dn = ~up
    keep = up | (rng.random(len(ts)) < rate[blk])
    t = ts.copy()
    t[dn] = t[dn] + rng.normal(0.0, 0.04, int(dn.sum()))
    t, ln, up = t[keep], ln[keep], up[keep]
    o = np.argsort(t, kind="stable")
    return t[o], ln[o], up[o]


DEFENSES = {"pad256": pad256, "pad512": pad512, "jitter200": jitter200, "const": const,
            "nocomp": nocomp, "nocomp_human": nocomp_human}
