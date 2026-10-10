"""Packet-level traffic-shaping defences, applied to the outer (encrypted) packet stream of every drone
before any view (bins, packet tokens, window statistics) is built, so the adversary is retrained on
the defended traffic.  Each transform maps (timestamps, IP lengths, is_uplink) to the same triple.

  pad256 / pad512 : every packet is padded up to the next multiple of 256 / 512 bytes
  jitter200       : every packet is delayed by an independent uniform 0-200 ms (timing obfuscation,
                    no byte overhead)
  const           : constant-rate cover traffic, 20 packets/s of 512 bytes in each direction, real
                    packets absorbed into the stream (sizes and timing both hidden)
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


DEFENSES = {"pad256": pad256, "pad512": pad512, "jitter200": jitter200, "const": const}
