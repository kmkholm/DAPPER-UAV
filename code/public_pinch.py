"""Convert the public PiNcH drone-traffic dataset (Sciancalepore et al., Computer Networks 2020; a real 3DR Solo
running ArduCopter, encrypted Wi-Fi between drone and remote controller, passively captured) into the same per-drone
files our pipeline reads, so DAPPER can be evaluated on real over-the-air traffic.

Source files (release/):
  static_movement/movement_static_flow_features.txt      columns: inter-arrival [s], size [B], label
      label 0 = moving, drone->RC   1 = moving, RC->drone   2 = steady, drone->RC   3 = steady, RC->drone
      (first ~601 s of the capture are the moving state, the second ~601 s the steady state)
  movement_discrimination/filtered_movements_all_flow_features.txt   label 1..7 = pitch down, pitch up, roll left,
      roll right, throttle up, yaw left, yaw right (about 2 min each, both directions merged)

Two "networks" are written:  P = steady / moving (2 classes),  Q = the seven stick movements (7 classes).
Time is cut into 30 s blocks ("missions"); block ids are chosen so that the hash split of data.py puts, per class,
the first 70 % of the blocks in training, the next 15 % in validation and the last 15 % in test.

  python public_pinch.py ../data/public_pinch/release ../data/public_pinch/proc
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process import bin_traffic, BIN

BLOCK = 30.0


def split_of(sysid, mi):
    h = ((sysid * 1_000_003 + mi) * 2_654_435_761) % 100
    return 0 if h < 70 else 1 if h < 85 else 2


def block_ids(n_blocks, sysid, start_id=1):
    """Consecutive block ids whose hash splits give train/val/test in time order (70/15/15 of the blocks)."""
    n_te = max(1, int(round(0.15 * n_blocks))); n_va = max(1, int(round(0.15 * n_blocks))) if n_blocks >= 3 else 0
    want = [0] * (n_blocks - n_va - n_te) + [1] * n_va + [2] * n_te
    ids, nxt = [], start_id
    for w in want:
        while split_of(sysid, nxt) != w:
            nxt += 1
        ids.append(nxt); nxt += 1
    return ids


def convert(iat, size, lab_pkt, up, y_of_pkt, nclass, net, site, sysid, out, block=BLOCK):
    ts = np.cumsum(iat); t0 = 0.0; nbin = int(ts[-1] / BIN)
    X = bin_traffic(ts, size.astype(np.int32), up, t0, nbin)
    # per-bin label: majority of the packets in the bin, else the label of the nearest packet in time
    bi = np.clip((ts / BIN).astype(int), 0, nbin - 1)
    y = np.full(nbin, -1, np.int8)
    for b in range(nbin):
        pass
    counts = np.zeros((nbin, nclass), int); np.add.at(counts, (bi, y_of_pkt), 1)
    has = counts.sum(1) > 0
    y[has] = counts[has].argmax(1)
    idx = np.flatnonzero(has); tb = (np.arange(nbin) + 0.5) * BIN
    near = idx[np.clip(np.searchsorted(tb[idx], tb[~has]), 0, len(idx) - 1)]
    y[~has] = y[near]
    # 30 s blocks per class segment, ids chosen for a time-ordered split within each class
    mission = np.zeros(nbin, np.int32); nxt = 1
    for c in range(nclass):
        seg = np.flatnonzero(y == c)
        if len(seg) == 0:
            continue
        nb = max(1, int(np.ceil(len(seg) * BIN / block)))
        ids = block_ids(nb, sysid, nxt); nxt = ids[-1] + 1
        for k, j in enumerate(np.array_split(seg, nb)):
            mission[j] = ids[k]
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "%s_%d.npz" % (site, sysid)), X=X.astype(np.float32), y=y, mission=mission,
                        t0=t0, site=site, net=net, sysid=sysid, vehicle="3dr-solo")
    np.savez_compressed(os.path.join(out, "%s_%d_pk.npz" % (site, sysid)), ts=ts, ln=size.astype(np.int16), up=up)
    print(net, site, "bins", nbin, "labels", np.bincount(y[y >= 0], minlength=nclass).tolist(), "blocks", int(mission.max()))


def main(src, out):
    a = np.loadtxt(os.path.join(src, "static_movement", "movement_static_flow_features.txt"))
    lab = a[:, 2].astype(int)
    up = np.isin(lab, (0, 2))                              # drone -> RC = uplink
    y = np.where(lab >= 2, 0, 1)                           # 0 steady, 1 moving
    convert(a[:, 0], a[:, 1], lab, up, y, 2, "P", "P1", 1, out)
    b = np.loadtxt(os.path.join(src, "movement_discrimination", "filtered_movements_all_flow_features.txt"))
    lab = b[:, 2].astype(int) - 1
    convert(b[:, 0], b[:, 1], lab, np.ones(len(b), bool), lab, 7, "Q", "Q1", 1, out, block=15.0)   # 2 min per movement: 15 s blocks, 6.4 s windows


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
