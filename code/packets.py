"""Per-drone packet arrays (timestamp, outer IP length, direction) from the cell-site captures, for
the packet-level tokens of the encoder.  Writes <out>/<site>_<sysid>_pk.npz next to the binned files.

  python packets.py <raw_dir> <out_dir>
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process import read_pcaps, drone_ip


def main(raw, out):
    meta = json.load(open(os.path.join(raw, "meta.json")))
    for d in meta["drones"]:
        net, site, sysid = d["net"], d["site"], d["sysid"]
        caps = glob.glob(os.path.join(raw, "cell-%s" % site, "data", "cap_%s_%d_*.pcap" % (site, sysid)))
        if not caps:
            print("missing", site, sysid); continue
        ts, ln, up = read_pcaps(caps, drone_ip(net, site, sysid))
        np.savez_compressed(os.path.join(out, "%s_%d_pk.npz" % (site, sysid)), ts=ts, ln=ln.astype(np.int16), up=up)
        print(site, sysid, len(ts), "packets", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
