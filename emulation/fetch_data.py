"""Pull every node's /data volume from the GNS3 VM to the local raw directory, and
write meta.json describing the drones. Safe to run repeatedly while the testbed runs:
the router packet captures and the drone logs keep growing, so later fetches get more.

  python fetch_data.py                 # sync once into ../data/raw
"""
import json, os, sys, tarfile, io
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vm_ssh

RAW = os.path.abspath(os.path.join(HERE, "..", "data", "raw"))
IDS = json.load(open(os.path.join(HERE, "project_ids.json")))
PID = IDS["project_id"]
VMBASE = "/opt/gns3/projects/%s/project-files/docker" % PID


def node_dir(node_id):
    return "%s/%s/data" % (VMBASE, node_id)


def fetch(node_name, node_id):
    dst = os.path.join(RAW, node_name, "data")
    os.makedirs(dst, exist_ok=True)
    remote = node_dir(node_id)
    # tar the remote data dir to stdout, extract locally (one SSH round trip, handles many files)
    c = vm_ssh.connect()
    _, out, err = c.exec_command("sudo -n tar -C '%s' -cf - . 2>/dev/null" % remote, timeout=1800)
    buf = out.read()
    c.close()
    if not buf:
        return 0
    with tarfile.open(fileobj=io.BytesIO(buf)) as t:
        t.extractall(dst)
    return sum(len(f.name) > 0 for f in [])  # placeholder


def main():
    os.makedirs(RAW, exist_ok=True)
    nodes = IDS["nodes"]
    total = 0
    for name, nid in nodes.items():
        try:
            fetch(name, nid)
            n = len(os.listdir(os.path.join(RAW, name, "data")))
            total += n
            print("%-10s %d files" % (name, n))
        except Exception as e:
            print("FAIL", name, str(e)[:120])
    # meta.json: drone descriptors for process.py
    drones = []
    for name in nodes:
        if "-d" in name and not name.startswith(("bg", "core", "cell", "gcs")):
            site = name.split("-")[0]
            j = int(name.split("-d")[1])
            k = int(site[1])
            sysid = 10 * k + j
            net = site[0]
            veh = "copter" if net == "A" else "quadplane"
            drones.append(dict(node=name, net=net, site=site, sysid=sysid, vehicle=veh))
    json.dump(dict(project=PID, drones=drones), open(os.path.join(RAW, "meta.json"), "w"), indent=1)
    print("meta.json:", len(drones), "drones; total files", total)


if __name__ == "__main__":
    main()
