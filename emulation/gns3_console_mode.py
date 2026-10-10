"""Give the existing (stopped) DAPPER project telnet consoles that stream each node's logs,
so the GNS3 GUI / web UI can show live SITL, MAVLink-forwarder and tcpdump output for the
screenshots. Also rebuilds the image with a log-streaming run.sh.

  python gns3_console_mode.py            # patch run.sh, rebuild image, set console_type=telnet
  python topology.py --start             # then start the project and open the web UI
"""
import json, os, sys, requests
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
INI = os.path.join(os.environ["APPDATA"], "GNS3", "2.2", "gns3_server.ini")
PW = [l.split("=", 1)[1].strip() for l in open(INI) if l.startswith("password")][0]
API, AUTH = "http://localhost:3080/v2", ("admin", PW)

RUN_SH = """#!/bin/sh
# GNS3 start command. CFG holds the base64 JSON node configuration written by topology.py.
# The node's logs are streamed to stdout so that the GNS3 telnet console shows them live.
mkdir -p /data
echo "$CFG" | base64 -d > /tmp/cfg.json
python3 -u /app/node.py /tmp/cfg.json >> /data/node.log 2>&1 &
sleep 2
touch /data/node.log
exec tail -n 20 -F /data/node.log /data/sitl.log /data/uplink.log /data/gcs.log /data/gcs_events.csv /data/tcpdump_*.log 2>/dev/null
"""


def main():
    open(os.path.join(HERE, "run.sh"), "w", newline="\n").write(RUN_SH)
    import build_image  # noqa: rebuilds dapper-node:latest with the new run.sh (runs on import)
    pid = json.load(open(os.path.join(HERE, "project_ids.json")))["project_id"]
    nodes = requests.get(API + "/projects/%s/nodes" % pid, auth=AUTH, timeout=60).json()
    n = 0
    for nd in nodes:
        if nd["status"] != "stopped":
            print("skip running node", nd["name"]); continue
        r = requests.put(API + "/projects/%s/nodes/%s" % (pid, nd["node_id"]), auth=AUTH, timeout=60,
                         json={"console_type": "telnet"})
        n += r.status_code == 200
    print("telnet console set on", n, "of", len(nodes), "nodes; project", pid)


if __name__ == "__main__":
    main()
