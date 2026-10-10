"""Build the DAPPER GNS3 project through the GNS3 REST API.

Two operator networks are emulated side by side on the GNS3 VM:
  SkyNet-A : 4 cell sites x 3 ArduCopter drones, MAVLink 2 over WireGuard
  SkyNet-B : 4 cell sites x 3 ArduPlane QuadPlane VTOLs, MAVLink 2 over IPsec ESP (AES-GCM-128)
Every cell-site router captures the traffic of each attached drone with tcpdump.

  python topology.py --smoke      # 1 site x 1 drone per network (pipeline test)
  python topology.py              # full testbed (24 drones)
  python topology.py --start      # (re)start all nodes of the existing project
"""
import argparse, base64, json, os, random, secrets, sys, time
import requests
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization

HERE = os.path.dirname(os.path.abspath(__file__))
INI = os.path.join(os.environ["APPDATA"], "GNS3", "2.2", "gns3_server.ini")
PW = [l.split("=", 1)[1].strip() for l in open(INI) if l.startswith("password")][0]
API = "http://localhost:3080/v2"
AUTH = ("admin", PW)
PROJECT = "DAPPER-SkyNet"
IMAGE = "dapper-node:latest"
rng = random.Random(20261010)

STREAMS = {  # ArduPilot data-stream ids: 1 raw sensors, 2 ext status, 3 rc, 6 position, 10 extra1, 11 extra2, 12 extra3
    "high": {1: 2, 2: 2, 3: 2, 6: 10, 10: 10, 11: 10, 12: 3},
    "standard": {1: 2, 2: 2, 3: 2, 6: 3, 10: 4, 11: 4, 12: 2},
    "low": {1: 0, 2: 1, 3: 0, 6: 2, 10: 2, 11: 2, 12: 1},
}
NE = "distribution paretonormal"
SITES = {
    "A": dict(vehicle="copter", tunnel="wg", sites=[
        dict(tag="A1", kind="urban-5G", up="delay 6ms 2ms %s loss 0.05%% rate 50mbit" % NE, down="delay 6ms 2ms %s loss 0.05%% rate 200mbit" % NE,
             bh="delay 3ms 0.5ms rate 200mbit limit 2000", load=dict(on_mean=15, off_mean=30, rate_mbit=[2, 20]),
             streams="high", batch_ms=0, teleop_hz=10, mix=dict(delivery=.2, teleop=.35, survey=.15, inspection=.3),
             home=(31.9539, 35.9106, 780)),
        dict(tag="A2", kind="suburban-LTE", up="delay 22ms 6ms %s loss 0.3%% rate 20mbit" % NE, down="delay 22ms 6ms %s loss 0.3%% rate 60mbit" % NE,
             bh="delay 6ms 1ms rate 50mbit limit 1000", load=dict(on_mean=20, off_mean=20, rate_mbit=[2, 25]),
             streams="standard", batch_ms=0, teleop_hz=5, mix=dict(survey=.25, delivery=.15, inspection=.3, teleop=.3),
             home=(32.5556, 35.8500, 620)),
        dict(tag="A3", kind="rural-LTE", up="delay 40ms 15ms %s loss 1%% rate 5mbit" % NE, down="delay 40ms 15ms %s loss 1%% rate 15mbit" % NE,
             bh="delay 12ms 2ms rate 20mbit limit 1000", load=dict(on_mean=10, off_mean=40, rate_mbit=[1, 8]),
             streams="low", batch_ms=50, teleop_hz=5, mix=dict(inspection=.35, survey=.2, delivery=.15, teleop=.3),
             home=(32.3326, 35.7517, 1200)),
        dict(tag="A4", kind="congested-LTE", up="delay 28ms 10ms %s loss 0.5%% rate 10mbit" % NE, down="delay 28ms 10ms %s loss 0.5%% rate 30mbit" % NE,
             bh="delay 8ms 2ms rate 15mbit limit 1000", load=dict(on_mean=30, off_mean=10, rate_mbit=[5, 14]),
             streams="standard", batch_ms=20, teleop_hz=10, mix=dict(delivery=.15, teleop=.35, survey=.2, inspection=.3),
             home=(31.9900, 35.8700, 900)),
    ]),
    "B": dict(vehicle="quadplane", tunnel="ipsec", sites=[
        dict(tag="B1", kind="5G-SA", up="delay 5ms 1.5ms %s loss 0.05%% rate 50mbit" % NE, down="delay 5ms 1.5ms %s loss 0.05%% rate 200mbit" % NE,
             bh="delay 3ms 0.5ms rate 200mbit limit 2000", load=dict(on_mean=15, off_mean=30, rate_mbit=[2, 20]),
             streams="high", batch_ms=0, teleop_hz=10, mix=dict(teleop=.35, patrol=.2, survey=.15, inspection=.3),
             home=(-35.3632, 149.1652, 584)),
        dict(tag="B2", kind="LTE", up="delay 25ms 8ms %s loss 0.4%% rate 20mbit" % NE, down="delay 25ms 8ms %s loss 0.4%% rate 60mbit" % NE,
             bh="delay 8ms 1ms rate 50mbit limit 1000", load=dict(on_mean=20, off_mean=20, rate_mbit=[2, 25]),
             streams="standard", batch_ms=0, teleop_hz=5, mix=dict(survey=.2, patrol=.2, inspection=.3, teleop=.3),
             home=(40.0726, -105.2299, 1580)),
        dict(tag="B3", kind="LEO-satellite", up="delay 15ms 4ms %s loss 0.2%% rate 30mbit" % NE, down="delay 15ms 4ms %s loss 0.2%% rate 100mbit" % NE,
             bh="delay {delay} 3ms loss {loss} rate 100mbit limit 2000", dyn=dict(delay_range=[20, 55], period=15, loss="0.3%"),
             load=dict(on_mean=10, off_mean=40, rate_mbit=[1, 10]),
             streams="low", batch_ms=40, teleop_hz=5, mix=dict(patrol=.2, inspection=.35, survey=.15, teleop=.3),
             home=(-27.4705, 153.0260, 20)),
        dict(tag="B4", kind="GEO-satellite", up="delay 20ms 5ms %s loss 0.3%% rate 20mbit" % NE, down="delay 20ms 5ms %s loss 0.3%% rate 60mbit" % NE,
             bh="delay 270ms 5ms loss 0.2% rate 10mbit limit 2000", load=dict(on_mean=15, off_mean=30, rate_mbit=[1, 6]),
             streams="low", batch_ms=100, teleop_hz=5, mix=dict(inspection=.35, survey=.2, patrol=.15, teleop=.3),
             home=(48.0000, 11.0000, 600)),
    ]),
}


def wg_keys():
    k = X25519PrivateKey.generate()
    priv = base64.b64encode(k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                            serialization.NoEncryption())).decode()
    pub = base64.b64encode(k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
    return priv, pub


def api(method, path, **kw):
    r = requests.request(method, API + path, auth=AUTH, timeout=120, **kw)
    if r.status_code >= 400:
        raise RuntimeError("%s %s -> %d %s" % (method, path, r.status_code, r.text[:400]))
    return r.json() if r.text else None


def b64(cfg):
    return base64.b64encode(json.dumps(cfg, separators=(",", ":")).encode()).decode()


def build(smoke):
    nodes, links = [], []                  # nodes: (name, cfg, adapters, symbol, x, y)
    n_sites, n_drones = (1, 1) if smoke else (4, 3)
    for net_i, (net, spec) in enumerate(SITES.items()):
        o = net_i + 1                      # 10.<o>.x.x core/backhaul, 172.<15+o>.x.x access
        x0 = -1150 + net_i * 1250
        gcs_tun = {"type": spec["tunnel"]}
        gcs_priv, gcs_pub = wg_keys()
        if spec["tunnel"] == "wg":
            gcs_tun.update(priv=gcs_priv, addr="10.100.0.1/24", listen=51820, peers=[])
            gcs_inner = "10.100.0.1"
        else:
            gcs_tun.update(inner="10.200.0.1", outer="10.%d.0.2" % o, peers=[])
            gcs_inner = "10.200.0.1"
        gcs_cfg = dict(role="gcs", name="gcs-%s" % net, ifaces=[dict(dev="eth0", addr="10.%d.0.2/30" % o)],
                       routes=[["default", "10.%d.0.1" % o]], tunnel=gcs_tun, drones=[])
        core_cfg = dict(role="router", name="core-%s" % net, forward=True,
                        ifaces=[dict(dev="eth0", addr="10.%d.0.1/30" % o)], routes=[], iperf_ports=[])
        sites = spec["sites"][:n_sites]
        for k, s in enumerate(sites, start=1):
            core_cfg["ifaces"].append(dict(dev="eth%d" % k, addr="10.%d.%d.1/30" % (o, k),
                                           netem=None if "{delay}" in s["bh"] else s["bh"]))
            core_cfg["routes"].append(["172.%d.%d.0/24" % (15 + o, k), "10.%d.%d.2" % (o, k)])
            core_cfg["iperf_ports"].append(5200 + k)
            cell = dict(role="router", name="cell-%s" % s["tag"], forward=True,
                        ifaces=[dict(dev="eth0", addr="10.%d.%d.2/30" % (o, k), netem=None if "{delay}" in s["bh"] else s["bh"])],
                        routes=[["default", "10.%d.%d.1" % (o, k)]], captures=[])
            if "{delay}" in s["bh"]:
                cell["dyn"] = dict(devs={"eth0": s["bh"]}, delay_range=s["dyn"]["delay_range"], period=s["dyn"]["period"],
                                   loss=s["dyn"]["loss"], seed=k)
                core_cfg.setdefault("dyn", dict(devs={}, delay_range=s["dyn"]["delay_range"], period=s["dyn"]["period"],
                                                loss=s["dyn"]["loss"], seed=100 + k))["devs"]["eth%d" % k] = s["bh"]
            cx = x0 + (k - 1) * 300
            for j in range(1, n_drones + 1):
                sysid = 10 * k + j
                did = "%s-d%d" % (s["tag"], j)
                r_ip, d_ip = "172.%d.%d.%d" % (15 + o, k, 4 * j + 1), "172.%d.%d.%d" % (15 + o, k, 4 * j + 2)
                cell["ifaces"].append(dict(dev="eth%d" % j, addr=r_ip + "/30", netem=s["down"]))
                cell["captures"].append(dict(dev="eth%d" % j, tag="%s_%d" % (s["tag"], sysid)))
                hlat, hlon, halt = s["home"]
                hlat, hlon = hlat + (j - 2) * 0.0003, hlon + (j - 2) * 0.0003
                params = dict(MAV_SYSID=sysid, LOG_BACKEND_TYPE=0, LOG_DISARMED=0, FS_GCS_ENABLE=0,
                              SIM_WIND_SPD=round(rng.uniform(0, 6), 1), SIM_WIND_DIR=rng.randint(0, 359),
                              SIM_WIND_TURB=round(rng.uniform(0, 1.5), 1))
                if spec["vehicle"] == "copter":
                    params.update(FS_THR_ENABLE=0, CAM1_TYPE=1, AUTO_OPTIONS=3, DISARM_DELAY=8)
                else:
                    params.update(THR_FAILSAFE=0, Q_RTL_MODE=1, FS_LONG_ACTN=0, FS_SHORT_ACTN=3, RTL_AUTOLAND=0)
                if spec["tunnel"] == "wg":
                    priv, pub = wg_keys()
                    inner = "10.100.0.%d" % sysid
                    tun = dict(type="wg", priv=priv, addr=inner + "/24",
                               peers=[dict(pub=gcs_pub, allowed="10.100.0.1/32", endpoint="10.%d.0.2:51820" % o, keepalive=25)])
                    gcs_tun["peers"].append(dict(pub=pub, allowed=inner + "/32"))
                else:
                    inner = "10.200.0.%d" % sysid
                    k_up, k_dn = "0x" + secrets.token_hex(20), "0x" + secrets.token_hex(20)
                    s_up, s_dn = "0x%08x" % rng.randint(0x10000000, 0x7fffffff), "0x%08x" % rng.randint(0x10000000, 0x7fffffff)
                    tun = dict(type="ipsec", inner=inner, outer=d_ip,
                               peers=[dict(inner="10.200.0.1", outer="10.%d.0.2" % o, spi_out=s_up, spi_in=s_dn, key_out=k_up, key_in=k_dn)])
                    gcs_tun["peers"].append(dict(inner=inner, outer=d_ip, spi_out=s_dn, spi_in=s_up, key_out=k_dn, key_in=k_up))
                port = 14500 + sysid
                dcfg = dict(role="drone", name=did, site=s["tag"], vehicle=spec["vehicle"], sysid=sysid,
                            ifaces=[dict(dev="eth0", addr=d_ip + "/30", netem=s["up"])], routes=[["default", r_ip]],
                            tunnel=tun, inner=inner, gcs_inner=gcs_inner, gcs_port=port, batch_ms=s["batch_ms"],
                            home="%.6f,%.6f,%d,%d" % (hlat, hlon, halt, rng.randint(0, 359)), params=params)
                gcs_cfg["drones"].append(dict(sysid=sysid, port=port, vehicle=spec["vehicle"], site=s["tag"],
                                              streams=STREAMS[s["streams"]], teleop_hz=s["teleop_hz"], mix=s["mix"]))
                nodes.append((did, dcfg, 1, "drone_" + spec["vehicle"], cx - 100 + (j - 1) * 100, 330))
                links.append((cell["name"], j, did, 0))
            bg_ip, bg_r = "172.%d.%d.18" % (15 + o, k), "172.%d.%d.17" % (15 + o, k)
            cell["ifaces"].append(dict(dev="eth%d" % (n_drones + 1), addr=bg_r + "/30", netem=s["down"]))
            nodes.append(("bg-%s" % s["tag"], dict(role="bg", name="bg-%s" % s["tag"], ifaces=[dict(dev="eth0", addr=bg_ip + "/30", netem=s["up"])],
                                                    routes=[["default", bg_r]], target="10.%d.%d.1" % (o, k), port=5200 + k, load=s["load"]),
                          1, ":/symbols/affinity/circle/gray/phone_wireless.svg", cx + 120, 180))
            links.append((cell["name"], n_drones + 1, "bg-%s" % s["tag"], 0))
            nodes.append((cell["name"], cell, n_drones + 2, ":/symbols/affinity/circle/blue/satellite_dish.svg"
                          if "satellite" in s["kind"] else ":/symbols/affinity/circle/blue/router.svg", cx, 150))
            links.append(("core-%s" % net, k, cell["name"], 0))
        nodes.append(("core-%s" % net, core_cfg, len(sites) + 1, ":/symbols/affinity/circle/green/router_cloud.svg", x0 + 450, -60))
        nodes.append(("gcs-%s" % net, gcs_cfg, 1, ":/symbols/affinity/circle/red/server.svg", x0 + 450, -230))
        links.append(("core-%s" % net, 0, "gcs-%s" % net, 0))
    return nodes, links


DRONE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60">
<g stroke="%s" stroke-width="3" fill="none"><line x1="14" y1="14" x2="46" y2="46"/><line x1="46" y1="14" x2="14" y2="46"/></g>
<g fill="%s" fill-opacity="0.35" stroke="%s" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="48" cy="12" r="9"/>
<circle cx="12" cy="48" r="9"/><circle cx="48" cy="48" r="9"/></g><rect x="22" y="22" width="16" height="16" rx="4" fill="%s"/></svg>"""
VTOL_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60">
<path d="M30 6 L34 24 L56 30 L56 34 L34 32 L32 48 L40 54 L40 56 L20 56 L20 54 L28 48 L26 32 L4 34 L4 30 L26 24 Z" fill="%s" stroke="%s" stroke-width="1.5"/>
<g fill="none" stroke="%s" stroke-width="2"><circle cx="14" cy="22" r="7"/><circle cx="46" cy="22" r="7"/><circle cx="14" cy="42" r="7"/><circle cx="46" cy="42" r="7"/></g></svg>"""


def upload_symbols():
    for sid, svg in (("drone_copter.svg", DRONE_SVG % ("#1f4e79", "#2e86c1", "#1f4e79", "#c0392b")),
                     ("drone_quadplane.svg", VTOL_SVG % ("#7d3c98", "#4a235a", "#4a235a"))):
        requests.post(API + "/symbols/%s/raw" % sid, data=svg.encode(), auth=AUTH, timeout=30)


def create(smoke, clean=True, console="none"):
    for p in (api("GET", "/projects") if clean else []):
        if p["name"].startswith(PROJECT):
            for call in (("POST", "/projects/%s/open"), ("POST", "/projects/%s/nodes/stop"),
                         ("DELETE", "/projects/%s"), ("POST", "/projects/%s/close"), ("DELETE", "/projects/%s")):
                try:
                    api(call[0], call[1] % p["project_id"])
                except RuntimeError as e:
                    print("note:", str(e)[:120])
            time.sleep(3)
    upload_symbols()
    names = {p["name"] for p in api("GET", "/projects")}
    pname = PROJECT if PROJECT not in names else PROJECT + time.strftime("-%m%d-%H%M")
    pid = api("POST", "/projects", json=dict(name=pname, auto_close=False, scene_width=3000, scene_height=1200))["project_id"]
    nodes, links = build(smoke)
    ids = {}
    for name, cfg, nad, sym, x, y in nodes:
        n = api("POST", "/projects/%s/nodes" % pid, json=dict(
            name=name, node_type="docker", compute_id="vm", x=int(x), y=int(y),
            symbol=sym if sym.startswith(":") else sym + ".svg",
            properties=dict(image=IMAGE, adapters=nad, start_command="/app/run.sh", console_type=console,
                            environment="CFG=" + b64(cfg), extra_volumes=["/data"])))
        ids[name] = n["node_id"]
    for a, ia, b, ib in links:
        api("POST", "/projects/%s/links" % pid, json=dict(nodes=[dict(node_id=ids[a], adapter_number=ia, port_number=0),
                                                                 dict(node_id=ids[b], adapter_number=ib, port_number=0)]))
    for net, txt, x in (("A", "SkyNet-A: ArduCopter fleet, MAVLink 2 over WireGuard", -1150),
                        ("B", "SkyNet-B: ArduPlane QuadPlane fleet, MAVLink 2 over IPsec ESP", 100)):
        api("POST", "/projects/%s/drawings" % pid, json=dict(x=x, y=-300, svg='<svg width="700" height="40"><text font-family="Arial" '
            'font-size="22" font-weight="bold" fill="#1b2631">%s</text></svg>' % txt))
    json.dump(dict(project_id=pid, nodes=ids), open(os.path.join(HERE, "project_ids.json"), "w"), indent=1)
    print("project", pid, "nodes", len(ids), "links", len(links))
    return pid


def start(pid=None):
    pid = pid or json.load(open(os.path.join(HERE, "project_ids.json")))["project_id"]
    api("POST", "/projects/%s/open" % pid)
    # routers and GCS first, then drones and background UEs
    nodes = api("GET", "/projects/%s/nodes" % pid)
    order = sorted(nodes, key=lambda n: (0 if n["name"].startswith(("core", "cell", "gcs")) else 1, n["name"]))
    for n in order:
        api("POST", "/projects/%s/nodes/%s/start" % (pid, n["node_id"]))
        print("started", n["name"], flush=True)
    return pid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--no-clean", action="store_true", help="keep existing DAPPER projects")
    ap.add_argument("--telnet", action="store_true", help="telnet consoles (log streaming) for screenshots")
    a = ap.parse_args()
    if a.start:
        start()
    else:
        start(create(a.smoke, clean=not a.no_clean, console="telnet" if a.telnet else "none"))
