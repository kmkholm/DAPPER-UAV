"""Role dispatcher for every Docker node of the DAPPER GNS3 testbeds.

Roles: router (core / cell site), drone (ArduPilot SITL + C2 forwarder), gcs (fleet
controller), bg (cross-traffic user equipment). Configuration comes from /tmp/cfg.json
(written by run.sh from the CFG environment variable that topology.py sets).
"""
import json, os, random, socket, subprocess, sys, threading, time

CFG = json.load(open(sys.argv[1]))
ROLE = CFG["role"]
LOG = lambda *a: print(time.strftime("%H:%M:%S"), CFG.get("name", "?"), *a, flush=True)


def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        LOG("CMD FAIL", cmd, r.stderr.strip()[:300])
        if check:
            raise RuntimeError(cmd)
    return r


def wait_ifaces():
    for itf in CFG.get("ifaces", []):
        for _ in range(120):
            if os.path.exists("/sys/class/net/%s" % itf["dev"]):
                break
            time.sleep(1)


def setup_ifaces():
    for itf in CFG.get("ifaces", []):
        d = itf["dev"]
        sh("ip addr flush dev %s" % d)
        sh("ip addr add %s dev %s" % (itf["addr"], d), check=True)
        sh("ip link set %s up" % d)
        sh("ethtool -K %s tso off gso off gro off 2>/dev/null" % d)
        if itf.get("netem"):
            sh("tc qdisc replace dev %s root netem %s" % (d, itf["netem"]), check=True)
    for dst, gw in CFG.get("routes", []):
        sh("ip route replace %s via %s" % (dst, gw), check=True)
    if CFG.get("forward"):
        sh("sysctl -w net.ipv4.ip_forward=1")
    sh("sysctl -w net.ipv4.conf.all.rp_filter=0")
    sh("sysctl -w net.ipv4.conf.default.rp_filter=0")


def spawn(name, cmd, cwd=None, restart=True, delay=3):
    """Run cmd in the background, restarting it when it exits."""
    def loop():
        n = 0
        while True:
            with open("/data/%s.log" % name, "a") as lf:
                lf.write("\n=== start %d %s\n" % (n, time.strftime("%F %T")))
                lf.flush()
                p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
                rc = p.wait()
                lf.write("=== exit rc=%s %s\n" % (rc, time.strftime("%F %T")))
            n += 1
            if not restart:
                return
            time.sleep(delay)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------- tunnels
def wg_setup(t):
    os.makedirs("/etc/wireguard", exist_ok=True)       # the host AppArmor profile of wg only allows this path
    open("/etc/wireguard/wg0.key", "w").write(t["priv"] + "\n")
    sh("ip link add wg0 type wireguard", check=True)
    cmd = "wg set wg0 private-key /etc/wireguard/wg0.key"
    if t.get("listen"):
        cmd += " listen-port %d" % t["listen"]
    sh(cmd, check=True)
    for p in t["peers"]:
        c = "wg set wg0 peer %s allowed-ips %s" % (p["pub"], p["allowed"])
        if p.get("endpoint"):
            c += " endpoint %s" % p["endpoint"]
        if p.get("keepalive"):
            c += " persistent-keepalive %d" % p["keepalive"]
        sh(c, check=True)
    sh("ip addr add %s dev wg0" % t["addr"], check=True)
    sh("ip link set wg0 up", check=True)


def ipsec_setup(t):
    sh("ip link add dummy0 type dummy", check=True)
    sh("ip addr add %s/32 dev dummy0" % t["inner"], check=True)
    sh("ip link set dummy0 up")
    for i, p in enumerate(t["peers"]):
        rq = 100 + i
        alg = "aead 'rfc4106(gcm(aes))' %s 128"
        sh("ip xfrm state add src %s dst %s proto esp spi %s reqid %d mode tunnel %s"
           % (t["outer"], p["outer"], p["spi_out"], rq, alg % p["key_out"]), check=True)
        sh("ip xfrm state add src %s dst %s proto esp spi %s reqid %d mode tunnel %s"
           % (p["outer"], t["outer"], p["spi_in"], rq, alg % p["key_in"]), check=True)
        sh("ip xfrm policy add src %s/32 dst %s/32 dir out tmpl src %s dst %s proto esp reqid %d mode tunnel"
           % (t["inner"], p["inner"], t["outer"], p["outer"], rq), check=True)
        sh("ip xfrm policy add src %s/32 dst %s/32 dir in tmpl src %s dst %s proto esp reqid %d mode tunnel"
           % (p["inner"], t["inner"], p["outer"], t["outer"], rq), check=True)


def tunnel_setup():
    t = CFG.get("tunnel")
    if not t:
        return
    for _ in range(5):
        try:
            (wg_setup if t["type"] == "wg" else ipsec_setup)(t)
            LOG("tunnel up", t["type"])
            return
        except RuntimeError:
            sh("ip link del wg0; ip link del dummy0; ip xfrm state flush; ip xfrm policy flush")
            time.sleep(3)
    LOG("TUNNEL FAILED")


# ---------------------------------------------------------------- dynamic links
def dyn_netem(d):
    """LEO-style backhaul: the propagation delay jumps every `period` s and each
    satellite hand-over drops packets for 60-150 ms."""
    rng = random.Random(d.get("seed", 7))
    while True:
        lo, hi = d["delay_range"]
        dl = rng.uniform(lo, hi)
        for dev, tpl in d["devs"].items():
            sh("tc qdisc replace dev %s root netem %s" % (dev, tpl.format(delay="%.1fms" % dl, loss="100%")))
        time.sleep(rng.uniform(0.06, 0.15))
        for dev, tpl in d["devs"].items():
            sh("tc qdisc replace dev %s root netem %s" % (dev, tpl.format(delay="%.1fms" % dl, loss=d["loss"])))
        time.sleep(d["period"] + rng.uniform(-1.0, 1.0))


# ---------------------------------------------------------------- sinks on the GCS
def udp_sink(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, port))
    while True:
        try:
            data, addr = s.recvfrom(65535)
            if data[:4] == b"NTPQ" or data[:4] == b"PING":
                s.sendto(data[:4] + os.urandom(44 if data[:4] == b"NTPQ" else 60), addr)
        except OSError:
            time.sleep(0.1)


def tcp_sink(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, port)); s.listen(64)

    def serve(c):
        try:
            c.settimeout(120)
            head = c.recv(64)
            if head.startswith(b"GET"):          # update check: small request, medium reply
                c.sendall(os.urandom(random.randint(2000, 9000)))
            else:
                while c.recv(65536):
                    pass
                c.sendall(b"OK")
        except OSError:
            pass
        finally:
            c.close()
    while True:
        c, _ = s.accept()
        threading.Thread(target=serve, args=(c,), daemon=True).start()


# ---------------------------------------------------------------- cross-traffic sink (router)
def bg_udp_sink(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    while True:
        try:
            data, addr = s.recvfrom(65535)
            if data[:7] == b"DOWNREQ":            # downlink request: stream a burst back
                for _ in range(8):
                    s.sendto(os.urandom(random.randint(400, 1400)), addr)
        except OSError:
            time.sleep(0.1)


def bg_tcp_sink(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port)); s.listen(32)

    def serve(c):
        try:
            c.settimeout(60)
            while c.recv(65536):
                pass
            c.sendall(b"OK")
        except OSError:
            pass
        finally:
            c.close()
    while True:
        try:
            c, _ = s.accept()
            threading.Thread(target=serve, args=(c,), daemon=True).start()
        except OSError:
            time.sleep(0.1)


# ---------------------------------------------------------------- main
def main():
    LOG("role", ROLE)
    wait_ifaces()
    setup_ifaces()
    tunnel_setup()
    if ROLE == "router":
        for cap in CFG.get("captures", []):
            spawn("tcpdump_" + cap["tag"],
                  "tcpdump -i %s -n -s 128 -U -Z root -G 3600 -w '/data/cap_%s_%%Y%%m%%d_%%H%%M%%S.pcap'" % (cap["dev"], cap["tag"]))
        for p in CFG.get("iperf_ports", []):
            threading.Thread(target=bg_udp_sink, args=(p,), daemon=True).start()
            threading.Thread(target=bg_tcp_sink, args=(p + 1000,), daemon=True).start()
        if CFG.get("dyn"):
            threading.Thread(target=dyn_netem, args=(CFG["dyn"],), daemon=True).start()
    elif ROLE == "drone":
        os.makedirs("/tmp/sitl", exist_ok=True)
        with open("/tmp/extra.parm", "w") as f:
            for k, v in CFG["params"].items():
                f.write("%s %s\n" % (k, v))
        if CFG["vehicle"] == "copter":
            sitl = "/app/bin/arducopter --model quad --speedup 1 --defaults /app/params/copter.parm,/tmp/extra.parm"
        else:
            sitl = "/app/bin/arduplane --model quadplane --speedup 1 --defaults /app/params/quadplane.parm,/tmp/extra.parm"
        sitl += " --home %s -I0 -w --sysid %d" % (CFG["home"], CFG["sysid"])
        spawn("sitl", sitl, cwd="/tmp/sitl")
        time.sleep(2)
        spawn("uplink", "python3 -u /app/uplink.py /tmp/cfg.json")
    elif ROLE == "gcs":
        inner = CFG["tunnel"]["addr"].split("/")[0] if CFG["tunnel"]["type"] == "wg" else CFG["tunnel"]["inner"]
        threading.Thread(target=udp_sink, args=(inner, 9000), daemon=True).start()
        threading.Thread(target=tcp_sink, args=(inner, 9001), daemon=True).start()
        spawn("gcs", "python3 -u /app/gcs.py /tmp/cfg.json")
    elif ROLE == "bg":
        spawn("bg", "python3 -u /app/bg.py /tmp/cfg.json")
    LOG("setup done")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
