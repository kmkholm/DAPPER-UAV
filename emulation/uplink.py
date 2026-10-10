"""Drone-side C2 forwarder, ground-truth logger and companion applications.

SITL (TCP 5760)  <->  this process  <->  UDP over the tunnel  <->  GCS
Every MAVLink frame produced by the autopilot is forwarded as its own UDP datagram
(batch_ms = 0) or aggregated for batch_ms milliseconds (operator-specific C2 stack).
The forwarder also decodes the stream it relays to log the vehicle state at 5 Hz;
that log is the ground truth for flight-phase labels. Companion applications
(network Remote-ID reports, fleet heartbeats, time sync, post-flight log upload,
software-update checks) share the same tunnel, as on a real companion computer.
"""
import json, os, random, socket, sys, threading, time
os.environ["MAVLINK20"] = "1"
from pymavlink.dialects.v20 import ardupilotmega as mav

CFG = json.load(open(sys.argv[1]))
SYSID = CFG["sysid"]
INNER, GCS_IN, GCS_PORT = CFG["inner"], CFG["gcs_inner"], CFG["gcs_port"]
BATCH = CFG.get("batch_ms", 0) / 1000.0
RNG = random.Random(SYSID * 7919 + int(time.time()))

state = dict(armed=0, mode=-1, alt=0.0, vx=0.0, vy=0.0, vz=0.0, gs=0.0, aspd=0.0, climb=0.0,
             seq=-1, last_override=0.0, last_setpoint=0.0, lat=0, lon=0, hdg=0, throttle=0)
mission_cmds = {}          # seq -> MAV_CMD of the uploaded mission (seen on the downlink)
lock = threading.Lock()
flight_t0 = [None]


def connect_sitl():
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", 5760), timeout=5)
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError:
            time.sleep(1)


udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for _ in range(60):
    try:
        udp.bind((INNER, 14550)); break
    except OSError:
        time.sleep(1)
DEST = (GCS_IN, GCS_PORT)


def note_uplink(m):
    t = m.get_type()
    with lock:
        if t == "HEARTBEAT" and m.get_srcComponent() == 1:
            state["armed"] = 1 if m.base_mode & 128 else 0
            state["mode"] = m.custom_mode
        elif t == "GLOBAL_POSITION_INT":
            state.update(alt=m.relative_alt / 1000.0, vx=m.vx / 100.0, vy=m.vy / 100.0, vz=m.vz / 100.0,
                         lat=m.lat, lon=m.lon, hdg=m.hdg)
        elif t == "VFR_HUD":
            state.update(gs=m.groundspeed, aspd=m.airspeed, climb=m.climb, throttle=m.throttle)
        elif t == "MISSION_CURRENT":
            state["seq"] = m.seq


def note_downlink(m):
    t = m.get_type()
    now = time.time()
    with lock:
        if t == "RC_CHANNELS_OVERRIDE" and any(v not in (0, 65535) for v in (m.chan1_raw, m.chan2_raw, m.chan3_raw)):
            state["last_override"] = now
        elif t in ("SET_POSITION_TARGET_LOCAL_NED",) and (m.type_mask & 0b111000) == 0:
            state["last_setpoint"] = now          # velocity set-point stream = remote piloting
        elif t == "MISSION_ITEM_INT":
            mission_cmds[m.seq] = m.command
        elif t == "MISSION_COUNT":
            mission_cmds.clear()


def sitl_to_gcs(sitl):
    parser = mav.MAVLink(None)
    parser.robust_parsing = True
    buf, t_first = [], None
    while True:
        try:
            data = sitl.recv(4096)
        except OSError:
            data = b""
        if not data:
            raise ConnectionError("sitl closed")
        try:
            msgs = parser.parse_buffer(data) or []
        except Exception:
            msgs = []
        for m in msgs:
            if m.get_type() == "BAD_DATA":
                continue
            note_uplink(m)
            raw = m.get_msgbuf()
            if BATCH <= 0:
                udp.sendto(raw, DEST)
            else:
                if not buf:
                    t_first = time.time()
                buf.append(raw)
                if time.time() - t_first >= BATCH or sum(map(len, buf)) > 1100:
                    udp.sendto(b"".join(buf), DEST); buf = []
        if buf and time.time() - t_first >= BATCH:
            udp.sendto(b"".join(buf), DEST); buf = []


def gcs_to_sitl(sitl):
    parser = mav.MAVLink(None)
    parser.robust_parsing = True
    while True:
        data, _ = udp.recvfrom(65535)
        try:
            for m in parser.parse_buffer(data) or []:
                if m.get_type() != "BAD_DATA":
                    note_downlink(m)
        except Exception:
            pass
        try:
            sitl.sendall(data)
        except OSError:
            return


def gt_logger():
    path = "/data/gt_%d.csv" % SYSID
    new = not os.path.exists(path)
    f = open(path, "a", buffering=1)
    if new:
        f.write("t,armed,mode,alt,vx,vy,vz,gs,aspd,climb,seq,cmd,override,setpoint,lat,lon,throttle\n")
    n_status = 0
    while True:
        now = time.time()
        with lock:
            s = dict(state)
            cmd = mission_cmds.get(s["seq"], -1)
        n_status += 1
        if n_status % 25 == 0:          # every 5 s: one human-readable status line on the node console
            print("%s sysid=%d armed=%d mode=%d alt=%.1fm gs=%.1fm/s climb=%.1fm/s seq=%d cmd=%d %s" % (
                time.strftime("%H:%M:%S"), SYSID, s["armed"], s["mode"], s["alt"], s["gs"], s["climb"], s["seq"], cmd,
                "TELEOP" if (now - s["last_override"] < 1.0 or now - s["last_setpoint"] < 1.0) else "auto"), flush=True)
        f.write("%.3f,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%d,%d,%d\n" % (
            now, s["armed"], s["mode"], s["alt"], s["vx"], s["vy"], s["vz"], s["gs"], s["aspd"], s["climb"],
            s["seq"], cmd, int(now - s["last_override"] < 1.0), int(now - s["last_setpoint"] < 1.0),
            s["lat"], s["lon"], s["throttle"]))
        time.sleep(0.2 - (time.time() - now) % 0.2)


# ---------------------------------------------------------------- companion applications
def remote_id():
    """Network Remote-ID / UTM position report, 1 Hz while armed."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind((INNER, 0))
    while True:
        with lock:
            st = dict(state)
        if st["armed"]:
            rep = {"id": "UAS-%05d" % SYSID, "t": round(time.time(), 2), "lat": st["lat"], "lon": st["lon"],
                   "alt": round(st["alt"], 1), "vx": st["vx"], "vy": st["vy"], "vz": st["vz"], "hdg": st["hdg"],
                   "status": "airborne" if st["alt"] > 1 else "ground", "op": CFG.get("site", "")}
            s.sendto(b"RID1" + json.dumps(rep).encode(), (GCS_IN, 9000))
        time.sleep(1.0)


def fleet_heartbeat():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind((INNER, 0))
    while True:
        s.sendto(b"PING" + os.urandom(RNG.randint(80, 140)), (GCS_IN, 9000))
        time.sleep(10 + RNG.uniform(-1, 1))


def time_sync():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind((INNER, 0))
    while True:
        s.sendto(b"NTPQ" + os.urandom(44), (GCS_IN, 9000))
        time.sleep(64 + RNG.uniform(-2, 2))


def tcp_send(nbytes, get=False):
    try:
        c = socket.create_connection((GCS_IN, 9001), timeout=30, source_address=(INNER, 0))
        if get:
            c.sendall(b"GET /fw/latest HTTP/1.1\r\n\r\n")
            while c.recv(65536):
                pass
        else:
            sent = 0
            while sent < nbytes:
                chunk = os.urandom(min(16384, nbytes - sent))
                c.sendall(chunk); sent += len(chunk)
            c.shutdown(socket.SHUT_WR)
            c.recv(16)
        c.close()
    except OSError:
        pass


def log_upload():
    """After each landing the companion uploads the flight log (size grows with flight time)."""
    was_armed = 0
    while True:
        with lock:
            a = state["armed"]
        if a and not was_armed:
            flight_t0[0] = time.time()
        if was_armed and not a and flight_t0[0]:
            dur = time.time() - flight_t0[0]
            if RNG.random() < 0.6:
                time.sleep(RNG.uniform(3, 20))
                tcp_send(int(dur * RNG.uniform(600, 1400)))
        was_armed = a
        time.sleep(0.5)


def update_check():
    while True:
        time.sleep(RNG.uniform(900, 2400))
        tcp_send(0, get=True)


def main():
    for f in (gt_logger, remote_id, fleet_heartbeat, time_sync, log_upload, update_check):
        threading.Thread(target=f, daemon=True).start()
    while True:
        sitl = connect_sitl()
        print(time.strftime("%T"), "connected to SITL", flush=True)
        t = threading.Thread(target=gcs_to_sitl, args=(sitl,), daemon=True); t.start()
        try:
            sitl_to_gcs(sitl)
        except Exception as e:
            print(time.strftime("%T"), "sitl link lost:", e, flush=True)
        try:
            sitl.close()
        except OSError:
            pass
        time.sleep(1)


if __name__ == "__main__":
    main()
