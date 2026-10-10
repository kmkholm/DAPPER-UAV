"""Fleet ground-control station: one controller thread per drone flies randomised,
operator-specific missions forever and logs mission events.

Copter missions : delivery (GUIDED), survey (AUTO lawn-mower), inspection (AUTO orbit), teleop (GUIDED velocity stream)
QuadPlane missions: survey, patrol, inspection (AUTO, VTOL take-off / RTL VTOL landing), teleop (FBWB + RC override stream)
"""
import json, math, os, queue, random, sys, threading, time
os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil

CFG = json.load(open(sys.argv[1]))
BIND = CFG["tunnel"]["addr"].split("/")[0] if CFG["tunnel"]["type"] == "wg" else CFG["tunnel"]["inner"]
COPTER = dict(STABILIZE=0, ALT_HOLD=2, AUTO=3, GUIDED=4, LOITER=5, RTL=6, CIRCLE=7, LAND=9, BRAKE=17)
PLANE = dict(MANUAL=0, CIRCLE=1, FBWA=5, FBWB=6, CRUISE=7, AUTO=10, RTL=11, LOITER=12, TAKEOFF=13,
             GUIDED=15, QHOVER=18, QLOITER=19, QLAND=20, QRTL=21)
QTYPES = {"COMMAND_ACK", "MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK", "PARAM_VALUE"}
EV_LOCK = threading.Lock()
EV = open("/data/gcs_events.csv", "a", buffering=1)
if EV.tell() == 0:
    EV.write("t,sysid,site,mission_id,event,detail\n")


class MissionError(Exception):
    pass


def offset(lat, lon, dn, de):
    return lat + dn / 111320.0, lon + de / (111320.0 * math.cos(math.radians(lat)))


def dist(a, b):
    dn = (b[0] - a[0]) * 111320.0
    de = (b[1] - a[1]) * 111320.0 * math.cos(math.radians(a[0]))
    return math.hypot(dn, de)


class Drone:
    def __init__(self, d):
        self.d = d
        self.sysid, self.site, self.kind = d["sysid"], d["site"], d["vehicle"]
        self.M = COPTER if self.kind == "copter" else PLANE
        self.rng = random.Random(d["sysid"] * 104729 + int(time.time()))
        self.conn = mavutil.mavlink_connection("udpin:%s:%d" % (BIND, d["port"]), source_system=255,
                                               source_component=190, dialect="ardupilotmega")
        self.st, self.st_t = {}, {}
        self.q = queue.Queue(maxsize=2000)
        self.mid, self.n_mission, self.home = "", 0, None
        threading.Thread(target=self.reader, daemon=True).start()
        threading.Thread(target=self.heartbeat, daemon=True).start()

    # ---------------------------------------------------------- plumbing
    def ev(self, event, detail=""):
        with EV_LOCK:
            EV.write("%.3f,%d,%s,%s,%s,%s\n" % (time.time(), self.sysid, self.site, self.mid, event,
                                                 str(detail).replace(",", ";")))

    def reader(self):
        while True:
            try:
                m = self.conn.recv_match(blocking=True, timeout=1.0)
            except Exception:
                time.sleep(0.2); continue
            if m is None:
                continue
            t = m.get_type()
            if t == "BAD_DATA" or m.get_srcSystem() != self.sysid:
                continue
            self.st[t] = m
            self.st_t[t] = time.time()
            if t in QTYPES:
                try:
                    self.q.put_nowait(m)
                except queue.Full:
                    self.q.get_nowait(); self.q.put_nowait(m)

    def heartbeat(self):
        while True:
            try:
                self.conn.mav.heartbeat_send(6, 8, 0, 0, 0)
            except Exception:
                pass
            time.sleep(1.0)

    def alive(self):
        return time.time() - self.st_t.get("HEARTBEAT", 0) < 45

    @property
    def armed(self):
        hb = self.st.get("HEARTBEAT")
        return bool(hb and hb.base_mode & 128)

    @property
    def mode(self):
        hb = self.st.get("HEARTBEAT")
        return hb.custom_mode if hb else -1

    @property
    def alt(self):
        g = self.st.get("GLOBAL_POSITION_INT")
        return g.relative_alt / 1000.0 if g else 0.0

    @property
    def pos(self):
        g = self.st.get("GLOBAL_POSITION_INT")
        return (g.lat / 1e7, g.lon / 1e7) if g else None

    @property
    def gs(self):
        g = self.st.get("GLOBAL_POSITION_INT")
        return math.hypot(g.vx, g.vy) / 100.0 if g else 0.0

    def wait_for(self, types, cond=lambda m: True, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            try:
                m = self.q.get(timeout=max(0.05, end - time.time()))
            except queue.Empty:
                return None
            if m.get_type() in types and cond(m):
                return m
        return None

    def drain(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def cmd(self, command, p=(0, 0, 0, 0, 0, 0, 0), retries=3):
        for i in range(retries):
            self.conn.mav.command_long_send(self.sysid, 1, command, i, *p)
            m = self.wait_for({"COMMAND_ACK"}, lambda m: m.command == command, 3.0)
            if m is not None and m.result in (0, 5):
                return True
        return False

    def set_mode(self, name):
        mid = self.M[name]
        for _ in range(6):
            self.conn.mav.set_mode_send(self.sysid, 1, mid)
            end = time.time() + 3
            while time.time() < end:
                if self.mode == mid:
                    return True
                time.sleep(0.1)
        raise MissionError("mode %s" % name)

    def set_param(self, name, val):
        for _ in range(3):
            self.conn.mav.param_set_send(self.sysid, 1, name.encode(), float(val), 9)
            if self.wait_for({"PARAM_VALUE"}, lambda m: m.param_id.rstrip("\x00") == name, 3.0):
                return True
        return False

    def request_streams(self):
        for sid, rate in self.d["streams"].items():
            self.conn.mav.request_data_stream_send(self.sysid, 1, int(sid), int(rate), 1 if rate > 0 else 0)

    def arm(self, timeout=360):
        end = time.time() + timeout
        while time.time() < end:
            self.cmd(400, (1, 0, 0, 0, 0, 0, 0), retries=1)
            for _ in range(30):
                if self.armed:
                    self.ev("armed"); return
                time.sleep(0.1)
            time.sleep(4)
        raise MissionError("arm")

    def force_disarm(self):
        self.cmd(400, (0, 21196, 0, 0, 0, 0, 0), retries=3)

    def upload(self, items):
        n = len(items)
        for _ in range(8):
            self.drain()
            self.conn.mav.mission_count_send(self.sysid, 1, n, 0)
            ok, end = False, time.time() + 60
            while time.time() < end:
                m = self.wait_for({"MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"}, timeout=6)
                if m is None:
                    break
                if m.get_type() == "MISSION_ACK":
                    ok = m.type == 0; break
                if m.seq >= n:
                    continue
                it = items[m.seq]
                self.conn.mav.mission_item_int_send(self.sysid, 1, m.seq, it[1], it[0], 0, 1, it[2], it[3], it[4],
                                                    it[5], int(it[6] * 1e7), int(it[7] * 1e7), it[8], 0)
            if ok:
                self.ev("mission_uploaded", n); return
        raise MissionError("upload")

    def wait_until(self, cond, timeout, what):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            if not self.alive():
                raise MissionError("link lost during " + what)
            time.sleep(0.2)
        raise MissionError("timeout " + what)

    def wait_disarmed(self, timeout):
        end, landed_since = time.time() + timeout, None
        while time.time() < end:
            if not self.armed:
                self.ev("disarmed"); return
            if self.alt < 0.6 and self.gs < 0.4:
                landed_since = landed_since or time.time()
                if time.time() - landed_since > 25:
                    self.force_disarm()
            else:
                landed_since = None
            time.sleep(0.5)
        raise MissionError("timeout landing")

    def W(self, lat, lon, alt, cmd=16, p=(0, 0, 0, 0), frame=3):
        return (cmd, frame, p[0], p[1], p[2], p[3], lat, lon, alt)

    # ---------------------------------------------------------- copter missions
    def takeoff(self, alt):
        if not self.cmd(22, (0, 0, 0, 0, 0, 0, alt)):
            raise MissionError("takeoff rejected")
        self.wait_until(lambda: self.alt >= 0.92 * alt, 120, "takeoff")

    def goto(self, tgt, alt, timeout):
        end, last = time.time() + timeout, 0
        while time.time() < end:
            if time.time() - last > 8:
                self.conn.mav.set_position_target_global_int_send(0, self.sysid, 1, 6, 0b0000111111111000,
                                                                  int(tgt[0] * 1e7), int(tgt[1] * 1e7), alt,
                                                                  0, 0, 0, 0, 0, 0, 0, 0)
                last = time.time()
            p = self.pos
            if p and dist(p, tgt) < 4 and abs(self.alt - alt) < 1.5:
                return
            if not self.alive():
                raise MissionError("link lost goto")
            time.sleep(0.2)
        raise MissionError("timeout goto")

    def c_delivery(self):
        r = self.rng
        alt, spd, dd, brg = r.uniform(25, 60), r.uniform(6, 12), r.uniform(300, 1100), r.uniform(0, 2 * math.pi)
        tgt = offset(*self.home, dd * math.cos(brg), dd * math.sin(brg))
        self.ev("mission_start", "delivery alt=%.0f spd=%.1f d=%.0f" % (alt, spd, dd))
        self.set_mode("GUIDED"); self.arm(); self.takeoff(alt)
        self.cmd(178, (1, spd, -1, 0, 0, 0, 0))
        self.goto(tgt, alt, dd / spd * 2.5 + 90)
        time.sleep(r.uniform(5, 25))
        drop = r.uniform(6, 10)
        self.goto(tgt, drop, 120); time.sleep(r.uniform(5, 15)); self.goto(tgt, alt, 120)
        self.set_mode("RTL"); self.wait_disarmed(dd / spd * 2.5 + 300)

    def c_survey(self):
        r = self.rng
        alt, spd = r.uniform(30, 70), r.uniform(5, 10)
        legs, L, sp, brg, off = r.randint(4, 9), r.uniform(150, 380), r.uniform(25, 60), r.uniform(0, 2 * math.pi), r.uniform(60, 250)
        h = self.home
        it = [self.W(h[0], h[1], 0, frame=0), self.W(0, 0, alt, cmd=22), self.W(0, 0, 0, cmd=178, p=(1, spd, -1, 0)),
              self.W(0, 0, 0, cmd=206, p=(r.uniform(15, 30), 0, 1, 0))]
        ux, uy = math.cos(brg), math.sin(brg)
        for k in range(legs):
            a = off + k * sp
            for s in ((0, L) if k % 2 == 0 else (L, 0)):
                n, e = a * ux - s * uy, a * uy + s * ux
                it.append(self.W(*offset(*h, n, e), alt))
        it += [self.W(0, 0, 0, cmd=206, p=(0, 0, 1, 0)), self.W(0, 0, 0, cmd=20)]
        self.ev("mission_start", "survey alt=%.0f spd=%.1f legs=%d L=%.0f" % (alt, spd, legs, L))
        self.upload(it); self.set_mode("GUIDED"); self.arm(); self.set_mode("AUTO")
        self.cmd(300, (0, 0, 0, 0, 0, 0, 0))
        self.wait_disarmed(legs * L / spd * 1.8 + 600)

    def c_inspection(self):
        r = self.rng
        alt, dd, brg, spd = r.uniform(20, 50), r.uniform(150, 600), r.uniform(0, 2 * math.pi), r.uniform(5, 10)
        poi = offset(*self.home, dd * math.cos(brg), dd * math.sin(brg))
        appr = offset(*self.home, (dd - 60) * math.cos(brg), (dd - 60) * math.sin(brg))
        turns, rad, hold = r.randint(2, 5), r.uniform(15, 50), r.uniform(15, 45)
        h = self.home
        it = [self.W(h[0], h[1], 0, frame=0), self.W(0, 0, alt, cmd=22), self.W(0, 0, 0, cmd=178, p=(1, spd, -1, 0)),
              self.W(*appr, alt), self.W(*poi, alt, cmd=18, p=(turns, 0, rad, 0)),
              self.W(*appr, alt, cmd=19, p=(hold, 0, 0, 0)), self.W(0, 0, 0, cmd=20)]
        self.ev("mission_start", "inspection alt=%.0f d=%.0f turns=%d r=%.0f" % (alt, dd, turns, rad))
        self.upload(it); self.set_mode("GUIDED"); self.arm(); self.set_mode("AUTO")
        self.cmd(300, (0, 0, 0, 0, 0, 0, 0))
        self.wait_disarmed(2 * dd / spd + turns * 2 * math.pi * rad / 3 + hold + 600)

    def c_teleop(self):
        r = self.rng
        alt, T, hz = r.uniform(15, 40), r.uniform(60, 180), self.d.get("teleop_hz", 10)
        self.ev("mission_start", "teleop alt=%.0f T=%.0f hz=%d" % (alt, T, hz))
        self.set_mode("GUIDED"); self.arm(); self.takeoff(alt)
        self.ev("teleop_start")
        vn = ve = 0.0
        end = time.time() + T
        while time.time() < end:
            vn += -vn / 3.0 / hz + r.gauss(0, 4.0) * math.sqrt(2 / 3.0 / hz)
            ve += -ve / 3.0 / hz + r.gauss(0, 4.0) * math.sqrt(2 / 3.0 / hz)
            p = self.pos
            if p and dist(p, self.home) > 500:          # steer back toward home
                dn, de = (self.home[0] - p[0]) * 111320, (self.home[1] - p[1]) * 111320 * math.cos(math.radians(p[0]))
                n = math.hypot(dn, de); vn, ve = 4 * dn / n, 4 * de / n
            vn, ve = max(-8, min(8, vn)), max(-8, min(8, ve))
            vd = 0.0 if 10 < self.alt < 60 else (-1.0 if self.alt <= 10 else 1.0)
            self.conn.mav.set_position_target_local_ned_send(0, self.sysid, 1, 1, 0b0000110111000111,
                                                             0, 0, 0, vn, ve, vd, 0, 0, 0, 0, 0)
            if not self.alive():
                raise MissionError("link lost teleop")
            time.sleep(1.0 / hz)
        self.ev("teleop_end")
        self.set_mode("LAND"); self.wait_disarmed(240)

    # ---------------------------------------------------------- quadplane missions
    def p_start(self, it, timeout):
        self.upload(it); self.set_mode("AUTO"); self.arm()
        self.cmd(300, (0, 0, 0, 0, 0, 0, 0), retries=2)
        self.wait_disarmed(timeout)

    def p_head(self, alt, aspd):
        h = self.home
        return [self.W(h[0], h[1], 0, frame=0), self.W(0, 0, alt, cmd=84), self.W(0, 0, 0, cmd=178, p=(0, aspd, -1, 0))]

    def p_survey(self):
        r = self.rng
        alt, aspd = r.uniform(40, 80), r.uniform(16, 22)
        legs, L, sp, brg, off = r.randint(4, 8), r.uniform(400, 900), r.uniform(80, 150), r.uniform(0, 2 * math.pi), r.uniform(200, 500)
        it = self.p_head(alt, aspd)
        ux, uy = math.cos(brg), math.sin(brg)
        for k in range(legs):
            a = off + k * sp
            for s in ((0, L) if k % 2 == 0 else (L, 0)):
                it.append(self.W(*offset(*self.home, a * ux - s * uy, a * uy + s * ux), alt))
        it.append(self.W(0, 0, 0, cmd=20))
        self.ev("mission_start", "survey alt=%.0f aspd=%.1f legs=%d L=%.0f" % (alt, aspd, legs, L))
        self.p_start(it, legs * L / aspd * 2 + 900)

    def p_patrol(self):
        r = self.rng
        alt, aspd, n = r.uniform(50, 100), r.uniform(17, 23), r.randint(3, 6)
        it = self.p_head(alt, aspd)
        b0, tot = r.uniform(0, 2 * math.pi), 0
        for k in range(n):
            dd, b = r.uniform(800, 2000), b0 + k * 2 * math.pi / n
            it.append(self.W(*offset(*self.home, dd * math.cos(b), dd * math.sin(b)), alt)); tot += dd
        it.append(self.W(0, 0, 0, cmd=20))
        self.ev("mission_start", "patrol alt=%.0f aspd=%.1f n=%d" % (alt, aspd, n))
        self.p_start(it, 3 * tot / aspd + 900)

    def p_inspection(self):
        r = self.rng
        alt, aspd, dd, brg = r.uniform(40, 80), r.uniform(16, 22), r.uniform(500, 1500), r.uniform(0, 2 * math.pi)
        turns, rad = r.randint(2, 6), r.uniform(60, 120)
        it = self.p_head(alt, aspd)
        it += [self.W(*offset(*self.home, dd * math.cos(brg), dd * math.sin(brg)), alt, cmd=18, p=(turns, 0, rad, 0)),
               self.W(0, 0, 0, cmd=20)]
        self.ev("mission_start", "inspection alt=%.0f d=%.0f turns=%d r=%.0f" % (alt, dd, turns, rad))
        self.p_start(it, 2 * dd / aspd + turns * 2 * math.pi * rad / aspd + 900)

    def p_teleop(self):
        r = self.rng
        alt, aspd, T, hz = r.uniform(50, 90), r.uniform(16, 22), r.uniform(60, 150), self.d.get("teleop_hz", 10)
        dd, brg = r.uniform(300, 600), r.uniform(0, 2 * math.pi)
        it = self.p_head(alt, aspd)
        it += [self.W(*offset(*self.home, dd * math.cos(brg), dd * math.sin(brg)), alt, cmd=17, p=(0, 0, 90, 0))]
        self.ev("mission_start", "teleop alt=%.0f T=%.0f hz=%d" % (alt, T, hz))
        self.upload(it); self.set_mode("AUTO"); self.arm()
        self.cmd(300, (0, 0, 0, 0, 0, 0, 0), retries=2)
        loiter_seq = len(it) - 1
        self.wait_until(lambda: self.st.get("MISSION_CURRENT") is not None and self.st["MISSION_CURRENT"].seq == loiter_seq
                        and self.alt > 0.8 * alt, 400, "reach loiter")
        self.set_mode("FBWB")
        self.ev("teleop_start")
        ro = pi_ = th = 0.0
        end = time.time() + T
        while time.time() < end:
            k = math.sqrt(2 / 2.0 / hz)
            ro += -ro / 2.0 / hz + r.gauss(0, 1) * k
            pi_ += -pi_ / 2.0 / hz + r.gauss(0, 1) * k
            th += -th / 4.0 / hz + r.gauss(0, 1) * k
            p, roll = self.pos, 1500 + 300 * max(-1, min(1, ro))
            if p and dist(p, self.home) > 1200:
                vfr = self.st.get("VFR_HUD")
                hdg = vfr.heading if vfr else 0
                dn, de = (self.home[0] - p[0]), (self.home[1] - p[1]) * math.cos(math.radians(p[0]))
                want = math.degrees(math.atan2(de, dn)) % 360
                roll = 1800 if ((want - hdg) % 360) < 180 else 1200
            ch = [int(roll), int(1500 + 150 * max(-1, min(1, pi_))), int(1550 + 150 * max(-1, min(1, th))), 1500]
            self.conn.mav.rc_channels_override_send(self.sysid, 1, *ch, *([65535] * 4))
            if not self.alive():
                raise MissionError("link lost teleop")
            time.sleep(1.0 / hz)
        for _ in range(3):
            self.conn.mav.rc_channels_override_send(self.sysid, 1, *([0] * 8)); time.sleep(0.1)
        self.ev("teleop_end")
        self.set_mode("RTL"); self.wait_disarmed(900)

    # ---------------------------------------------------------- recovery and main loop
    def recover(self):
        self.ev("recover_start")
        try:
            if self.armed:
                if self.kind == "copter":
                    try:
                        self.set_mode("RTL"); self.wait_disarmed(400)
                    except MissionError:
                        self.set_mode("LAND"); self.wait_disarmed(240)
                else:
                    try:
                        self.set_mode("QRTL"); self.wait_disarmed(600)
                    except MissionError:
                        self.set_mode("QLAND"); self.wait_disarmed(300)
        except MissionError:
            pass
        if self.armed:
            self.force_disarm()
        self.ev("recover_end", "armed=%d" % self.armed)

    def run(self):
        time.sleep(self.rng.uniform(0, 90))
        while not self.alive():
            time.sleep(1)
        last_req = 0
        missions = {"copter": dict(delivery=self.c_delivery, survey=self.c_survey, inspection=self.c_inspection,
                                   teleop=self.c_teleop),
                    "quadplane": dict(survey=self.p_survey, patrol=self.p_patrol, inspection=self.p_inspection,
                                      teleop=self.p_teleop)}[self.kind]
        names, w = zip(*sorted(self.d["mix"].items()))
        while True:
            while not self.alive():
                time.sleep(1)
            if time.time() - last_req > 12:
                self.request_streams(); last_req = time.time()
            time.sleep(2)
            p = self.pos
            if p is None or self.armed:
                if self.armed:
                    self.recover()
                continue
            self.home = p
            self.n_mission += 1
            self.mid = "%d-%04d" % (self.sysid, self.n_mission)
            name = self.rng.choices(names, weights=w)[0]
            try:
                self.request_streams()
                missions[name]()
                self.ev("mission_end", "ok")
            except MissionError as e:
                self.ev("mission_end", "fail:%s" % e)
                self.recover()
            except Exception as e:
                self.ev("mission_end", "error:%r" % e)
                self.recover()
            self.mid = ""
            idle = self.rng.uniform(20, 90)
            self.ev("idle", "%.0f" % idle)
            end = time.time() + idle
            while time.time() < end:
                if time.time() - last_req > 12:
                    self.request_streams(); last_req = time.time()
                time.sleep(1)


def main():
    drones = [Drone(d) for d in CFG["drones"]]
    for dr in drones:
        threading.Thread(target=dr.run, daemon=True).start()
    while True:
        time.sleep(60)
        print(time.strftime("%T"), " ".join("%d:%s%s" % (d.sysid, "A" if d.armed else "-", "" if d.alive() else "X")
                                            for d in drones), flush=True)


if __name__ == "__main__":
    main()
