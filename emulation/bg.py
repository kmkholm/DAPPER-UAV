"""Cross traffic of other user equipment sharing the cell's backhaul. Pure-python
on/off sessions (UDP bursts and occasional TCP bulk) to the core router's sink, so
nothing accumulates and the CPU cost is bounded (no external iperf processes)."""
import json, os, random, socket, sys, threading, time

CFG = json.load(open(sys.argv[1]))
L = CFG["load"]                      # {"on_mean","off_mean","rate_mbit":[lo,hi]}
TARGET, PORT = CFG["target"], CFG["port"]
rng = random.Random(hash(CFG["name"]) & 0xffffffff)
BIND = CFG["ifaces"][0]["addr"].split("/")[0]


def udp_session(rate_mbit, dur, down):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((BIND, 0))
    except OSError:
        return
    pkt = rng.randint(400, 1400)
    ipg = max(0.0005, pkt * 8 / (rate_mbit * 1e6))
    tag = b"DOWNREQ" if down else b"UP"
    end = time.time() + dur
    while time.time() < end:
        try:
            s.sendto(tag + os.urandom(pkt), (TARGET, PORT))
            if down:                         # ask sink to stream back a burst
                s.settimeout(0.05)
                try:
                    for _ in range(8):
                        s.recvfrom(2048)
                except OSError:
                    pass
        except OSError:
            break
        time.sleep(ipg)
    s.close()


def tcp_bulk(nbytes):
    try:
        c = socket.create_connection((TARGET, PORT + 1000), timeout=10, source_address=(BIND, 0))
        sent = 0
        while sent < nbytes:
            chunk = os.urandom(min(16384, nbytes - sent))
            c.sendall(chunk); sent += len(chunk)
        c.shutdown(socket.SHUT_WR); c.recv(16); c.close()
    except OSError:
        pass


def main():
    time.sleep(rng.uniform(5, 30))
    while True:
        dur = min(40.0, max(2.0, rng.expovariate(1.0 / L["on_mean"])))
        rate = rng.uniform(*L["rate_mbit"])
        if rng.random() < 0.3:
            threading.Thread(target=tcp_bulk, args=(int(rate * 1e6 / 8 * dur),), daemon=True).start()
        else:
            udp_session(rate, dur, down=rng.random() < 0.6)
        time.sleep(max(1.0, rng.expovariate(1.0 / L["off_mean"])))


if __name__ == "__main__":
    main()
