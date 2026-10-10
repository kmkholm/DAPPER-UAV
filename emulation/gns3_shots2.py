"""Second round of GNS3 web-UI screenshots: the tunnels, the link shaping and the live captures.

For each node the console is opened in the GNS3 web UI first (the telnet console only shows output produced after
it connects), then a real command is executed inside that node's container with its output appended to the
node's log, which the console is streaming; the screenshot is taken a few seconds later.  Key material in the
IPsec state is masked before it is printed.

  python gns3_shots2.py <project_id>
Writes figures/gns3/console2_<node>.png
"""
import os, sys, subprocess, time
sys.argv = [sys.argv[0], sys.argv[1]]                      # consoles viewport (not the topology mode)
import gns3_shots as G
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
PID = sys.argv[1]
LOG = "/data/node.log"


def c(name):
    return "GNS3.%s.%s" % (name, PID)


def sh(node, script, detach=False):
    inner = script.replace("'", "'\"'\"'")
    cmd = "docker exec %s %s sh -c '%s'" % ("-d" if detach else "", c(node), inner)
    return subprocess.run([sys.executable, os.path.join(HERE, "vm_ssh.py"), "run", cmd], capture_output=True, text=True, timeout=120)


INJECT = {
    "A2-d1": None,                                                     # ArduCopter SITL status lines (already streaming)
    "B3-d1": None,                                                     # ArduPlane SITL status lines, LEO site
    "A1-d1": "(echo; echo \"### wg show   (WireGuard tunnel carrying this drone's MAVLink)\"; wg show) >> %s 2>&1" % LOG,
    "B1-d1": ("(echo; echo \"### ip xfrm state   (IPsec ESP security associations; key material hidden)\"; "
              "ip xfrm state | sed -E \"s/0x[0-9a-f]{16,}/0x<hidden>/g\") >> %s 2>&1" % LOG),
    "cell-B3": ("(echo; echo \"### tc qdisc show   (LEO satellite backhaul of site B3)\"; tc qdisc show; "
                "echo; echo \"### tc -s qdisc show dev eth0\"; tc -s qdisc show dev eth0) >> %s 2>&1" % LOG),
    "cell-A1": ("DETACH", "tcpdump -n -l -q -i eth1 2>&1 | head -4000 >> %s" % LOG),
    "cell-B1": ("DETACH", "tcpdump -n -l -q -i eth1 2>&1 | head -4000 >> %s" % LOG),
    "gcs-A": None,                                                     # mission events stream on their own
}


def main():
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        ctx = br.new_context(viewport={"width": G.VW, "height": 1000}, device_scale_factor=2,
                             http_credentials={"username": "admin", "password": G.PW})
        page = ctx.new_page()
        G.add_server(page)
        G.open_project(page)
        for name, inj in INJECT.items():
            try:
                lab = page.locator(f"svg text:text-is('{name}')").first
                lab.click(button="right", timeout=5000)
                page.wait_for_timeout(800)
                page.locator("button:has-text('Console'), .mat-menu-item:has-text('Console')").first.click(timeout=4000)
                page.wait_for_timeout(3000)
                if isinstance(inj, tuple):
                    sh(name, inj[1], detach=True); page.wait_for_timeout(9000)
                elif inj:
                    r = sh(name, inj); print(name, "inject rc", r.returncode, r.stderr[-200:]); page.wait_for_timeout(4000)
                else:
                    page.wait_for_timeout(12000)
                page.screenshot(path=os.path.join(G.OUT, f"console2_{name}.png"), full_page=False)
                print("saved console2_%s.png" % name, flush=True)
                try:
                    page.locator("mat-icon:text('close')").last.click(timeout=1500)
                except Exception:
                    pass
            except Exception as e:
                print("console", name, "failed:", str(e)[:160])
        br.close()


if __name__ == "__main__":
    main()
