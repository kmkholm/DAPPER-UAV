"""Save screenshots of the live GNS3 testbed (web UI) as files for the paper:
  figures/gns3/topology.png      the whole project map with the 44 running nodes
  figures/gns3/console_<node>.png the built-in console of a drone, a cell router and a GCS
Uses the Playwright Chromium already installed for Python.

  python gns3_shots.py [project_id]
"""
import json, os, sys, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "figures", "gns3")); os.makedirs(OUT, exist_ok=True)
INI = os.path.join(os.environ["APPDATA"], "GNS3", "2.2", "gns3_server.ini")
PW = [l.split("=", 1)[1].strip() for l in open(INI) if l.startswith("password")][0]
PID = sys.argv[1] if len(sys.argv) > 1 else json.load(open(os.path.join(HERE, "project_ids.json")))["project_id"]
BASE = "http://localhost:3080/static/web-ui"
CONSOLE_NODES = ["A1-d1", "cell-A1", "gcs-A", "cell-B1", "B3-d1"]
VW = 2900 if (len(sys.argv) > 2 and sys.argv[2] == "topology") else 1900


def add_server(page):
    page.goto(BASE + "/servers", wait_until="networkidle")
    if page.locator("text=Add server").count() == 0:
        return
    page.click("text=Add server")
    page.wait_for_selector("mat-dialog-container")
    inputs = page.locator("mat-dialog-container input")
    inputs.nth(0).fill("local")
    inputs.nth(1).fill("localhost")          # same origin as the page -> symbol images get the credentials
    # authorization -> basic
    page.locator("mat-dialog-container mat-select").last.click()
    page.locator("mat-option", has_text="Basic").click()
    inputs = page.locator("mat-dialog-container input")
    inputs.nth(inputs.count() - 2).fill("admin")
    inputs.nth(inputs.count() - 1).fill(PW)
    page.locator("mat-dialog-container button", has_text="Add").click()
    page.wait_for_timeout(1000)


def open_project(page):
    page.goto(f"{BASE}/server/1/project/{PID}", wait_until="networkidle")
    page.wait_for_selector("svg", timeout=60000)
    page.wait_for_timeout(5000)
    for sel in ("text=Close",):
        try:
            page.locator(sel).first.click(timeout=1500)
        except Exception:
            pass
    # hide the node-list side panel (top-right toggle button), then zoom out so the whole map fits
    page.mouse.click(VW - 41, 30); page.wait_for_timeout(1500)
    if len(sys.argv) > 2 and sys.argv[2] == "topology":
        # shrink the whole page so both fleets fit inside the viewport (device_scale_factor keeps it crisp)
        page.evaluate("document.body.style.zoom = 0.62"); page.wait_for_timeout(1500)
    # zoom out with the mouse wheel over the map (d3-zoom), centred so both fleets fit
    page.wait_for_timeout(2500)


def main():
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        ctx = br.new_context(viewport={"width": VW, "height": 1000}, device_scale_factor=2,
                             http_credentials={"username": "admin", "password": PW})
        page = ctx.new_page()
        add_server(page)
        open_project(page)
        page.screenshot(path=os.path.join(OUT, "topology.png"), full_page=False)
        print("saved topology.png")
        if len(sys.argv) > 2 and sys.argv[2] == "topology":
            br.close(); return
        # consoles: right-click a node label on the map -> Console
        for name in CONSOLE_NODES:
            try:
                lab = page.locator(f"svg text:text-is('{name}')").first
                lab.click(button="right", timeout=5000)
                page.wait_for_timeout(800)
                page.locator("button:has-text('Console'), .mat-menu-item:has-text('Console')").first.click(timeout=4000)
                page.wait_for_timeout(12000)          # let the telnet console stream a few log lines
                page.screenshot(path=os.path.join(OUT, f"console_{name}.png"), full_page=False)
                print("saved console_%s.png" % name)
                # close the console tab/panel if the UI opened one in place
                try:
                    page.locator("mat-icon:text('close')").last.click(timeout=1500)
                except Exception:
                    pass
            except Exception as e:
                print("console", name, "failed:", str(e)[:120])
                page.screenshot(path=os.path.join(OUT, f"debug_{name}.png"))
        br.close()


if __name__ == "__main__":
    main()
