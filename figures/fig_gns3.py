"""Figure 2 - the GNS3 testbed in operation (real screenshots of the GNS3 web UI).
(a) project map; (b,c) SITL state of a flying multirotor and VTOL; (d) WireGuard tunnel; (e) IPsec SAs (keys
hidden); (f) LEO link shaping; (g,h) encrypted packets at two cell routers; (i) mission events.
Input: figures/gns3/topology.png (gns3_shots.py topology) and console2_*.png (gns3_shots2.py).
Output: figures/out/Figure2.pdf/png"""
import os
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, 'gns3'); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8})


def crop(name, box):
    return Image.open(os.path.join(G, name)).crop(box)


topo = crop('topology.png', (270, 330, 3420, 1280))
CON = (165, 1112, 1596, 1950)                      # console window body (2 px per CSS px)
panels = [('console2_A2-d1.png', '(b) multirotor A2-d1 in flight: ArduCopter SITL state'),
          ('console2_B3-d1.png', '(c) VTOL B3-d1 in flight behind the LEO site'),
          ('console2_A1-d1.png', '(d) A1-d1: WireGuard tunnel (wg show)'),
          ('console2_B1-d1.png', '(e) B1-d1: IPsec ESP SAs (ip xfrm state)'),
          ('console2_cell-B3.png', '(f) cell-B3: LEO link shaping (tc qdisc)'),
          ('console2_cell-A1.png', '(g) cell-A1: encrypted WireGuard packets'),
          ('console2_cell-B1.png', '(h) cell-B1: encrypted ESP packets'),
          ('console2_gcs-A.png', '(i) ground-control station A: mission events')]

fig = plt.figure(figsize=(7.2, 8.6))
ax = fig.add_axes([0.005, 0.765, 0.99, 0.215]); ax.imshow(topo); ax.axis('off')
ax.set_title('(a) GNS3 project map: two fleets, eight cell sites, 24 drones, all 44 nodes running', fontsize=8, fontweight='bold', loc='left', pad=3)
H, top, gap = 0.172, 0.735, 0.012
for i, (name, title) in enumerate(panels):
    r, col = divmod(i, 2)
    a = fig.add_axes([0.005 + col * 0.5, top - (r + 1) * H - r * gap + 0.012, 0.49, H - 0.012])
    a.imshow(crop(name, CON)); a.axis('off')
    a.set_title(title, fontsize=7, fontweight='bold', loc='left', pad=2)
fig.savefig(os.path.join(OUT, 'Figure2.pdf'), dpi=300); fig.savefig(os.path.join(OUT, 'Figure2.png'), dpi=300)
print('Figure 2 (testbed screenshots) written')
