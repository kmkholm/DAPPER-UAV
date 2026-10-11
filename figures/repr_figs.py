"""Figure 2 (testbed + encrypted-traffic phase signatures, from processed data) and
Figure 10 (t-SNE of learned embeddings + differential-attention map on a real window)."""
import glob, os, sys, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import style
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from phases import COPTER_CLASSES, PLANE_CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "proc")
RUNS = os.path.join(HERE, "..", "results", "runs")
MODELS = os.path.join(HERE, "..", "results", "models")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
NB = 41
# bucket centres for the up/down size histograms (must match process.EDGES)
EDGES = np.concatenate([np.arange(64, 577, 16), np.arange(704, 1601, 128), [100000]])
CENT = (EDGES[:-1] + np.minimum(EDGES[1:], 1700)) / 2


def net_bins(net):
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(PROC, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if str(z["net"]) != net:
            continue
        X.append(np.asarray(z["X"], np.float32)); Y.append(np.asarray(z["y"]))
    return np.concatenate(X), np.concatenate(Y)


# ---------------------------------------------------------------- Figure 2
def fig_testbed():
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.22)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("DAPPER GNS3 testbed: two operator networks, four cell sites each", fontweight="bold", loc="left")

    def box(x, y, w, h, t, fc, ec, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color="#1b2631")

    for col, (net, lab, tun, fc, ec) in enumerate([("A", "SkyNet-A · ArduCopter", "MAVLink 2 / WireGuard", "#eaf2fb", "#2e6da4"),
                                                    ("B", "SkyNet-B · ArduPlane VTOL", "MAVLink 2 / IPsec ESP", "#fdecea", "#c0392b")]):
        x0 = 0.03 + col * 0.5
        box(x0, 0.82, 0.44, 0.13, "%s\nGCS  —  %s" % (lab, tun), fc, ec, 9)
        sites = {"A": ["urban-5G", "suburban-LTE", "rural-LTE", "congested-LTE"],
                 "B": ["5G-SA", "LTE", "LEO-sat", "GEO-sat"]}[net]
        for i, s in enumerate(sites):
            sx = x0 + i * 0.11
            box(sx, 0.45, 0.10, 0.12, "%s%d\n%s" % (net, i + 1, s), "#ffffff", ec, 7)
            ax.add_patch(FancyArrowPatch((sx + 0.05, 0.57), (x0 + 0.22, 0.82), arrowstyle="-", lw=0.8, color="#8a9099"))
            for j in range(3):
                dx = sx + 0.012 + j * 0.028
                ax.add_patch(FancyBboxPatch((dx, 0.30), 0.022, 0.07, boxstyle="round,pad=0.002,rounding_size=0.01",
                                            fc=ec, ec=ec, alpha=0.5))
                ax.add_patch(FancyArrowPatch((dx + 0.011, 0.37), (sx + 0.05, 0.45), arrowstyle="-", lw=0.6, color="#bbb"))
            ax.text(sx + 0.05, 0.265, "3 drones", ha="center", fontsize=6.5, color="#555")
        ax.text(x0 + 0.22, 0.24, "per-drone capture at each cell router (tcpdump, outer/encrypted packets)",
                ha="center", fontsize=7, color="#444", style="italic")

    # per-phase traffic signatures
    for col, net in enumerate("AB"):
        axp = fig.add_subplot(gs[1, col])
        X, Y = net_bins(net)
        cls = COPTER_CLASSES if net == "A" else PLANE_CLASSES
        for c in range(len(cls)):
            m = Y == c
            if m.sum() < 50:
                continue
            up = X[m][:, :NB].mean(0)
            axp.plot(CENT, up, color=style.PHASE_C[c], lw=1.6, label=cls[c])
        axp.set_xlabel("uplink packet size (bytes)"); axp.set_ylabel("mean packets / 100 ms bin")
        axp.set_title("Uplink size signature per phase — " + ("Multirotor" if net == "A" else "VTOL"))
        axp.legend(fontsize=7, ncol=2)
    style.save(fig, os.path.join(OUT, "Figure1"))
    print("Figure1 (testbed) written")


# ---------------------------------------------------------------- Figure 10 (needs a saved model)
def fig_xai():
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "code"))
    from model import DAPPER
    from data import load_network, WindowStore
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    try:
        from sklearn.manifold import TSNE
    except Exception:
        TSNE = None
    for ax, net in zip(axes, "AB"):
        mp = os.path.join(MODELS, "%st_dapper_s1.pt" % net)
        if not os.path.exists(mp):
            continue
        ck = torch.load(mp, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]; cls = ck["classes"]; site = sorted(ck["sites"])[0]
        model = DAPPER(**cfg); model.load_state_dict(ck["sites"][site]); model.eval()
        TASK_MAP = {"op4": {"A": [0, 2, 1, 2, 1, 3, 2], "B": [0, 2, 2, 2, 1, 3]}, "op3": {"A": [0, 1, 1, 1, 1, 2, 1], "B": [0, 1, 1, 1, 1, 2]}}
        task = ck.get("task", "op3")
        X, clients, V = load_network(PROC, net, T=cfg["T"], label_map=None if task == "fine" else np.array(TASK_MAP[task][net]))
        store = WindowStore(X, V, T=cfg["T"], device="cpu")
        d = clients[site]; rows = d["row"][d["split"] == 2][:1500]; yl = d["y"][d["split"] == 2][:1500]
        with torch.no_grad():
            emb = model.embed(*store.get(rows)).numpy()
        if TSNE is not None and len(emb) > 60:
            z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb)
            for c in range(len(cls)):
                m = yl == c
                if m.sum():
                    ax.scatter(z[m, 0], z[m, 1], s=6, color=style.PHASE_C[c], label=cls[c], alpha=0.7, lw=0)
            ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7, ncol=2, markerscale=1.5)
        ax.set_title("Learned embedding (t-SNE) — " + ("Multirotor" if net == "A" else "VTOL") + ", site " + site)
    fig.suptitle("DAPPER [CLS] embeddings coloured by true flight phase", fontweight="bold")
    style.save(fig, os.path.join(OUT, "Figure11"))
    print("Figure10 written")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "testbed"):
        fig_testbed()
    if which in ("all", "xai"):
        try:
            fig_xai()
        except Exception as e:
            print("Figure10 skipped:", repr(e))
