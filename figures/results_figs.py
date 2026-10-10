"""Result figures from results/runs/*.json (+ *_pred.npz). Run after the training grid.
Produces Figure3..Figure9. Figure2 (testbed + traffic signatures) and Figure10 (t-SNE +
differential-attention XAI) are in repr_figs.py."""
import glob, json, os, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import style

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "results", "runs")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
MAIN = ["local", "central", "fedavg", "fedprox", "fedbn", "dapper"]
ABL = ["dapper", "dapper_softmax", "dapper_nogate", "dapper_cal", "dapper_prompt", "dapper_nost", "dapper_nobins", "dapper_pk"]
NETS = {"A": "Multirotor fleet (WireGuard)", "B": "VTOL fleet (IPsec)"}
TT = os.environ.get("DAPPER_TASK_TAG", "t")        # t = op3 headline, c = op4, "" = fine


def load(net, method, seed, ho=""):
    p = os.path.join(RUNS, "%s%s_%s_s%d%s.json" % (net, TT, method, seed, "_ho" + ho if ho else ""))
    return json.load(open(p)) if os.path.exists(p) else None


def collect(net, method, key="macro_f1", seeds=(1, 2, 3)):
    v = [load(net, method, s)["overall"][key] for s in seeds if load(net, method, s)]
    return (np.mean(v), np.std(v), len(v)) if v else (np.nan, 0, 0)


# ---------------------------------------------------------------- Figure 3: main comparison
def fig_main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, net in zip(axes, "AB"):
        xs = np.arange(len(MAIN))
        m = [collect(net, me)[0] for me in MAIN]
        e = [collect(net, me)[1] for me in MAIN]
        bars = ax.bar(xs, m, yerr=e, capsize=3, color=[style.METHOD_C[me] for me in MAIN],
                      edgecolor="#2b2f36", linewidth=0.7)
        for b, v in zip(bars, m):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, "%.3f" % v, ha="center", fontsize=7.5)
        ax.set_xticks(xs); ax.set_xticklabels([style.METHOD_LABEL[me] for me in MAIN], rotation=25, ha="right")
        ax.set_ylim(0, 1.0); ax.set_title("%s" % NETS[net]); ax.set_ylabel("Macro-F1 (held-out flights)")
    fig.suptitle("Flight-phase inference from encrypted C2 traffic — federated methods", fontweight="bold")
    style.save(fig, os.path.join(OUT, "Figure4"))


# ---------------------------------------------------------------- Figure 4: per-site heatmap
def fig_persite():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, net in zip(axes, "AB"):
        r1 = load(net, "dapper", 1)
        sites = r1["train_sites"]
        M = np.full((len(MAIN), len(sites)), np.nan)
        for i, me in enumerate(MAIN):
            accs = {s: [] for s in sites}
            for seed in (1, 2, 3):
                r = load(net, me, seed)
                if not r:
                    continue
                for s in sites:
                    if s in r["per_site"]:
                        accs[s].append(r["per_site"][s]["f1"])
            for j, s in enumerate(sites):
                if accs[s]:
                    M[i, j] = np.mean(accs[s])
        im = ax.imshow(M, cmap="viridis", vmin=0.3, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_yticks(range(len(MAIN))); ax.set_yticklabels([style.METHOD_LABEL[m] for m in MAIN])
        for i in range(len(MAIN)):
            for j in range(len(sites)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] < 0.72 else "black")
        ax.set_title(NETS[net]); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Macro-F1")
    fig.suptitle("Per-site macro-F1: heterogeneity across cell sites", fontweight="bold")
    style.save(fig, os.path.join(OUT, "Figure5"))


# ---------------------------------------------------------------- Figure 5: confusion matrices
def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, net in zip(axes, "AB"):
        r = load(net, "dapper", 1)
        cm = np.array(r["confusion"], float)
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        cl = r["classes"]
        ax.set_xticks(range(len(cl))); ax.set_xticklabels(cl, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(cl))); ax.set_yticklabels(cl, fontsize=7.5)
        for i in range(len(cl)):
            for j in range(len(cl)):
                ax.text(j, i, "%.2f" % cmn[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] > 0.5 else "#333")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("DAPPER — " + NETS[net])
    fig.suptitle("Row-normalised confusion (held-out flights)", fontweight="bold")
    style.save(fig, os.path.join(OUT, "Figure6"))


# ---------------------------------------------------------------- Figure 6: ROC per phase
def fig_roc():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, net in zip(axes, "AB"):
        p = os.path.join(RUNS, "%s%s_dapper_s1_pred.npz" % (net, TT))
        if not os.path.exists(p):
            continue
        d = np.load(p); y, prob = d["y"], d["prob"].astype(np.float32)
        cl = load(net, "dapper", 1)["classes"]
        for c in range(len(cl)):
            yb = (y == c).astype(int)
            if yb.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(yb, prob[:, c])
            ax.plot(fpr, tpr, color=style.PHASE_C[c], lw=1.6, label="%s (AUC %.3f)" % (cl[c], auc(fpr, tpr)))
        ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=0.8)
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
        ax.set_title("DAPPER — " + NETS[net]); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Per-phase one-vs-rest ROC", fontweight="bold")
    style.save(fig, os.path.join(OUT, "Figure7"))


# ---------------------------------------------------------------- Figure 7: training curves
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, net in zip(axes, "AB"):
        for me in ["fedavg", "fedprox", "fedbn", "dapper"]:
            r = load(net, me, 1)
            if not r or not r["history"]:
                continue
            h = r["history"]
            ax.plot([x["round"] for x in h], [x["val_macro_f1"] for x in h],
                    "-o", ms=3, color=style.METHOD_C[me], label=style.METHOD_LABEL[me])
        ax.set_xlabel("Federated round"); ax.set_ylabel("Validation macro-F1")
        ax.set_title(NETS[net]); ax.legend(fontsize=8)
    fig.suptitle("Convergence of federated training", fontweight="bold")
    style.save(fig, os.path.join(OUT, "Figure8"))


# ---------------------------------------------------------------- Figure 8: ablations
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, net in zip(axes, "AB"):
        xs = np.arange(len(ABL))
        m = [collect(net, me)[0] for me in ABL]
        e = [collect(net, me)[1] for me in ABL]
        ax.bar(xs, m, yerr=e, capsize=3, color=[style.METHOD_C[me] for me in ABL], edgecolor="#2b2f36", lw=0.7)
        for x, v in zip(xs, m):
            if not np.isnan(v):
                ax.text(x, v + 0.012, "%.3f" % v, ha="center", fontsize=7.5)
        ax.set_xticks(xs); ax.set_xticklabels([style.METHOD_LABEL[me] for me in ABL], rotation=20, ha="right")
        ax.set_ylim(0, 1.0); ax.set_title(NETS[net]); ax.set_ylabel("Macro-F1")
    fig.suptitle("Ablation of DAPPER components", fontweight="bold")
    style.save(fig, os.path.join(OUT, "Figure9"))


# ---------------------------------------------------------------- Figure 9: new-site onboarding
def fig_onboard():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, net in zip(axes, "AB"):
        ho = {"A": "A3", "B": "B3"}[net]
        for me in ("dapper", "fedavg"):
            curves = []
            for seed in (1, 2, 3):
                r = load(net, me, seed, ho=ho)
                if r and r["onboard"]:
                    shots = sorted(int(k) for k in r["onboard"])
                    curves.append([r["onboard"][str(s)]["bacc"] for s in shots])     # held-out site may lack a state: balanced accuracy over present states
            if curves:
                shots = sorted(int(k) for k in load(net, me, 1, ho=ho)["onboard"])
                mu = np.mean(curves, 0); sd = np.std(curves, 0)
                ax.plot(shots, mu, "-o", ms=4, color=style.METHOD_C[me], label=style.METHOD_LABEL[me])
                ax.fill_between(shots, mu - sd, mu + sd, color=style.METHOD_C[me], alpha=0.15)
        ax.set_xlabel("Labelled windows from the new site"); ax.set_ylabel("Balanced accuracy on new site"); ax.set_ylim(0.4, 1.02)
        ax.set_title("%s — held-out site %s" % (NETS[net], ho)); ax.legend(fontsize=8)
    fig.suptitle("Onboarding a previously unseen cell site", fontweight="bold")
    style.save(fig, os.path.join(OUT, "Figure10"))


# ---------------------------------------------------------------- Figure 12: three granularities, DAPPER vs the prior pipelines
def fig_granularity():
    import json as _j
    def prior(tag):
        p = os.path.join(RUNS, "prior_%s.json" % tag)
        return _j.load(open(p)) if os.path.exists(p) else {}
    def method(tag, me):
        v = []
        for s in (1, 2, 3):
            p = os.path.join(RUNS, "%s_%s_s%d.json" % (tag, me, s))
            if os.path.exists(p):
                v.append(_j.load(open(p))["overall"]["macro_f1"])
        return (np.mean(v), np.std(v)) if v else (np.nan, 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    gran = [("t", "Operational state\n(3 classes)"), ("c", "Operational phases\n(4 classes)"), ("", "Kinematic phases\n(7 / 6 classes)")]
    series = [("FedAvg", "fedavg", style.METHOD_C["fedavg"]), ("Central (reference)", "central", style.METHOD_C["central"]),
              ("DAPPER", "dapper", style.METHOD_C["dapper"])]
    for ax, net in zip(axes, "AB"):
        xs = np.arange(len(gran)); w = 0.26
        for i, (lab, key, col) in enumerate(series):
            m, e = [], []
            for tag, _ in gran:
                mu, sd = method(net + tag, key); m.append(mu); e.append(sd)
            bars = ax.bar(xs + (i - 1) * w, m, w, yerr=e, capsize=2, color=col, edgecolor="#2b2f36", lw=0.6, label=lab)
            for b, v in zip(bars, m):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, "%.2f" % v, ha="center", fontsize=6.5)
        ax.set_xticks(xs); ax.set_xticklabels([g[1] for g in gran]); ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro-F1 (held-out flights)"); ax.set_title(NETS[net]); ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("What the encrypted tunnel reveals: leakage by granularity", fontweight="bold")
    style.save(fig, os.path.join(OUT, "Figure12"))


if __name__ == "__main__":
    for f in (fig_main, fig_persite, fig_confusion, fig_roc, fig_curves, fig_ablation, fig_onboard, fig_granularity):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAIL", f.__name__, repr(e))
