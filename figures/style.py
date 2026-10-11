"""Shared publication style for all DAPPER figures (colour-blind-safe, vivid, consistent)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#30363d",
    "axes.grid": True, "grid.color": "#dfe3e8", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.fontsize": 8, "legend.frameon": False, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# one colour per federated method, used everywhere
METHOD_C = {
    "local": "#9aa0a6", "central": "#111827", "fedavg": "#1f77b4", "fedprox": "#17becf",
    "fedbn": "#2ca02c", "dapper": "#d62728", "dapper_softmax": "#ff7f0e",
    "dapper_cal": "#9467bd", "dapper_prompt": "#8c564b", "dapper_nogate": "#e377c2", "dapper_nobins": "#7f7f7f", "dapper_nost": "#bcbd22", "dapper_pk": "#17becf",
}
METHOD_LABEL = {
    "local": "Local", "central": "Central", "fedavg": "FedAvg", "fedprox": "FedProx",
    "fedbn": "FedBN", "dapper": "DAPPER", "dapper_softmax": "DAPPER (softmax)",
    "dapper_cal": "DAPPER (cal only)", "dapper_prompt": "DAPPER (prompt only)", "dapper_nogate": "DAPPER (no gate)", "dapper_nost": "DAPPER (patches only)", "dapper_nobins": "DAPPER (statistics only)", "dapper_pk": "DAPPER (+ packet tokens)",
}
# site palette
SITE_C = {"A1": "#1b9e77", "A2": "#d95f02", "A3": "#7570b3", "A4": "#e7298a",
          "B1": "#66a61e", "B2": "#e6ab02", "B3": "#a6761d", "B4": "#666666"}
# phase (class) palette
PHASE_C = ["#6c757d", "#1f77b4", "#8c564b", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e"]


def save(fig, path):
    for ext in ("pdf", "png"):
        fig.savefig(path + "." + ext)
    plt.close(fig)
