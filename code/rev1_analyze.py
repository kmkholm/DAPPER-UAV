"""Revision 1 analysis of results/rev1/runs (no GPU, numpy + scikit-learn only).

  python rev1_analyze.py    -> results/rev1/analysis.json, paper/auto_rev1_*.tex, paper/auto_rev1_macros.tex

Pooled out-of-fold scores: the five test folds of the sortie-grouped cross-validation together cover every
sortie once, so their predictions are concatenated and scored once; 95 % intervals come from a bootstrap
over sorties (the independent units), and the spread over folds is given alongside.
"""
import glob, json, os
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, roc_curve, auc

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "results", "rev1", "runs")
OUT = os.path.join(HERE, "..", "results", "rev1")
PAPER = os.path.join(HERE, "..", "paper")
CLASSES = ["Ground", "Autonomous flight", "Tele-operation"]
NC = 3
METHODS = ["local", "central", "fedavg", "fedprox", "fedbn", "dapper"]
T_S, STRIDE_S = 12.8, 2.0
NBOOT = 2000


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def pred(name):
    p = os.path.join(RUNS, name + "_pred.npz")
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def mf1(y, yp, present=True):
    labels = np.unique(y) if present else np.arange(NC)
    return float(f1_score(y, yp, average="macro", labels=labels, zero_division=0))


def pooled(names):
    """Concatenate the test predictions of several runs (folds)."""
    P = [pred(n) for n in names]
    if any(p is None for p in P):
        return None
    out = {}
    for k in P[0]:
        out[k] = np.concatenate([p[k] for p in P])
    out["fold"] = np.concatenate([np.full(len(p["y"]), i) for i, p in enumerate(P)])
    if "all_y" in P[0]:
        out["all_fold"] = np.concatenate([np.full(len(p["all_y"]), i) for i, p in enumerate(P)])
    return out


def sortie_key(d, pre=""):
    return np.asarray(d[pre + "drone"]).astype(np.int64) * 100000 + np.asarray(d[pre + "mission"]).astype(np.int64)


def boot_ci(y, yp, key, rng, present=True):
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(k, []).append(i)
    g = [np.asarray(v) for v in groups.values()]
    vals = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(g), len(g))
        idx = np.concatenate([g[j] for j in pick])
        vals.append(mf1(y[idx], yp[idx], present))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def chance(y):
    """Macro-F1 of uninformed predictors on a label vector y (expected values for the random ones)."""
    pi = np.bincount(y, minlength=NC) / len(y)
    present = pi > 0
    K = int(present.sum())
    uni = np.mean([2 * p * (1 / K) / (p + 1 / K) for p in pi[present]])          # uniform random guess
    prior = np.mean(pi[present])                                                  # guess drawn from the class prior
    maj = (2 * pi.max() / (1 + pi.max())) / K                                     # always the majority class
    return dict(uniform=float(uni), prior=float(prior), majority=float(maj), best=float(max(uni, prior, maj)),
                priors=pi.tolist())


def causal_smooth(d, k):
    key = sortie_key(d); order = np.lexsort((d["start"], key))
    P = d["prob"].astype(np.float64)[order]; kk = key[order]
    S = P.copy()
    for g in np.unique(kk):
        idx = np.where(kk == g)[0]
        c = np.cumsum(P[idx], 0)
        for j, i in enumerate(idx):
            lo = j - k
            S[i] = (c[j] - (c[lo] if lo >= 0 else 0.0)) / min(k, j + 1)
    out = np.empty_like(S); out[order] = S
    return out


def detection_delays(d):
    """Seconds from the first window whose last 100 ms bin shows a new state to the first window whose
    prediction equals that state, over every state change in the test sorties (all windows kept)."""
    key = sortie_key(d, "all_"); order = np.lexsort((d["all_start"], key))
    yl = d["all_ylast"][order]; yp = d["all_yp"][order]; kk = key[order]
    res = {}
    for g in np.unique(kk):
        idx = np.where(kk == g)[0]
        for j in range(1, len(idx)):
            a, b = yl[idx[j - 1]], yl[idx[j]]
            if a == b or a < 0 or b < 0:
                continue
            hit = np.where(yp[idx[j:]] == b)[0]
            nxt = np.where(yl[idx[j:]] != b)[0]
            limit = nxt[0] if len(nxt) else len(idx) - j
            delay = float(hit[0] * STRIDE_S) if (len(hit) and hit[0] < limit) else np.nan
            res.setdefault("%s->%s" % (CLASSES[a], CLASSES[b]), []).append(delay)
    out = {}
    for k, v in res.items():
        v = np.asarray(v)
        ok = v[~np.isnan(v)]
        out[k] = dict(n=int(len(v)), detected=float(len(ok) / len(v)),
                      median=float(np.median(ok)) if len(ok) else None, p90=float(np.percentile(ok, 90)) if len(ok) else None)
    return out


def summarize_pool(d, rng, folds_json):
    y, yp = d["y"], d["yp"]; key = sortie_key(d)
    lo, hi = boot_ci(y, yp, key, rng)
    per_fold = [mf1(y[d["fold"] == f], yp[d["fold"] == f]) for f in np.unique(d["fold"])]
    s = dict(macro_f1=mf1(y, yp), ci95=[lo, hi], bacc=float(balanced_accuracy_score(y, yp)), acc=float((y == yp).mean()),
             fold_mean=float(np.mean(per_fold)), fold_sd=float(np.std(per_fold)), per_fold=per_fold,
             n_windows=int(len(y)), n_sorties=int(len(np.unique(key))),
             sorties_per_class=[int(len(np.unique(key[y == c]))) for c in range(NC)],
             windows_per_class=np.bincount(y, minlength=NC).tolist(),
             recall=[float(((yp == c) & (y == c)).sum() / max((y == c).sum(), 1)) for c in range(NC)],
             confusion=confusion_matrix(y, yp, labels=range(NC)).tolist(), chance=chance(y))
    sites = np.unique(d["site"])
    s["per_site"] = {}
    for st in sites:
        m = d["site"] == st
        s["per_site"][str(st)] = dict(f1_present=mf1(y[m], yp[m]), f1_all3=mf1(y[m], yp[m], present=False),
                                      bacc=float(balanced_accuracy_score(y[m], yp[m])), n=int(m.sum()),
                                      classes_present=np.unique(y[m]).tolist())
    for k in (1, 3, 5):
        ps = causal_smooth(d, k).argmax(1)
        s["causal_k%d" % k] = mf1(y, ps)
    if "all_y" in d:
        ya, ypa, pa = d["all_y"], d["all_yp"], d["all_pur"]
        tr = pa < 0.7
        s["all_windows"] = dict(macro_f1=mf1(ya, ypa), acc=float((ya == ypa).mean()), n=int(len(ya)),
                                transition_n=int(tr.sum()),
                                transition_acc_majority=float((ya[tr] == ypa[tr]).mean()) if tr.any() else None,
                                transition_acc_last=float((d["all_ylast"][tr] == ypa[tr]).mean()) if tr.any() else None)
        s["delays"] = detection_delays(d)
    if folds_json:
        h = [r["history"] for r in folds_json if r.get("history")]
        if h:
            s["val_curve"] = dict(rounds=[x["round"] for x in h[0]],
                                  present=np.mean([[x.get("val_macro_f1_present", np.nan) for x in hh] for hh in h], 0).tolist(),
                                  all3=np.mean([[x["val_macro_f1"] for x in hh] for hh in h], 0).tolist())
    return s


def roc_data(d):
    out = {}
    for c in range(NC):
        yt = (d["y"] == c).astype(int)
        if yt.min() == yt.max():
            continue
        fpr, tpr, _ = roc_curve(yt, d["prob"][:, c].astype(np.float64))
        out[CLASSES[c]] = dict(auc=float(auc(fpr, tpr)), fpr=fpr[:: max(1, len(fpr) // 400)].tolist(),
                               tpr=tpr[:: max(1, len(tpr) // 400)].tolist())
    return out


def main():
    rng = np.random.default_rng(2026)
    A = {"cv": {}, "companion": {}, "lodo": {}, "lomt": {}, "onboarding": {}}
    for net in ("A", "B"):
        for m in METHODS:
            names = ["%st_%s_s1_cv5%d" % (net, m, f) for f in range(5)]
            d = pooled(names)
            if d is None:
                continue
            js = [jl(os.path.join(RUNS, n + ".json")) for n in names]
            s = summarize_pool(d, rng, js)
            if m == "dapper":
                s["roc"] = roc_data(d)
            A["cv"]["%s_%s" % (net, m)] = s
        for tr in ("nocomp", "nocomp_human"):
            d = pooled(["%st_dapper_s1_def%s_cv5%d" % (net, tr, f) for f in range(5)])
            if d is not None:
                s = summarize_pool(d, rng, None)
                s["overhead"] = float(np.mean([jl(os.path.join(RUNS, "%st_dapper_s1_def%s_cv5%d.json" % (net, tr, f)))["overhead"] for f in range(5)]))
                A["companion"]["%s_%s" % (net, tr)] = s
        for f in (1, 2, 3):
            d = pred("%st_dapper_s1_lodo%d" % (net, f))
            if d is not None:
                d["fold"] = np.zeros(len(d["y"]), int)
                A["lodo"]["%s_%d" % (net, f)] = summarize_pool(d, rng, None)
        mts = ["delivery", "survey", "inspection"] if net == "A" else ["patrol", "survey", "inspection"]
        for t in mts:
            d = pred("%st_dapper_s1_lomt%s" % (net, t))
            if d is not None:
                d["fold"] = np.zeros(len(d["y"]), int)
                A["lomt"]["%s_%s" % (net, t)] = summarize_pool(d, rng, None)
        ho = {"A": "A3", "B": "B3"}[net]
        ob = [jl(os.path.join(RUNS, "%st_dapper_s%d_ho%s.json" % (net, s, ho))) for s in (1, 2)]
        ob = [o for o in ob if o]
        if ob:
            shots = sorted(ob[0]["onboard"], key=int)
            A["onboarding"][net] = {k: dict(bacc=float(np.mean([o["onboard"][k]["bacc"] for o in ob])),
                                            f1=float(np.mean([o["onboard"][k]["f1"] for o in ob])), n_seeds=len(ob)) for k in shots}
    # chance references of the original fixed split (defence table, Table 8)
    cnt = jl(os.path.join(OUT, "sortie_counts.json"))
    if cnt:
        A["fixed_split_chance"] = {}
        for net in ("A", "B"):
            w = np.asarray(cnt["summary"][net]["windows"][2])
            yv = np.repeat(np.arange(NC), w)
            A["fixed_split_chance"][net] = chance(yv)
    A["latency"] = jl(os.path.join(OUT, "latency.json"))
    json.dump(A, open(os.path.join(OUT, "analysis.json"), "w"), indent=1, default=float)
    write_tex(A)
    print_summary(A)


# ---------------------------------------------------------------- LaTeX
LAB = {"local": "Local", "central": "Centralised (reference)", "fedavg": "FedAvg", "fedprox": "FedProx", "fedbn": "FedBN", "dapper": "DAPPER"}


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else "%.3f" % x


def write_tex(A):
    mac = []
    def M(name, val):
        mac.append("\\newcommand{\\%s}{%s}" % (name, val))
    # main CV table
    rows = []
    for m in METHODS:
        cells = []
        for net in ("A", "B"):
            s = A["cv"].get("%s_%s" % (net, m))
            cells += ["--", "--"] if s is None else ["%.3f [%.3f, %.3f]" % (s["macro_f1"], s["ci95"][0], s["ci95"][1]),
                                                     "%.3f $\\pm$ %.3f" % (s["fold_mean"], s["fold_sd"])]
        rows.append(LAB[m] + " & " + " & ".join(cells) + " \\\\")
    open(os.path.join(PAPER, "auto_rev1_cv_rows.tex"), "w").write("\n".join(rows) + "\n")
    for net, L in (("A", "A"), ("B", "B")):
        s = A["cv"].get("%s_dapper" % net)
        if s:
            M("CvF%s" % L, "%.3f" % s["macro_f1"]); M("CvLo%s" % L, "%.3f" % s["ci95"][0]); M("CvHi%s" % L, "%.3f" % s["ci95"][1])
            M("CvBacc%s" % L, "%.3f" % s["bacc"]); M("CvSort%s" % L, "%d" % s["n_sorties"])
            M("CvTeleSort%s" % L, "%d" % s["sorties_per_class"][2]); M("CvAutoSort%s" % L, "%d" % s["sorties_per_class"][1])
            M("CvGroundSort%s" % L, "%d" % s["sorties_per_class"][0])
            M("CvChance%s" % L, "%.3f" % s["chance"]["best"]); M("CvMaj%s" % L, "%.3f" % s["chance"]["majority"])
            M("CvCausalFive%s" % L, "%.3f" % s["causal_k5"]); M("CvCausalThree%s" % L, "%.3f" % s["causal_k3"])
            if "all_windows" in s:
                aw = s["all_windows"]
                M("CvAllF%s" % L, "%.3f" % aw["macro_f1"]); M("CvTransN%s" % L, "%d" % aw["transition_n"])
                M("CvTransAccLast%s" % L, "%.3f" % (aw["transition_acc_last"] or 0)); M("CvTransAccMaj%s" % L, "%.3f" % (aw["transition_acc_majority"] or 0))
            for i, c in enumerate(["Ground", "Auto", "Tele"]):
                M("CvRec%s%s" % (c, L), "%.3f" % s["recall"][i])
        fb = A["cv"].get("%s_fedbn" % net)
        if fb:
            M("CvFedbn%s" % L, "%.3f" % fb["macro_f1"])
        ce = A["cv"].get("%s_central" % net)
        if ce:
            M("CvCentral%s" % L, "%.3f" % ce["macro_f1"])
        fa = A["cv"].get("%s_fedavg" % net)
        if fa:
            M("CvFedavg%s" % L, "%.3f" % fa["macro_f1"])
        lo = A["cv"].get("%s_local" % net)
        if lo:
            M("CvLocal%s" % L, "%.3f" % lo["macro_f1"])
        for tr, T in (("nocomp", "NoComp"), ("nocomp_human", "Human")):
            s = A["companion"].get("%s_%s" % (net, tr))
            if s:
                M("%s%s" % (T, L), "%.3f" % s["macro_f1"]); M("%sLo%s" % (T, L), "%.3f" % s["ci95"][0]); M("%sHi%s" % (T, L), "%.3f" % s["ci95"][1])
                for i, c in enumerate(["Ground", "Auto", "Tele"]):
                    M("%sRec%s%s" % (T, c, L), "%.3f" % s["recall"][i])
        fc = A.get("fixed_split_chance", {}).get(net)
        if fc:
            M("FixChance%s" % L, "%.3f" % fc["best"]); M("FixMaj%s" % L, "%.3f" % fc["majority"]); M("FixUni%s" % L, "%.3f" % fc["uniform"])
        ob = A["onboarding"].get(net)
        if ob and "0" in ob:
            M("OnbZero%s" % L, "%.3f" % ob["0"]["bacc"])
        lod = [v for k, v in A["lodo"].items() if k.startswith(net)]
        if lod:
            M("LodoMin%s" % L, "%.3f" % min(v["macro_f1"] for v in lod)); M("LodoMax%s" % L, "%.3f" % max(v["macro_f1"] for v in lod))
            M("LodoBaccMin%s" % L, "%.3f" % min(v["bacc"] for v in lod))
        lom = {k.split("_")[1]: v for k, v in A["lomt"].items() if k.startswith(net)}
        if lom:
            M("LomtMin%s" % L, "%.3f" % min(v["macro_f1"] for v in lom.values())); M("LomtMax%s" % L, "%.3f" % max(v["macro_f1"] for v in lom.values()))
            M("LomtAutoRecMin%s" % L, "%.3f" % min(v["recall"][1] for v in lom.values()))
        for m, T in (("dapper", "Dapper"), ("fedbn", "Fedbn"), ("fedavg", "Fedavg"), ("fedprox", "Fedprox"), ("local", "Local"), ("central", "Central")):
            s = A["cv"].get("%s_%s" % (net, m))
            if s:
                M("CvLo%s%s" % (T, L), "%.3f" % s["ci95"][0]); M("CvHi%s%s" % (T, L), "%.3f" % s["ci95"][1])
                M("CvPS%s%s" % (T, L), "%.3f" % min(v["f1_present"] for v in s["per_site"].values()))
        s = A["cv"].get("%s_dapper" % net)
        if s and "delays" in s:
            dl = s["delays"]
            def g(k, f):
                return dl.get(k, {}).get(f)
            for k, T in (("Ground->Autonomous flight", "Takeoff"), ("Autonomous flight->Ground", "Landing"),
                         ("Autonomous flight->Tele-operation", "ToTele"), ("Tele-operation->Autonomous flight", "FromTele")):
                if g(k, "median") is not None:
                    M("Del%sMed%s" % (T, L), "%.0f" % g(k, "median")); M("Del%sNinety%s" % (T, L), "%.0f" % g(k, "p90"))
                    M("Del%sDet%s" % (T, L), "%.0f" % (100 * g(k, "detected"))); M("Del%sN%s" % (T, L), "%d" % g(k, "n"))
    # companion + generalisation table
    rows = []
    for net, fleet in (("A", "Multirotor"), ("B", "VTOL")):
        base = A["cv"].get("%s_dapper" % net)
        items = [("Original traffic (5-fold CV)", base)]
        items += [("Remote-ID and log upload removed", A["companion"].get("%s_nocomp" % net)),
                  ("\\quad + humanised command cadence", A["companion"].get("%s_nocomp_human" % net))]
        for k in sorted(A["lodo"]):
            if k.startswith(net):
                items.append(("Unseen drone %s (one per site)" % k.split("_")[1], A["lodo"][k]))
        for k in sorted(A["lomt"]):
            if k.startswith(net):
                items.append(("Unseen mission type: %s" % k.split("_")[1], A["lomt"][k]))
        for i, (lab, s) in enumerate(items):
            pre = ("\\multirow{%d}{*}{%s}" % (len(items), fleet)) if i == 0 else ""
            if s is None:
                rows.append("%s & %s & -- & -- & -- & -- & -- \\\\" % (pre, lab)); continue
            rec = ["%.3f" % r if s["windows_per_class"][c] else "--" for c, r in enumerate(s["recall"])]
            rows.append("%s & %s & %.3f & %.3f & %s & %s & %s \\\\" % (pre, lab, s["macro_f1"], s["bacc"], rec[0], rec[1], rec[2]))
        rows.append("\\midrule")
    open(os.path.join(PAPER, "auto_rev1_control_rows.tex"), "w").write("\n".join(rows[:-1]) + "\n")
    # latency table
    L = A.get("latency")
    if L:
        rows = []
        for key, lab in (("jit_reorder", "Random delay U(0,200) ms, reordering allowed"), ("jit_fifo", "Random delay, order preserved")):
            rows.append("%s & +0\\%% & %s & %s & %s & %s & %s \\\\" % (lab, f1d(L["A"][key]["median"]), f1d(L["A"][key]["p95"]),
                                                                 f1d(L["B"][key]["median"]), f1d(L["B"][key]["p95"]),
                                                                 "%.1f / %.1f" % (100 * L["A"][key]["frac_over_100ms"], 100 * L["B"][key]["frac_over_100ms"])))
        for r in (20, 40, 60, 100):
            k = "c%d_c2" % r
            rows.append("Constant rate, %d cells/s & +%d\\%% & %s & %s & %s & %s & %s \\\\" % (
                r, round(100 * L["A"]["cover_overhead"][str(r)]), f1d(L["A"][k]["median"]), f1d(L["A"][k]["p95"]),
                f1d(L["B"][k]["median"]), f1d(L["B"][k]["p95"]), "%.1f / %.1f" % (100 * L["A"][k]["frac_over_100ms"], 100 * L["B"][k]["frac_over_100ms"])))
        open(os.path.join(PAPER, "auto_rev1_latency_rows.tex"), "w").write("\n".join(rows) + "\n")
        for net in ("A", "B"):
            M("LatTwentyMed%s" % net, f1d(L[net]["c20_c2"]["median"])); M("LatTwentyPns%s" % net, f1d(L[net]["c20_c2"]["p95"] / 1000.0, 1))
            M("LatSixtyPns%s" % net, f1d(L[net]["c60_c2"]["p95"])); M("LatSixtyOvh%s" % net, "%d" % round(100 * L[net]["cover_overhead"]["60"]))
            M("LatTeleTwentyPns%s" % net, f1d(L[net]["c20_tele"]["p95"]))
            M("LatJitFifoMed%s" % net, f1d(L[net]["jit_fifo"]["median"]))
    # sorties and windows per site and state (all sorties; under 5-fold CV each is tested once)
    cnt = jl(os.path.join(OUT, "sortie_counts.json"))
    if cnt:
        import glob as _g
        sys_path = os.path.join(HERE)
        import sys as _s
        _s.path.insert(0, sys_path)
        from rev1_counts import windows as _win, PROC as _PROC
        rows = []
        for net, fleet in (("A", "Multirotor"), ("B", "VTOL")):
            per = {}
            for f in sorted(_g.glob(os.path.join(_PROC, "%s*_*.npz" % net))):
                if f.endswith("_pk.npz"):
                    continue
                site, sid, lab, split, mis, _, _ = _win(f, net)
                d = per.setdefault(site, {"drones": 0, "sorties": 0, "ss": [0, 0, 0], "ww": [0, 0, 0]})
                d["drones"] += 1; d["sorties"] += len(np.unique(mis))
                for c in range(3):
                    d["ss"][c] += len(np.unique(mis[lab == c])); d["ww"][c] += int((lab == c).sum())
            for i, (site, d) in enumerate(sorted(per.items())):
                pre = ("\\multirow{%d}{*}{%s}" % (len(per), fleet)) if i == 0 else ""
                rows.append("%s & %s & %d & %d & %d / %d / %d & %s / %s / %s \\\\" % (
                    pre, site, d["drones"], d["sorties"], d["ss"][0], d["ss"][1], d["ss"][2],
                    "{:,}".format(d["ww"][0]), "{:,}".format(d["ww"][1]), "{:,}".format(d["ww"][2])))
            rows.append("\\midrule")
        open(os.path.join(PAPER, "auto_rev1_sortie_rows.tex"), "w").write("\n".join(rows[:-1]) + "\n")
    open(os.path.join(PAPER, "auto_rev1_macros.tex"), "w").write("\n".join(mac) + "\n")


def f1d(x, nd=0):
    return ("%%.%df" % nd) % x


def print_summary(A):
    for k, s in A["cv"].items():
        print("CV %-12s F1 %.3f [%.3f, %.3f] folds %.3f+-%.3f bacc %.3f sorties %s chance %.3f causal5 %.3f" % (
            k, s["macro_f1"], s["ci95"][0], s["ci95"][1], s["fold_mean"], s["fold_sd"], s["bacc"], s["sorties_per_class"], s["chance"]["best"], s["causal_k5"]))
        if "all_windows" in s:
            print("     all windows F1 %.3f, transitions %d acc(last) %s" % (s["all_windows"]["macro_f1"], s["all_windows"]["transition_n"], s["all_windows"]["transition_acc_last"]))
            print("     delays", {k: (v["n"], v["median"], v["p90"], round(v["detected"], 2)) for k, v in s.get("delays", {}).items()})
    for grp in ("companion", "lodo", "lomt"):
        for k, s in A[grp].items():
            print("%-9s %-18s F1 %.3f bacc %.3f recall %s" % (grp, k, s["macro_f1"], s["bacc"], ["%.3f" % r for r in s["recall"]]))
    print("onboarding", A["onboarding"])


if __name__ == "__main__":
    main()
