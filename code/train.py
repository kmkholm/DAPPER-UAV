"""Federated training of the DAPPER flight-phase classifier across cell-site observers.

Usage:
  python train.py --net A --method dapper --seed 1

methods: local | central | fedavg | fedprox | fedbn | dapper | dapper_cal | dapper_prompt | dapper_softmax
  dapper          differential-attention encoder, site-personal calibration + site prompts
  dapper_cal      only the input calibration is kept site-local (prompts off)
  dapper_prompt   only the site prompts are kept site-local (calibration shared)
  dapper_softmax  same recipe with plain softmax attention instead of the differential map
  fedavg/fedprox/fedbn/local/central share the differential-attention encoder
Option --holdout SITE evaluates a left-out site (zero-shot, then after local calibration).
"""
import argparse, copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_network, WindowStore, stats
from model import DAPPER, personal_keys
from defenses import DEFENSES
from phases import COPTER_CLASSES, PLANE_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="A")
ap.add_argument("--method", default="dapper")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--rounds", type=int, default=40)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--mu", type=float, default=0.01)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--holdout", default="")
ap.add_argument("--proc", default=os.path.join(os.path.dirname(__file__), "..", "data", "proc"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "runs"))
ap.add_argument("--save_preds", action="store_true")
ap.add_argument("--save_model", action="store_true")
ap.add_argument("--coarse", action="store_true", help="alias for --task op4")
ap.add_argument("--task", default="", help="fine (kinematic phases) | op4 (ground / stationary / transit / tele-operation) | op3 (ground / autonomous flight / tele-operation)")
ap.add_argument("--tag", default="", help="extra suffix for the result file name")
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--tok_drop", type=float, default=0.15)
ap.add_argument("--defense", default="none", help="packet-level traffic-shaping defence: pad256 | pad512 | jitter200 | const")
A = ap.parse_args()
A.task = A.task or ("op4" if A.coarse else "fine")
A.coarse = A.task != "fine"
if A.defense != "none":
    A.tag += "_def" + A.defense

torch.manual_seed(A.seed); np.random.seed(A.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TASK_MAP = {"op4": {"A": [0, 2, 1, 2, 1, 3, 2], "B": [0, 2, 2, 2, 1, 3]}, "op3": {"A": [0, 1, 1, 1, 1, 2, 1], "B": [0, 1, 1, 1, 1, 2]}}
TASK_CLASSES = {"op4": ["Ground", "Stationary", "Transit", "Tele-operation"], "op3": ["Ground", "Autonomous flight", "Tele-operation"]}
TASK_TAG = {"fine": "", "op4": "c", "op3": "t"}
COARSE_MAP = np.array(TASK_MAP[A.task][A.net]) if A.coarse else None
CLASSES = TASK_CLASSES[A.task] if A.coarse else (COPTER_CLASSES if A.net == "A" else PLANE_CLASSES)
NC = len(CLASSES)
X, clients, VIEWS = load_network(A.proc, A.net, T=A.T, transform=DEFENSES.get(A.defense), label_map=COARSE_MAP if A.coarse else None)
TAG = A.net + TASK_TAG[A.task]
STORE = WindowStore(X, VIEWS, T=A.T, device=DEV)
F_IN = X.shape[1]
SITES = sorted(clients)
TRAIN_SITES = [s for s in SITES if s != A.holdout]

DIFF = A.method != "dapper_softmax"
GATE = A.method != "dapper_nogate"
NP_ = 4 if A.method in ("dapper", "dapper_prompt", "dapper_softmax", "dapper_nogate", "dapper_pk", "dapper_nost", "dapper_nobins") else 0
# DAPPER reads the statistics token + bin patches; packet tokens were evaluated and lower held-out accuracy (ablation dapper_pk)
VIEWS_ = {"dapper_pk": ("st", "bins", "pk"), "dapper_nost": ("bins",), "dapper_nobins": ("st",)}.get(A.method, ("st", "bins"))
SCHEME = {"dapper": "dapper", "dapper_softmax": "dapper", "dapper_nogate": "dapper", "dapper_pk": "dapper", "dapper_nost": "dapper",
          "dapper_nobins": "dapper", "dapper_cal": "dapper_cal",
          "dapper_prompt": "dapper_prompt", "fedbn": "fedbn"}.get(A.method, "none")
LS = 0.05                                           # label smoothing
NOISE = 0.05                                        # training-time Gaussian noise on the log features
WD = 0.05                                           # AdamW weight decay


def make_model():
    return DAPPER(F_IN, NC, T=A.T, n_prompt=NP_, diff=DIFF, gate=GATE, views=VIEWS_, dropout=A.dropout, tok_drop=A.tok_drop).to(DEV)


def lr_at(frac):
    """Cosine schedule from A.lr down to 0.1 A.lr over the training budget (frac in [0, 1])."""
    return A.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(frac, 1.0))))


def split_idx(site, split):
    d = clients[site]
    m = d["split"] == split
    return d["row"][m], d["y"][m]


def set_class_weights(model):
    ytr = np.concatenate([split_idx(s, 0)[1] for s in TRAIN_SITES])
    cnt = np.bincount(ytr, minlength=NC).astype(np.float64)
    w = (cnt.sum() / (NC * np.clip(cnt, 1, None)))
    return torch.tensor(w, dtype=torch.float32, device=DEV)


def init_calibration(model, site):
    """Fill the frozen calibration buffers from a site's own training windows."""
    st, _ = split_idx(site, 0)
    if len(st) == 0:
        return
    set_calibration(model, stats(STORE, st, rng=np.random.default_rng(A.seed)))


def set_calibration(model, S):
    with torch.no_grad():
        for name, mod in (("bins", model.cal), ("pk", model.cal_pk), ("st", model.cal_st)):
            mod.mu.copy_(S[name][0]); mod.sd.copy_(S[name][1])


def noisy(xb):
    """Training-time Gaussian noise on the continuous views (bins and statistics)."""
    b, pk, mk, st = xb
    return (b + NOISE * torch.randn_like(b), pk, mk, st + NOISE * torch.randn_like(st))


def batch(site, split, bs, rng):
    st, y = split_idx(site, split)
    sel = rng.integers(0, len(st), size=bs)
    xb = STORE.get(st[sel])
    yb = torch.as_tensor(y[sel], device=DEV)
    return xb, yb


def local_steps(model, site, steps, wclass, rng, proximal=None, lr=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr or A.lr, weight_decay=WD)
    model.train()
    for _ in range(steps):
        xb, yb = batch(site, 0, A.bs, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            logit = model(*noisy(xb))
        loss = F.cross_entropy(logit.float(), yb, weight=wclass, label_smoothing=LS)
        if proximal is not None:
            reg = 0.0
            for p, g in zip(model.parameters(), proximal):
                reg = reg + ((p - g) ** 2).sum()
            loss = loss + 0.5 * A.mu * reg
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def evaluate(model, site, split=2):
    st, y = split_idx(site, split)
    if len(st) == 0:
        return None
    model.eval()
    preds, probs = [], []
    for i in range(0, len(st), 512):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            logit = model(*STORE.get(st[i:i + 512]))
        p = torch.softmax(logit.float(), -1)
        probs.append(p.cpu().numpy()); preds.append(p.argmax(1).cpu().numpy())
    yp = np.concatenate(preds); pr = np.concatenate(probs)
    return dict(y=y, yp=yp, prob=pr,
                f1=float(f1_score(y, yp, average="macro", labels=range(NC), zero_division=0)),
                bacc=float(balanced_accuracy_score(y, yp)),
                acc=float((y == yp).mean()))


# ---------------------------------------------------------------- federation
def avg_state(states, keep_local):
    """FedAvg over the shared keys; local keys are left untouched per site."""
    out = copy.deepcopy(states[0])
    for k in out:
        if k in keep_local:
            continue
        if out[k].dtype.is_floating_point:
            out[k] = sum(s[k] for s in states) / len(states)
    return out


def run_federated():
    rng = np.random.default_rng(A.seed)
    global_model = make_model()
    wclass = set_class_weights(global_model)
    site_models = {s: make_model() for s in TRAIN_SITES}
    for s, m in site_models.items():
        m.load_state_dict(global_model.state_dict())
        init_calibration(m, s)
    keep_local = set()
    if SCHEME != "none":
        keep_local = set(personal_keys(global_model, SCHEME))
    hist = []
    for r in range(A.rounds):
        gstate = global_model.state_dict()
        proximal = [p.detach().clone() for p in global_model.parameters()] if A.method == "fedprox" else None
        states = []
        for s in TRAIN_SITES:
            m = site_models[s]
            shared = {k: v for k, v in gstate.items() if k not in keep_local}
            m.load_state_dict(shared, strict=False)
            local_steps(m, s, A.steps, wclass, rng, proximal, lr=lr_at(r / A.rounds))
            states.append({k: v.detach().clone() for k, v in m.state_dict().items()})
        new = avg_state(states, keep_local)
        global_model.load_state_dict(new)
        if (r + 1) % 5 == 0 or r == A.rounds - 1:
            mf = []
            for s in TRAIN_SITES:
                site_models[s].load_state_dict({k: v for k, v in global_model.state_dict().items()
                                                if k not in keep_local}, strict=False)
                e = evaluate(site_models[s], s, split=1)
                if e:
                    mf.append(e["f1"])
            hist.append(dict(round=r + 1, val_macro_f1=float(np.mean(mf))))
            print("round %2d  val macroF1 %.4f" % (r + 1, np.mean(mf)), flush=True)
    return global_model, site_models, keep_local, hist


def run_local():
    rng = np.random.default_rng(A.seed)
    wclass = set_class_weights(make_model())
    site_models = {}
    for s in TRAIN_SITES:
        m = make_model()
        init_calibration(m, s)
        for r in range(A.rounds):
            local_steps(m, s, A.steps, wclass, rng, lr=lr_at(r / A.rounds))
        site_models[s] = m
    return None, site_models, set(), []


def run_central():
    """Pooled upper bound: one model on all sites with the same total step budget as the federation."""
    rng = np.random.default_rng(A.seed)
    m = make_model()
    allst = np.concatenate([split_idx(s, 0)[0] for s in TRAIN_SITES])
    set_calibration(m, stats(STORE, allst, rng=rng))
    wclass = set_class_weights(m)
    ally = np.concatenate([split_idx(s, 0)[1] for s in TRAIN_SITES])
    opt = torch.optim.AdamW(m.parameters(), lr=A.lr, weight_decay=WD)
    m.train()
    total = A.rounds * A.steps * len(TRAIN_SITES)
    for i in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(i / total)
        sel = rng.integers(0, len(allst), size=A.bs)
        xb = STORE.get(allst[sel]); yb = torch.as_tensor(ally[sel], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            logit = m(*noisy(xb))
        loss = F.cross_entropy(logit.float(), yb, weight=wclass, label_smoothing=LS)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m, {s: m for s in TRAIN_SITES}, set(), []


# ---------------------------------------------------------------- new-site adaptation
def onboard(global_model, keep_local, site, shots, rng):
    """Few-shot onboarding of a left-out site: freeze the shared backbone, fit only the
    site-personal parameters (calibration + prompts) on `shots` labelled windows."""
    m = make_model()
    m.load_state_dict(global_model.state_dict())
    init_calibration(m, site)
    local_keys = set(personal_keys(m, SCHEME)) if SCHEME != "none" else set()
    if not local_keys:                       # methods without personal params: adapt the head only
        local_keys = {"head.weight", "head.bias"}
    for n, p in m.named_parameters():
        p.requires_grad = n in local_keys
    st, y = split_idx(site, 0)
    if shots > 0:
        sel = rng.choice(len(st), size=min(shots, len(st)), replace=False)
        st, y = st[sel], y[sel]
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
        m.train()
        for _ in range(200):
            b = rng.integers(0, len(st), size=min(A.bs, len(st)))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                logit = m(*STORE.get(st[b]))
            loss = F.cross_entropy(logit.float(), torch.as_tensor(y[b], device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(m, site, split=2)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    runner = {"local": run_local, "central": run_central}.get(A.method)
    if runner is None:
        gm, site_models, keep_local, hist = run_federated()
    else:
        gm, site_models, keep_local, hist = runner()

    per_site = {}
    agg = {"y": [], "yp": [], "prob": []}
    for s in TRAIN_SITES:
        e = evaluate(site_models[s], s, split=2)
        if not e:
            continue
        per_site[s] = dict(f1=e["f1"], bacc=e["bacc"], acc=e["acc"], n=int(len(e["y"])))
        for k in agg:
            agg[k].append(e[k])
    y = np.concatenate(agg["y"]); yp = np.concatenate(agg["yp"]); prob = np.concatenate(agg["prob"])
    tr = [evaluate(site_models[s], s, split=0) for s in TRAIN_SITES]
    overall = dict(macro_f1=float(f1_score(y, yp, average="macro", labels=range(NC), zero_division=0)),
                   bacc=float(balanced_accuracy_score(y, yp)), acc=float((y == yp).mean()),
                   site_mean_f1=float(np.mean([v["f1"] for v in per_site.values()])),
                   train_f1=float(np.mean([e["f1"] for e in tr if e])))
    cm = confusion_matrix(y, yp, labels=range(NC)).tolist()

    onb = {}
    if A.holdout and gm is not None:
        rng = np.random.default_rng(A.seed + 1)
        for shots in (0, 5, 10, 20, 50):
            e = onboard(gm, keep_local, A.holdout, shots, rng)
            if e:
                onb[shots] = dict(f1=e["f1"], bacc=e["bacc"], acc=e["acc"])
                print("holdout %s  shots %2d  macroF1 %.4f" % (A.holdout, shots, e["f1"]), flush=True)

    os.makedirs(A.out, exist_ok=True)
    name = "%s_%s_s%d%s%s" % (TAG, A.method, A.seed, ("_ho" + A.holdout) if A.holdout else "", A.tag)
    res = dict(net=A.net, task=A.task, method=A.method, seed=A.seed, holdout=A.holdout, classes=CLASSES,
               rounds=A.rounds, steps=A.steps, T=A.T, n_feat=F_IN, train_sites=TRAIN_SITES,
               overall=overall, per_site=per_site, confusion=cm, history=hist, onboard=onb,
               defense=A.defense, overhead=float(VIEWS.get("overhead", 0.0)),
               minutes=round((time.time() - t0) / 60, 2))
    json.dump(res, open(os.path.join(A.out, name + ".json"), "w"), indent=1)
    if A.save_preds:
        np.savez_compressed(os.path.join(A.out, name + "_pred.npz"), y=y, yp=yp, prob=prob.astype(np.float16))
    if A.save_model:
        md = os.path.join(A.out, "..", "models"); os.makedirs(md, exist_ok=True)
        torch.save(dict(cfg=dict(n_feat=F_IN, n_cls=NC, T=A.T, n_prompt=NP_, diff=DIFF, gate=GATE, views=VIEWS_),
                        sites={s: site_models[s].state_dict() for s in TRAIN_SITES},
                        classes=CLASSES, net=A.net, method=A.method, task=A.task, coarse=A.coarse, robust_cal=True), os.path.join(md, name + ".pt"))
    print("SAVED", name, "macroF1 %.4f  bacc %.4f  trainF1 %.4f" % (overall["macro_f1"], overall["bacc"], overall["train_f1"]),
          "(%.1f min)" % res["minutes"], flush=True)


if __name__ == "__main__":
    main()
