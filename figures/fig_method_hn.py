"""Figure 2 (methodology) in the HeartNest layout, with NO equations: (a) real input signals,
(b) three network bands drawn as layer pills, (c)-(f) compact word-only detail panels.
Real data throughout: traffic traces and the drone's own altitude log from the frozen snapshot;
attention maps, the class-token attention and the per-phase recall from the trained A_dapper_s1
model.  Output: figures/out/Figure2.pdf/png"""
import os, sys, json, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
PROC, RAW, RUNS, MODELS = [os.path.join(ROOT, p) for p in ('data/proc', 'data/raw', 'results/runs', 'results/models')]
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'code'))
from model import DAPPER
from phases import COPTER_CLASSES

# ------------------------------------------------------------------ palette (HeartNest family) and primitives; y down
INK, MUTE = '#1D1E2C', '#625C68'
B_TOK, B_ENC, B_FED, B_PAN = '#0F4C5C', '#5F0F40', '#3D6B4F', '#7A7268'
C_TEAL, C_TEALF, C_EMB, C_EMBF, C_PLUM, C_PLUMF = '#0F4C5C', '#DCEBEE', '#E36414', '#FCE8DA', '#5F0F40', '#F3E4EC'
C_AMB, C_SLATE, C_ROSE, C_GRN, C_SAND, C_GRNF = '#E9A23B', '#3D405B', '#9E2A2B', '#3D6B4F', '#B9B2A7', '#E3EEE6'
PH = ['#B9B2A7', '#0F4C5C', '#8E3B66', '#3D6B4F', '#5F0F40', '#9E2A2B', '#E36414']   # Ground, Ascent, Hover, Cruise, Orbit, Teleop, Descent
TOK = ['#0F4C5C', '#2C7A7B', '#4FA3A5', '#E36414', '#F08A4B', '#5F0F40', '#8E3B66', '#B9799A']
ATTR = LinearSegmentedColormap.from_list('attr', ['#D9D4CC', '#F2C14E', '#E36414', '#9E2A2B', '#5F0F40'])
plt.rcParams.update({'font.family': 'Arial'})
W, H = 930, 1046
fig = plt.figure(figsize=(7.2, 7.2 * H / W)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal'); ax.axis('off')
LW = 0.95


def rbox(x, y, w, h, fc, ec='none', lw=0.0, r=None, z=2):
    r = min(w, h) * 0.45 if r is None else r
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f'round,pad={r}', fc=fc, ec=ec, lw=lw, zorder=z))


def pill(x, y, w, h, fc, text, tc='white', fs=7.0):
    rbox(x, y, w, h, fc, z=3)
    ax.text(x + w / 2, y + h / 2, text, rotation=90 if h > w else 0, ha='center', va='center', color=tc, fontsize=fs, zorder=4)


def arrow(x0, y0, x1, y1, ms=7, c=INK, lw=LW):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=5, arrowprops=dict(arrowstyle='-|>', lw=lw, color=c, mutation_scale=ms, shrinkA=0, shrinkB=0))


def line(xs, ys, c=INK, lw=LW):
    ax.plot(xs, ys, color=c, lw=lw, zorder=5, solid_capstyle='butt', solid_joinstyle='miter')


def dot(x, y):
    ax.plot(x, y, 'o', ms=3.6, color=INK, zorder=6)


def label(x, y, s, fs=7.5, ha='center', va='center', c=INK, z=7, **kw):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=c, zorder=z, **kw)


def band(x, y, w, h, ec, lw=1.7):
    ax.add_patch(FancyBboxPatch((x + 5, y + 5), w - 10, h - 10, boxstyle='round,pad=5', fc='none', ec=ec, lw=lw, zorder=1))


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x + 5, y + 5), w - 10, h - 10, boxstyle='round,pad=5', fc='#FBF8F3', ec=B_PAN, lw=0.9, zorder=0))
    label(x + 9, y + 15, title, fs=7.6, ha='left', fontweight='bold')


def frame(x, y, w, h, fc='#FFFFFF', ec='#3A3540'):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.6, zorder=3))


def trace(sig, x, y, w, h, c=INK, lw=0.7, lo=None, hi=None, z=4):
    lo = sig.min() if lo is None else lo; hi = sig.max() if hi is None else hi
    xs = x + np.linspace(0, w, len(sig)); ys = y + h - (sig - lo) / (hi - lo + 1e-9) * h
    ax.plot(xs, ys, color=c, lw=lw, zorder=z, solid_joinstyle='round'); return xs, ys


def strip(y_lab, x, y, w, h, z=4):
    n = len(y_lab); bw = w / n
    for k in range(n):
        ax.add_patch(Rectangle((x + k * bw, y), bw + 0.3, h, fc=PH[y_lab[k]] if y_lab[k] >= 0 else '#E8E4DE', ec='none', zorder=z))


def smooth(v, k=5):
    return np.convolve(v, np.ones(k) / k, mode='same')


# ------------------------------------------------------------------ real data (frozen snapshot)
def stretch(z, n=600, step=100):
    y = z['y']; best = None
    for s in range(0, len(y) - n, step):
        seg = y[s:s + n]; k = len(set(seg[seg >= 0].tolist()))
        if best is None or k > best[0]: best = (k, s)
    return best[1]


zA = np.load(os.path.join(PROC, 'A1_11.npz'), allow_pickle=True); zB = np.load(os.path.join(PROC, 'B3_31.npz'), allow_pickle=True)
sA, sB = stretch(zA), stretch(zB)                                   # 60 s stretches with the most phases
upA = smooth(zA['X'].astype(float)[sA:sA + 600, 82]); yA = zA['y'][sA:sA + 600]
upB = smooth(zB['X'].astype(float)[sB:sB + 600, 82]); yB = zB['y'][sB:sB + 600]
gt = pd.read_csv(os.path.join(RAW, 'A1-d1', 'data', 'gt_11.csv'))
tA0 = float(zA['t0']) + sA * 0.1; g = gt[(gt.t >= tA0) & (gt.t < tA0 + 60)]
altA = g.alt.values
summ = json.load(open(os.path.join(PROC, 'summary.json'))); hours, pkts = sum(d['hours'] for d in summ), sum(d['packets'] for d in summ)

ck = torch.load(os.path.join(MODELS, 'At_dapper_s1.pt'), map_location='cpu', weights_only=False)
model = DAPPER(**ck['cfg']); model.load_state_dict(ck['sites']['A1']); model.eval()
OPC = list(ck['classes'])                                            # operational states of the trained model
res = json.load(open(os.path.join(RUNS, 'At_dapper_s1.json')))
cm = np.array(res['confusion'], float); recall = np.diag(cm) / np.clip(cm.sum(1), 1, None)
PH3 = [PH[0], PH[3], PH[5]]

# a real held-out tele-operation window of site A1, read through the two views the model uses
from data import load_network, WindowStore
OP3 = np.array([0, 1, 1, 1, 1, 2, 1])
Xall, clients, V = load_network(PROC, 'A', label_map=OP3)
store = WindowStore(Xall, V, device='cpu')
dA1 = clients['A1']
cand = np.zeros(0, int)
for spl in (2, 1, 0):                       # a held-out tele-operation window if there is one, else validation, else training
    cand = np.where((dA1['split'] == spl) & (dA1['y'] == 2) & (dA1['drone'] == 11))[0]
    if len(cand):
        break
ri = int(cand[len(cand) // 2]) if len(cand) else 0
row = int(dA1['row'][ri]); w0 = int(dA1['start'][ri]); win_true = int(dA1['y'][ri])
bins, pk, mk, st = store.get([row])
Xw = np.expm1(bins[0].numpy())                                       # raw per-bin features of that window
with torch.no_grad():
    xc = model.cal(bins); zb = model.stem(xc.transpose(1, 2)).transpose(1, 2); zb = zb + model.pos[:, :zb.shape[1]]
    stok = model.st_proj(model.cal_st(st)).unsqueeze(1)
    toks = torch.cat([model.cls.expand(1, -1, -1), model.prompt.expand(1, -1, -1), stok, zb], 1)
    blk = model.blocks[0]; n1 = blk.n1(toks); at = blk.attn; Bn, Tn, _ = n1.shape
    q = at.q(n1).view(Bn, Tn, at.h, 2, at.dh).transpose(1, 3); k = at.k(n1).view(Bn, Tn, at.h, 2, at.dh).transpose(1, 3)
    a1 = torch.softmax(q[:, 0] @ k[:, 0].transpose(-1, -2) / at.dh ** 0.5, -1); a2 = torch.softmax(q[:, 1] @ k[:, 1].transpose(-1, -2) / at.dh ** 0.5, -1)
    lam = (torch.exp((at.lq1 * at.lk1).sum()) - torch.exp((at.lq2 * at.lk2).sum()) + at.lambda_init).item()
    A1, A2 = a1[0, 0].numpy(), a2[0, 0].numpy(); AD = A1 - lam * A2
    pred = int(model(bins, pk, mk, st, keep_map=True).argmax(1))
    NF = model.n_fixed
    patch_attn = np.abs(np.mean([b.attn.last_map[0, :, 0, NF:].numpy() for b in model.blocks], axis=(0, 1)))
    cls_attn = np.repeat(patch_attn, 4)[:128]                          # one patch token covers 4 bins
win_up = Xw[:, 82]; yw = zA['y'][w0:w0 + 128] if w0 + 128 <= len(zA['y']) else zA['y'][:128]

# ================================================================== (a) input data
label(8, 16, '(a) Input data', fs=8.6, ha='left', fontweight='bold')
label(8, 36, 'encrypted C2 link\nof a cellular drone,\nseen at the cell-site\nrouter', fs=6.5, ha='left', va='top', c=MUTE, linespacing=1.25)
cards = [('Multirotor fleet', upA, yA, 'ArduCopter · MAVLink 2 in WireGuard', '12 drones · 4 cell sites: 5G, LTE, rural, congested', C_TEAL, 'uplink bytes per 100 ms, 60 s'),
         ('VTOL fleet', upB, yB, 'ArduPlane VTOL · MAVLink 2 in IPsec ESP', '12 drones · 4 cell sites: 5G-SA, LTE, LEO, GEO', C_EMB, 'uplink bytes per 100 ms, 60 s'),
         ('Ground truth', altA, yA, 'drone state log at 5 Hz, never on the network', f'{len(summ)} drones · {hours:.1f} h · {pkts / 1e6:.1f} M packets', C_PLUM, 'altitude of the same 60 s')]
for j, (nm, sg, yl, l1, l2, cc, cap) in enumerate(cards):
    x0 = 112 + 272 * j
    rbox(x0, 4, 264, 142, '#FFFFFF', B_PAN, 0.9, r=8, z=1)
    frame(x0 + 10, 12, 244, 62, fc='#FDFBF8', ec='#CFC8BF')
    for gg in range(1, 6): line([x0 + 10 + 244 * gg / 6] * 2, [12, 74], c='#EFE7DC', lw=0.4)
    trace(sg, x0 + 12, 16, 240, 46, c=cc, lw=0.8)
    strip(yl, x0 + 11, 66, 242, 7)
    label(x0 + 132, 83, cap, fs=5.8, c=MUTE)
    label(x0 + 10, 101, nm, fs=7.6, ha='left', fontweight='bold')
    label(x0 + 10, 115, l1, fs=6.2, ha='left'); label(x0 + 10, 128, l2, fs=6.2, ha='left')
    label(x0 + 10, 140, 'strip: true flight phase', fs=6.0, ha='left', c=MUTE)
for j, nm in enumerate(COPTER_CLASSES):
    yy = 100 + 8.2 * j
    ax.add_patch(Rectangle((10, yy - 3.2), 6.5, 6.5, fc=PH[j], ec='none', zorder=4)); label(19, yy, nm, fs=5.5, ha='left', c=MUTE)
line([248, 248], [146, 156]); line([520, 520], [146, 156]); line([792, 792], [146, 156]); line([792, 60], [156, 156]); arrow(60, 156, 60, 194)

# ================================================================== (b) network: three bands
band(112, 166, 814, 178, B_TOK); band(112, 354, 814, 250, B_ENC); band(112, 614, 814, 148, B_FED)
label(125, 183, '(b) Network', fs=8.6, ha='left', fontweight='bold')
label(914, 183, 'Traffic tokens', fs=10.5, ha='right', fontweight='bold', c=B_TOK)
label(914, 384, 'Gated differential-attention encoder', fs=10.5, ha='right', fontweight='bold', c=B_ENC)
label(914, 748, 'Federation and read-out', fs=10.5, ha='right', fontweight='bold', c=B_FED)
frame(14, 200, 92, 58, fc='#FDFBF8', ec='#CFC8BF'); trace(smooth(win_up, 3), 16, 204, 88, 44, c=C_TEAL, lw=0.7); strip(yw, 16, 250, 88, 5)
label(60, 270, 'one window', fs=6.6); label(60, 283, '128 bins = 12.8 s', fs=6.0, c=MUTE)
YE = 254
arrow(106, YE, 134, YE)
pill(134, 198, 26, 112, C_SLATE, '100 ms bins', fs=6.4)
arrow(160, YE, 176, YE)
pill(176, 198, 26, 112, C_SAND, 'sizes, bytes, IAT', tc=INK, fs=5.8)
arrow(202, YE, 220, YE)
rbox(220, 196, 186, 120, C_TEALF, C_TEAL, 1.0, r=8)
label(313, 208, '128 bins of 90 features', fs=6.8, c=C_TEAL, fontweight='bold')
hb = Xw[::16, :41]
for j in range(8):
    xx = 228 + 22 * j; frame(xx, 218, 20, 62, fc='#FFFFFF', ec=TOK[j])
    hj = np.log1p(hb[j]) / (np.log1p(hb).max() + 1e-9)
    for kk in range(41):
        ax.add_patch(Rectangle((xx + 1 + kk * 18 / 41, 278 - 56 * hj[kk]), 18 / 41, 56 * hj[kk], fc=TOK[j], ec='none', zorder=4))
label(313, 292, 'uplink size histogram of each bin', fs=6.0, c=MUTE); label(313, 305, '41 buckets, 64 B to 1.6 kB', fs=6.0, c=MUTE)
arrow(406, YE, 428, YE)
rbox(428, 198, 150, 112, '#FFFFFF', C_TEAL, 1.0, r=8)
label(503, 210, 'two views, tokenised', fs=6.9, c=C_TEAL, fontweight='bold')
for j, (nm, cc) in enumerate((('site scale', C_TEAL), ('site affine', C_TEAL), ('patches', C_SLATE), ('statistics', C_EMB))):
    pill(438 + 33 * j, 222, 26, 66, cc, nm, tc='white', fs=5.6)
label(503, 299, 'first two pills are personal', fs=5.9, c=MUTE)
arrow(578, YE, 598, YE)
label(608, 254, '+', fs=10, c=INK)
pill(618, 222, 80, 24, C_AMB, 'class + 4 prompts', tc=INK, fs=5.8)
pill(618, 262, 80, 22, C_SAND, 'position', tc=INK, fs=5.8)
arrow(698, YE, 718, YE)
for k_ in range(3):
    ax.add_patch(Rectangle((726 - 5 * k_, 228 + 5 * k_), 74, 44, fc=['#DCEBEE', '#BFDDE1', '#9FCBD0'][k_], ec=C_TEAL, lw=0.7, zorder=3 + k_))
label(763, 293, 'tokens 38 × 128', fs=7.0); label(763, 306, '1 statistics + 32 patch tokens', fs=6.0, c=MUTE)
XT = 880; line([800, XT], [YE, YE]); dot(XT, YE)

# ---------------------------------------------------------------- encoder band
line([XT, XT], [YE, 372]); line([XT, 160], [372, 372]); arrow(160, 372, 160, 396)
pill(146, 396, 28, 150, C_SLATE, 'RMSNorm', fs=6.4)
arrow(174, 470, 196, 470)
rbox(196, 400, 330, 190, C_PLUMF, C_PLUM, 1.0, r=10)
label(361, 413, 'gated differential attention (4 heads)', fs=7.0, c=C_PLUM, fontweight='bold')


def heat(M, x, y, s, title):
    signed = (M < 0).any(); vmax = np.abs(M).max()
    ax.imshow(M, extent=(x, x + s, y + s, y), cmap='RdBu_r' if signed else 'Purples', vmin=-vmax if signed else 0, vmax=vmax, zorder=4, interpolation='nearest', aspect='auto')
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=C_PLUM, lw=0.6, zorder=5)); label(x + s / 2, y + s + 8, title, fs=6.2, c=C_PLUM)


heat(A1, 212, 430, 62, 'map 1'); label(286, 461, '−', fs=11, c=C_PLUM)
heat(A2, 298, 430, 62, 'map 2'); label(372, 461, '=', fs=11, c=C_PLUM)
heat(AD, 384, 430, 62, 'difference')
label(329, 518, 'real maps of head 1 on the window above', fs=5.9, c=MUTE)
for j, (nm, cc) in enumerate((('Q, K, V', C_PLUM), ('two maps', C_PLUM), ('subtract', '#8E3B66'), ('norm', C_SLATE), ('gate', C_EMB))):
    pill(210 + 62 * j, 538, 54, 22, cc, nm, fs=5.6)
    if j < 4: arrow(264 + 62 * j, 549, 272 + 62 * j, 549, ms=4)
label(361, 576, 'the weight shared by both maps cancels; the gate mutes idle heads', fs=5.6, c=MUTE)
arrow(526, 492, 550, 492)
rbox(550, 400, 248, 190, C_EMBF, C_EMB, 1.0, r=10)
label(674, 413, 'feed-forward', fs=7.0, c=C_EMB, fontweight='bold')
for li, (nm, cc, tc) in enumerate((('RMSNorm', C_SLATE, 'white'), ('SwiGLU 128 → 341', C_EMB, 'white'), ('Linear 341 → 128', '#F08A4B', INK))):
    yy = 434 + 44 * li; pill(600, yy, 148, 32, cc, nm, tc=tc, fs=6.4)
    if li < 2: arrow(674, yy + 32, 674, yy + 44, ms=5, c=C_EMB, lw=0.7)
line([590, 574, 574, 590], [450, 450, 582, 582], c=C_EMB, lw=0.7); label(566, 516, 'residual', fs=5.6, c=C_EMB, rotation=90)
label(674, 576, 'dropout 0.2', fs=5.9, c=MUTE)
arrow(798, 492, 826, 492)
pill(826, 436, 28, 112, C_PLUM, '×4 blocks', fs=6.6)
line([854, 886, 886], [492, 492, 650])

# ---------------------------------------------------------------- federation and read-out band
YD = 690
arrow(886, 650, 886, 662); pill(860, 662, 52, 56, C_SLATE, 'read-out', fs=6.0)
arrow(860, YD, 838, YD)
pill(812, 650, 26, 80, C_GRN, 'Linear 256 → C', fs=6.0)
arrow(812, YD, 790, YD)
pill(764, 650, 26, 80, INK, 'softmax', fs=6.6)
arrow(764, YD, 730, YD); label(747, YD - 9, 'phase', fs=6.2)
rbox(556, 640, 172, 102, '#FFFFFF', C_GRN, 0.9, r=8)
label(642, 652, 'state recall (held-out)', fs=6.6, c=C_GRN, fontweight='bold')
for j, (nmp, rv) in enumerate(zip(OPC, recall)):
    yy = 672 + 18 * j
    ax.add_patch(Rectangle((632, yy - 4), 66 * rv, 8, fc=PH3[j], ec='none', zorder=4))
    ax.add_patch(Rectangle((632, yy - 4), 66, 8, fill=False, ec='#B8B2AA', lw=0.5, zorder=4))
    label(628, yy, nmp.replace('Autonomous flight', 'Autonomous'), fs=5.4, ha='right'); label(701, yy, f'{rv:.2f}', fs=5.2, ha='left', c=MUTE)
arrow(556, YD, 522, YD)
rbox(372, 646, 150, 90, C_GRNF, C_GRN, 0.9, r=8)
label(447, 660, 'server', fs=6.8, c=C_GRN, fontweight='bold')
pill(382, 672, 130, 22, C_GRN, 'FedAvg, shared encoder', fs=5.9)
label(447, 708, '20 rounds × 50 local steps', fs=6.0, c=MUTE)
label(447, 722, 'personal part never sent', fs=6.0, c=MUTE)
rbox(124, 640, 238, 102, '#FFFFFF', C_GRN, 0.9, r=8)
label(243, 652, 'cell-site clients', fs=6.6, c=C_GRN, fontweight='bold')
for j, (nm, cc) in enumerate((('5G', C_TEAL), ('LTE', C_TEAL), ('rural', C_EMB), ('congested', C_PLUM))):
    xx = 134 + 56 * j
    rbox(xx, 664, 52, 22, '#FFFFFF', cc, 0.9, r=5, z=3); label(xx + 26, 675, nm, fs=5.9, c=cc)
    arrow(xx + 26, 686, xx + 26, 700, ms=4, c=cc, lw=0.6)
pill(134, 702, 218, 18, C_ROSE, 'personal: calibration + prompts (780)', fs=5.6)
label(243, 732, 'a new site fits only these', fs=5.9, c=MUTE)
line([886, 886], [718, 770]); line([886, 60], [770, 770]); arrow(60, 770, 60, 790)

# ================================================================== detail panels
panel(4, 782, 292, 260, '(c) Outputs and explanations')
frame(16, 804, 276, 80, fc='#FDFBF8', ec='#CFC8BF')
norm = Normalize(0, np.percentile(cls_attn, 99)); bw = 272 / 128; hh = win_up / (np.percentile(win_up, 99) + 1e-9) * 66
for k in range(128):
    ax.add_patch(Rectangle((18 + k * bw, 880 - min(hh[k], 66)), bw, min(hh[k], 66), fc=ATTR(norm(cls_attn[k])), ec='none', zorder=5))
label(154, 894, f'class-token attention on a real {OPC[win_true].lower()} window', fs=6.2, c=MUTE)
label(154, 908, f'predicted {OPC[pred]} · true {OPC[win_true]}', fs=6.0, c=MUTE)
cb = fig.add_axes([0.05, 1 - 950 / H, 0.22, 0.011]); cb.imshow(np.linspace(0, 1, 100)[None], cmap=ATTR, aspect='auto'); cb.set_xticks([]); cb.set_yticks([])
label(154, 930, 'low → high', fs=5.8, c=MUTE)
label(154, 984, f'macro-F1 {res["overall"]["macro_f1"]:.3f} on held-out flights', fs=6.6)
label(154, 998, '3 operational states, chance 0.33', fs=6.0, c=MUTE)
label(154, 1020, 'ROC, confusion and t-SNE in Figures 4-11', fs=6.0, c=MUTE)

panel(300, 782, 268, 260, '(d) Differential attention, one head')
steps = (('split queries and keys into two halves', C_SLATE, 'white'), ('softmax attention map from each half', C_AMB, INK),
         ('subtract map 2, scaled by a learned factor', C_PLUM, 'white'), ('apply to values, RMSNorm per head', C_EMB, 'white'),
         ('multiply by a sigmoid gate of the input', '#F08A4B', INK), ('project the four heads back to 128', C_TEAL, 'white'))
for j, (tx, cc, tc) in enumerate(steps):
    pill(312, 804 + 32 * j, 244, 24, cc, tx, tc=tc, fs=6.0)
    if j < 5: arrow(434, 828 + 32 * j, 434, 836 + 32 * j, ms=5)
label(434, 1004, 'cross-traffic and jitter weight the two maps alike,', fs=6.0, c=MUTE)
label(434, 1018, 'so it cancels; the gate mutes idle heads', fs=6.0, c=MUTE)

panel(572, 782, 172, 260, '(e) Path personalisation')
for j, (tx, cc) in enumerate((('site statistics per view', C_TEAL), ('site affine + 4 prompts', C_PLUM), ('50 local steps, then average', C_GRN))):
    pill(582, 808 + 40 * j, 152, 28, cc, tx, fs=5.9)
    if j < 2: arrow(658, 836 + 40 * j, 658, 848 + 40 * j, ms=5)
for li, (nm, flag) in enumerate((('encoder', 1), ('standardise', 0), ('affine', 0), ('prompts', 0))):
    yy = 936 + 17 * li; label(590, yy, nm, fs=6.0, ha='left', c=MUTE)
    for t in range(4):
        ax.add_patch(Circle((668 + 16 * t, yy), 4.5, fc=C_GRN if flag else '#FFFFFF', ec=C_GRN, lw=0.6, zorder=4))
label(658, 1010, 'filled: averaged across sites', fs=6.0, c=MUTE); label(658, 1024, 'empty: kept at the site', fs=6.0, c=MUTE)

panel(748, 782, 178, 260, '(f) Compared training')
rungs = [('DAPPER', 'shared encoder + personal path', C_PLUM, 'white'), ('FedBN', 'local normalisation', '#8E3B66', 'white'),
         ('FedProx', 'proximal term', C_EMB, 'white'), ('FedAvg', 'plain averaging', C_AMB, INK),
         ('Local / Central', 'no sharing / pooled', C_SLATE, 'white')]
for j, (nm, ds_, cc, tc) in enumerate(rungs):
    yy = 806 + 42 * j
    pill(772, yy, 130, 20, cc, nm, fs=6.6, tc=tc); label(837, yy + 30, ds_, fs=5.9, c=MUTE)
label(837, 1026, 'same data, splits and seeds', fs=6.0, c=MUTE)

fig.savefig(os.path.join(OUT, 'Figure3.pdf')); fig.savefig(os.path.join(OUT, 'Figure3.png'), dpi=500)
print('Figure 3 written; window', w0, 'true', OPC[win_true], 'pred', OPC[pred], 'gt rows', len(g))
