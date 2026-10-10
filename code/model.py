"""DAPPER encoder: gated differential attention over the three views of an encrypted-traffic window
(packet tokens, 100 ms bin patches, window statistics) with site-personal path calibration and prompts.

Differential attention (Ye et al., Differential Transformer, ICLR 2025):
  DiffAttn(X) = (softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))) V
  lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init,  lambda_init = 0.8 - 0.6 exp(-0.3 (l - 1))
Each head output is RMS-normalised, scaled by (1 - lambda_init) and passed through a head-specific
sigmoid output gate (gated attention) before the output projection.
diff=False -> ordinary softmax attention; gate=False -> no output gate (ablations).

Token sequence: [class] + [path prompts] + [statistics token] + [T/patch bin patches] + [<= NPK packet
tokens].  Padded packet positions are masked out of every attention map and of the pooling; during
training a random subset of the variable tokens is masked as well (token dropout).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d)) if affine else None

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight if self.weight is not None else x


class DiffAttention(nn.Module):
    def __init__(self, d, heads, layer_idx, diff=True, gate=True):
        super().__init__()
        self.h, self.diff, self.gate = heads, diff, gate
        self.dh = d // heads // 2 if diff else d // heads           # query/key width per (sub)map
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if gate:
            self.g = nn.Linear(d, d, bias=True)                     # head-specific sigmoid output gate
        if diff:
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
            for n in ("lq1", "lk1", "lq2", "lk2"):
                setattr(self, n, nn.Parameter(torch.randn(self.dh) * 0.1))
            self.subln = RMSNorm(2 * self.dh, affine=False)
        self.last_map = None

    def forward(self, x, valid, keep_map=False):
        B, T, D = x.shape
        mask4 = valid[:, None, None, :]                                          # True = may be attended
        if not self.diff:
            q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
            k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
            v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
            if keep_map:
                a = torch.softmax((q @ k.transpose(-1, -2) / math.sqrt(self.dh)).masked_fill(~mask4, -1e4), -1)
                self.last_map = a.detach()
                y = a @ v
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask4)
            y = y.transpose(1, 2).reshape(B, T, D)
        else:
            q = self.q(x).view(B, T, self.h, 2, self.dh).transpose(1, 3)          # (B, 2, h, T, dh)
            k = self.k(x).view(B, T, self.h, 2, self.dh).transpose(1, 3)
            v = self.v(x).view(B, T, self.h, 2 * self.dh).transpose(1, 2)        # (B, h, T, 2dh)
            lam = (torch.exp((self.lq1 * self.lk1).sum()) - torch.exp((self.lq2 * self.lk2).sum()) + self.lambda_init)
            if keep_map:
                a1 = torch.softmax((q[:, 0] @ k[:, 0].transpose(-1, -2) / math.sqrt(self.dh)).masked_fill(~mask4, -1e4), -1)
                a2 = torch.softmax((q[:, 1] @ k[:, 1].transpose(-1, -2) / math.sqrt(self.dh)).masked_fill(~mask4, -1e4), -1)
                a = a1 - lam * a2
                self.last_map = a.detach()
                y = a @ v
            else:
                # (softmax(Q1K1) - lambda softmax(Q2K2)) V == SDPA(Q1,K1,V) - lambda SDPA(Q2,K2,V)
                y = F.scaled_dot_product_attention(q[:, 0], k[:, 0], v, attn_mask=mask4)                     - lam * F.scaled_dot_product_attention(q[:, 1], k[:, 1], v, attn_mask=mask4)
            y = (self.subln(y) * (1 - self.lambda_init)).transpose(1, 2).reshape(B, T, D)
        if self.gate:
            y = y * torch.sigmoid(self.g(x))
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, d, mult=8 / 3):
        super().__init__()
        h = int(d * mult)
        self.w1, self.w2, self.w3 = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, d, heads, layer_idx, diff, gate=True, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = DiffAttention(d, heads, layer_idx, diff, gate)
        self.ff = SwiGLU(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid, keep_map=False):
        x = x + self.drop(self.attn(self.n1(x), valid, keep_map))
        return x + self.drop(self.ff(self.n2(x)))


class PathCalibration(nn.Module):
    """Site-personal input calibration: standardise with the site's own traffic statistics,
    then a learnable per-feature affine map (gamma, beta). Never leaves the site."""
    def __init__(self, f):
        super().__init__()
        self.register_buffer("mu", torch.zeros(f))
        self.register_buffer("sd", torch.ones(f))
        self.gamma = nn.Parameter(torch.ones(f))
        self.beta = nn.Parameter(torch.zeros(f))
        self.robust = True

    def forward(self, x):
        # robust standardisation: a floor of 0.1 on the site s.d. and a clip at +-10 keep rare bursts
        # (e.g. log uploads on satellite links) from blowing up the attention
        if self.robust:
            return torch.clamp((x - self.mu) / self.sd.clamp_min(0.1), -10.0, 10.0) * self.gamma + self.beta
        return (x - self.mu) / self.sd * self.gamma + self.beta


class DAPPER(nn.Module):
    def __init__(self, n_feat, n_cls, T=128, d=128, layers=4, heads=4, n_prompt=4, diff=True, gate=True,
                 patch=4, pk_f=4, st_f=40, views=("st", "bins"), dropout=0.2, tok_drop=0.15):
        super().__init__()
        self.views = tuple(views)
        self.cal = PathCalibration(n_feat)
        self.cal_pk = PathCalibration(pk_f)
        self.cal_st = PathCalibration(st_f)
        self.patch = patch
        self.stem = nn.Conv1d(n_feat, d, kernel_size=2 * patch, stride=patch, padding=patch // 2)
        n_tok = (T + 2 * (patch // 2) - 2 * patch) // patch + 1
        self.pos = nn.Parameter(torch.randn(1, n_tok, d) * 0.02)
        self.pk_proj = nn.Linear(pk_f, d)
        self.pk_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.st_proj = nn.Linear(st_f, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.n_prompt = n_prompt
        self.prompt = nn.Parameter(torch.randn(1, n_prompt, d) * 0.02) if n_prompt > 0 else None
        self.blocks = nn.ModuleList([Block(d, heads, i, diff, gate, dropout) for i in range(layers)])
        self.norm = RMSNorm(d)
        self.drop = nn.Dropout(dropout)
        self.tok_drop = tok_drop
        self.head = nn.Linear(2 * d, n_cls)
        self.n_fixed = 1 + n_prompt + (1 if "st" in self.views else 0)

    def embed(self, bins, pk, mask, st, keep_map=False):
        B = bins.shape[0]
        dev = bins.device
        toks = [self.cls.expand(B, -1, -1)]
        if self.prompt is not None:
            toks.append(self.prompt.expand(B, -1, -1))
        valid = [torch.ones(B, len(toks) if len(toks) == 1 else 1 + self.n_prompt, dtype=torch.bool, device=dev)]
        if "st" in self.views:
            toks.append(self.st_proj(self.cal_st(st)).unsqueeze(1)); valid.append(torch.ones(B, 1, dtype=torch.bool, device=dev))
        if "bins" in self.views:
            z = self.stem(self.cal(bins).transpose(1, 2)).transpose(1, 2)
            toks.append(z + self.pos[:, : z.shape[1]]); valid.append(torch.ones(B, z.shape[1], dtype=torch.bool, device=dev))
        if "pk" in self.views:
            toks.append(self.pk_proj(self.cal_pk(pk)) + self.pk_type); valid.append(mask)
        z = torch.cat(toks, 1); v = torch.cat(valid, 1)
        if self.training and self.tok_drop > 0:
            keep = torch.rand(B, z.shape[1], device=dev) >= self.tok_drop
            keep[:, : self.n_fixed] = True
            v = v & keep
        z = self.drop(z)
        for b in self.blocks:
            z = b(z, v, keep_map)
        z = self.norm(z)
        w = v.clone(); w[:, : self.n_fixed] = False
        w = w.float().unsqueeze(-1)
        pooled = (z * w).sum(1) / w.sum(1).clamp_min(1.0)
        return torch.cat([z[:, 0], pooled], -1)                        # class token + masked mean of the variable tokens

    def forward(self, bins, pk, mask, st, keep_map=False):
        return self.head(self.drop(self.embed(bins, pk, mask, st, keep_map)))


def personal_keys(model, scheme):
    """Parameter/buffer names that stay at the site under each federated scheme."""
    keys = []
    for n, _ in list(model.named_parameters()) + list(model.named_buffers()):
        is_cal = n.startswith("cal")
        is_stat = is_cal and (n.endswith(".mu") or n.endswith(".sd"))
        if scheme == "dapper" and (is_cal or n == "prompt"):
            keys.append(n)
        elif scheme == "dapper_cal" and is_cal:
            keys.append(n)
        elif scheme == "dapper_prompt" and (n == "prompt" or is_stat):
            keys.append(n)
        elif scheme == "fedbn" and (is_stat or n.endswith(".n1.weight") or n.endswith(".n2.weight") or n == "norm.weight"):
            keys.append(n)                       # FedBN: every normalisation layer stays local
    return keys
