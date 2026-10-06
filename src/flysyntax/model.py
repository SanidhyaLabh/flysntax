import json
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore", message=".*[Ss]parse.*")

ROOT = Path("data/raw/fly-connectome-49k")


def load_graph(subgraph=0, shuffle=False, seed=0):
    """Returns (W, in_idx, out_idx, kept). W[i, j] = weight of synapse j -> i."""
    g = ROOT / "graph"
    offsets = np.fromfile(g / "edges_offsets.i32", dtype=np.int32).astype(np.int64)
    source = np.fromfile(g / "edges_source.u16", dtype=np.uint16).astype(np.int64)
    weight = np.fromfile(g / "edges_weight.f32", dtype=np.float32)
    N = len(offsets) - 1
    sup = np.array(json.loads((ROOT / "neurons/superclass.json").read_text()))
    in_all = np.fromfile(ROOT / "interface/in_index.i32", dtype=np.int32).astype(np.int64)
    rng = np.random.default_rng(seed)
    if shuffle:
        # control: same fan-in per neuron, same weights/signs, same out-degrees,
        # but every connection's source neuron is reassigned at random
        source = source[rng.permutation(len(source))]
    W = csr_matrix((weight, source, offsets), shape=(N, N))
    W.sum_duplicates()
    kept = np.arange(N)
    if subgraph and subgraph < N:
        kept = np.sort(rng.choice(N, subgraph, replace=False))
        W = W[kept][:, kept].tocsr()
        sup = sup[kept]
    is_in = np.zeros(N, dtype=bool)
    is_in[in_all] = True
    in_idx = np.where(is_in[kept])[0]
    out_idx = np.where(sup == "descending_neuron")[0]
    return W.tocsr(), in_idx, out_idx, kept


def to_torch_csr(M):
    M = M.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(M.indptr.astype(np.int64)),
        torch.from_numpy(M.indices.astype(np.int64)),
        torch.from_numpy(M.data.astype(np.float32)),
        size=M.shape,
    )


def spectral_radius(W, iters=60, seed=0):
    """Estimate the average per-step growth of activity through the wiring."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(W.shape[0]).astype(np.float32)
    x /= np.linalg.norm(x)
    total, count = 0.0, 0
    for i in range(iters):
        x = W @ x
        n = np.linalg.norm(x)
        if n == 0:
            return 1e-8
        if i >= iters // 2:                  # skip the warm-up transient
            total += np.log(n)
            count += 1
        x = x / n
    return float(np.exp(total / count))


class SpMM(torch.autograd.Function):
    """y = W @ x with the wiring frozen; backward uses the transposed wiring."""

    @staticmethod
    def forward(ctx, x, W, Wt):
        ctx.Wt = Wt
        return W @ x

    @staticmethod
    def backward(ctx, grad):
        return ctx.Wt @ grad.contiguous(), None, None


class FlyNet(nn.Module):
    def __init__(self, W, in_idx, out_idx, vocab_size, settle=8, rho=1.2, leak_init=0.0):
        super().__init__()
        self.N = W.shape[0]
        self.settle = settle
        self.W = to_torch_csr(W)
        self.Wt = to_torch_csr(W.T)
        self.register_buffer("in_idx", torch.from_numpy(in_idx).long())
        self.register_buffer("out_idx", torch.from_numpy(out_idx).long())
        self.embed = nn.Embedding(vocab_size, len(in_idx))
        nn.init.normal_(self.embed.weight, std=0.5)
        # one learned value per neuron (frozen wiring + frozen synapse signs)
        self.radius = spectral_radius(W)
        self.rec_gain = nn.Parameter(torch.full((self.N,), rho / self.radius))
        self.bias = nn.Parameter(torch.zeros(self.N))
        self.leak = nn.Parameter(torch.full((self.N,), float(leak_init)))
        n_out = len(out_idx)
        # readout sees: mean over time, max over time, and final state
        self.norm = nn.LayerNorm(3 * n_out, eps=1e-6)
        self.readout = nn.Linear(3 * n_out, 2)

    def forward(self, ids, lengths, return_state=False, record=None):
        B, T = ids.shape
        n_out = len(self.out_idx)
        h = torch.zeros(self.N, B)
        alpha = torch.sigmoid(self.leak)[:, None]
        gain = self.rec_gain[:, None]
        bias = self.bias[:, None]
        s_sum = torch.zeros(n_out, B)
        s_max = torch.full((n_out, B), -1.0)
        s_cnt = torch.zeros(1, B)
        for t in range(T + self.settle):
            pre = gain * SpMM.apply(h, self.W, self.Wt) + bias
            if t < T:
                emb = self.embed(ids[:, t]).T.contiguous()
                pre = pre + torch.zeros(self.N, B).index_copy(0, self.in_idx, emb)
            new = (1 - alpha) * h + alpha * torch.tanh(pre)
            if t < T:
                active = (t < lengths)[None, :]       # frozen on padding
                new = torch.where(active, new, h)
            h = new
            if record is not None:
                record.append(h[:, 0].detach().clone())
            ho = h[self.out_idx]
            if t < T:
                s_sum = s_sum + ho * active.float()
                s_max = torch.where(active, torch.maximum(s_max, ho), s_max)
                s_cnt = s_cnt + active.float()
            else:
                s_sum = s_sum + ho
                s_max = torch.maximum(s_max, ho)
                s_cnt = s_cnt + 1.0
        feat = torch.cat([s_sum / s_cnt, s_max, h[self.out_idx]], dim=0)
        logits = self.readout(self.norm(feat.T))
        return (logits, h) if return_state else logits