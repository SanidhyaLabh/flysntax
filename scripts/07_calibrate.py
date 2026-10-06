import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import torch
from flysyntax.model import FlyNet, load_graph

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["real", "shuffled"], default="real")
ap.add_argument("--subgraph", type=int, default=0)
ap.add_argument("--rhos", type=float, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-len", type=int, default=100)
args = ap.parse_args()

torch.manual_seed(0)
d = np.load("data/processed/c_tokens.npz")
ids, lens, split = d["ids"].astype(np.int64), d["lengths"], d["split"]
vocab = json.loads(Path("data/processed/vocab.json").read_text())

pick = np.where((split == 0) & (lens <= args.max_len))[0][: args.batch]
T = int(lens[pick].max())
x = torch.from_numpy(ids[pick, :T])
l = torch.from_numpy(lens[pick]).long()

W, in_idx, out_idx, _ = load_graph(args.subgraph, shuffle=(args.mode == "shuffled"))
model = FlyNet(W, in_idx, out_idx, len(vocab))
print(f"mode={args.mode} neurons={W.shape[0]} synapses={W.nnz} radius={model.radius:.4f}")
print(f"{'rho':>6} {'gain':>8} {'all|h|':>8} {'out|h|':>8} {'saturated':>10} {'spread':>8} {'sec':>6}")

for rho in args.rhos:
    model.rec_gain.data.fill_(rho / model.radius)
    t0 = time.time()
    with torch.no_grad():
        _, h = model(x, l, return_state=True)
    out = h[model.out_idx]
    sat = (h.abs() > 0.9).float().mean().item()
    spread = (out.std(dim=1).mean() / (out.abs().mean() + 1e-9)).item()
    print(f"{rho:6.1f} {rho / model.radius:8.2f} {h.abs().mean():8.4f} {out.abs().mean():8.4f} "
          f"{sat:10.3f} {spread:8.2f} {time.time() - t0:6.1f}", flush=True)