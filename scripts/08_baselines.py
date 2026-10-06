import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="v2")
ap.add_argument("--model", choices=["gru", "bow", "bigram"], default="gru",
                help="gru = reads in order; bow = unigram token counts; bigram = counts of adjacent token pairs")
ap.add_argument("--max-len", type=int, default=80)
ap.add_argument("--max-val", type=int, default=300)
ap.add_argument("--epochs", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
P = Path("data/processed")
rows = [json.loads(l) for l in (P / f"{args.data}_syntax.jsonl").read_text().splitlines()]
kinds = np.array([r["kind"] for r in rows])
d = np.load(P / f"{args.data}_tokens.npz")
ids, lens, labels, split = d["ids"].astype(np.int64), d["lengths"], d["labels"], d["split"]
vocab_file = "vocab.json" if args.data == "c" else f"{args.data}_vocab.json"
V = len(json.loads((P / vocab_file).read_text()))

tr = np.where((split == 0) & (lens <= args.max_len))[0]
va = np.where((split == 1) & (lens <= args.max_len))[0][: args.max_val]
epochs = args.epochs or (30 if args.model == "gru" else 60)
print(f"model={args.model} data={args.data}  train {len(tr)}  val {len(va)}  epochs {epochs}")


class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32)
        self.rnn = nn.GRU(32, 64, batch_first=True)
        self.out = nn.Linear(64, 2)

    def forward(self, x, l):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(x), l, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.out(h[-1])


def feats(idx):
    dim = V if args.model == "bow" else V * V
    X = np.zeros((len(idx), dim), dtype=np.float32)
    for r, i in enumerate(idx):
        t = ids[i, :lens[i]]
        key = t if args.model == "bow" else t[:-1] * V + t[1:]
        np.add.at(X[r], key, 1)
    return torch.from_numpy(X) / 10.0


def seq(b):
    T = int(lens[b].max())
    return torch.from_numpy(ids[b, :T]), torch.from_numpy(lens[b]).long()


if args.model == "gru":
    model = GRUNet()
else:
    Xtr, Xva = feats(tr), feats(va)
    model = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Linear(128, 128),
                          nn.ReLU(), nn.Linear(128, 2))
ytr = torch.from_numpy(labels[tr]).long()
yva = torch.from_numpy(labels[va]).long()
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=3e-3)


@torch.no_grad()
def predict():
    model.eval()
    if args.model == "gru":
        out = []
        for s in range(0, len(va), 128):
            x, l = seq(va[s:s + 128])
            out.append(model(x, l).argmax(1))
        pred = torch.cat(out)
    else:
        pred = model(Xva).argmax(1)
    model.train()
    return (pred == yva).numpy()


for ep in range(1, epochs + 1):
    perm = np.random.permutation(len(tr))
    for s in range(0, len(tr), 64):
        p = perm[s:s + 64]
        if args.model == "gru":
            x, l = seq(tr[p])
            out = model(x, l)
        else:
            out = model(Xtr[p])
        loss = F.cross_entropy(out, ytr[p])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 5 == 0:
        print(f"epoch {ep:3d}  val acc {predict().mean():.3f}", flush=True)

ok = predict()
print("per kind:", {str(k): round(float(ok[kinds[va] == k].mean()), 2) for k in sorted(set(kinds[va]))})