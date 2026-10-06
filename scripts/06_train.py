import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import torch
import torch.nn.functional as F
from flysyntax.model import FlyNet, load_graph

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="v2")
ap.add_argument("--mode", choices=["real", "shuffled"], default="real")
ap.add_argument("--subgraph", type=int, default=0, help="0 = full 49k graph")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max-len", type=int, default=80)
ap.add_argument("--max-train", type=int, default=0)
ap.add_argument("--max-val", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--settle", type=int, default=8)
ap.add_argument("--rho", type=float, default=4.0)
ap.add_argument("--leak", type=float, default=-1.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/fly")
args = ap.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
P = Path("data/processed")
rows = [json.loads(l) for l in (P / f"{args.data}_syntax.jsonl").read_text().splitlines()]
kinds = np.array([r["kind"] for r in rows])
d = np.load(P / f"{args.data}_tokens.npz")
ids, lens, labels, split = d["ids"].astype(np.int64), d["lengths"], d["labels"], d["split"]
vocab_file = "vocab.json" if args.data == "c" else f"{args.data}_vocab.json"
vocab = json.loads((P / vocab_file).read_text())

ok = lens <= args.max_len
tr = np.where((split == 0) & ok)[0]
va = np.where((split == 1) & ok)[0]
if args.max_train: tr = tr[: args.max_train]
if args.max_val: va = va[: args.max_val]
print(f"data={args.data}  train {len(tr)}  val {len(va)}  (snippets longer than {args.max_len} tokens skipped)")

print("loading graph ...")
W, in_idx, out_idx, _ = load_graph(args.subgraph, shuffle=(args.mode == "shuffled"), seed=args.seed)
print(f"mode={args.mode}  neurons={W.shape[0]}  synapses={W.nnz}  inputs={len(in_idx)}  outputs={len(out_idx)}")
model = FlyNet(W, in_idx, out_idx, len(vocab), settle=args.settle, rho=args.rho, leak_init=args.leak)
print(f"wiring spectral radius {model.radius:.4f} -> starting gain {args.rho / model.radius:.2f}")
params = [p for p in model.parameters()]
print("trainable parameters:", sum(p.numel() for p in params))
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
total_steps = args.epochs * ((len(tr) + args.batch - 1) // args.batch)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)


def make_batches(idx, shuffle):
    key = lens[idx] + (np.random.uniform(0, 8, len(idx)) if shuffle else 0)
    order = idx[np.argsort(key)]
    chunks = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
    if shuffle: random.shuffle(chunks)
    return chunks


def tensors(b):
    T = int(lens[b].max())
    return (torch.from_numpy(ids[b, :T]), torch.from_numpy(lens[b]).long(),
            torch.from_numpy(labels[b]).long())


@torch.no_grad()
def evaluate(idx):
    model.eval()
    correct = np.zeros(len(rows), dtype=bool)
    for b in make_batches(idx, False):
        x, l, y = tensors(b)
        correct[b] = (model(x, l).argmax(1) == y).numpy()
    model.train()
    acc = correct[idx].mean()
    per = {str(k): round(float(correct[idx][kinds[idx] == k].mean()), 2)
           for k in sorted(set(kinds[idx]))}
    return acc, per


Path(args.out).mkdir(parents=True, exist_ok=True)
log = []
for ep in range(args.epochs):
    t0, run_loss, run_acc, n = time.time(), 0.0, 0.0, 0
    for i, b in enumerate(make_batches(tr, True)):
        x, l, y = tensors(b)
        logits, h = model(x, l, return_state=True)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run_loss += loss.item(); run_acc += (logits.argmax(1) == y).float().mean().item(); n += 1
        if ep == 0 and i == 0:
            print(f"[diag] mean |activity| over all neurons {h.abs().mean():.4f}, "
                  f"over output neurons {h[model.out_idx].abs().mean():.4f}  "
                  "(near 0 = signal dying, near 1 = saturated)")
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"ep {ep+1} batch {i+1}  loss {run_loss/n:.3f}  acc {run_acc/n:.3f}  {el/(i+1):.1f}s/batch", flush=True)
            run_loss, run_acc, n = 0.0, 0.0, 0
    acc, per = evaluate(va)
    print(f"== epoch {ep+1}: val acc {acc:.3f}\n   per kind {per}", flush=True)
    log.append({"epoch": ep + 1, "val_acc": float(acc), "per_kind": per})
    torch.save({k: v for k, v in model.state_dict().items()}, f"{args.out}/model.pt")
    Path(f"{args.out}/log.json").write_text(json.dumps(log, indent=1))