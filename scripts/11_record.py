"""Run a trained fly model on twin snippets (valid / broken) and record neuron activity."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import torch
from flysyntax.model import FlyNet, load_graph, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/v2_real_long/model_epoch16.pt")
ap.add_argument("--mode", choices=["real", "shuffled"], default="real")
ap.add_argument("--rho", type=float, default=4.0)          # only needed to build the model
ap.add_argument("--data", default="v2")
ap.add_argument("--nshow", type=int, default=6000)
ap.add_argument("--max-search", type=int, default=80)
ap.add_argument("--out", default="runs/viz/real.npz")
args = ap.parse_args()

P = Path("data/processed")
rows = [json.loads(l) for l in (P / f"{args.data}_syntax.jsonl").read_text().splitlines()]
d = np.load(P / f"{args.data}_tokens.npz")
ids, lens, split = d["ids"].astype(np.int64), d["lengths"], d["split"]
V = len(json.loads((P / f"{args.data}_vocab.json").read_text()))

W, in_idx, out_idx, _ = load_graph(0, shuffle=(args.mode == "shuffled"), seed=0)
model = FlyNet(W, in_idx, out_idx, V, rho=args.rho)
model.load_state_dict(torch.load(args.ckpt))
model.eval()
print("loaded", args.ckpt, "| mode", args.mode)


@torch.no_grad()
def run(i):
    T = int(lens[i])
    rec = []
    logits = model(torch.from_numpy(ids[i:i + 1, :T]), torch.tensor([T]), record=rec)
    return torch.stack(rec).numpy(), torch.softmax(logits, 1)[0, 1].item()


# which neurons to draw: all descending (output) neurons + a random sample of the rest
pos = np.fromfile(ROOT / "neurons/soma_position.i32", dtype=np.int32).reshape(-1, 3).astype(np.float32)
valid = np.fromfile(ROOT / "neurons/soma_valid.u8", dtype=np.uint8).astype(bool)
sup = np.array(json.loads((ROOT / "neurons/superclass.json").read_text()))
rng = np.random.default_rng(0)
outs = np.array([i for i in out_idx if valid[i]])
others = np.setdiff1d(np.where(valid)[0], outs)
show = np.sort(np.concatenate([outs, rng.choice(others, max(0, args.nshow - len(outs)), replace=False)]))
print("drawing", len(show), "neurons")

# find demo pairs: validation pairs where the model is right on BOTH twins, one per error kind
by_pair = {}
for i, r in enumerate(rows):
    if split[i] == 1 and lens[i] <= 80:
        by_pair.setdefault(r["pair"], []).append(i)
pair_ids = sorted(by_pair)
rng.shuffle(pair_ids)

want = ["moved_brace", "moved_paren", "moved_semicolon", "swap_adjacent"]
chosen, tested, both_right = {}, 0, 0
for p in pair_ids[: args.max_search]:
    iv = next(i for i in by_pair[p] if rows[i]["label"] == 0)
    ib = next(i for i in by_pair[p] if rows[i]["label"] == 1)
    kind = rows[ib]["kind"]
    av, pv = run(iv)
    ab, pb = run(ib)
    tested += 1
    ok = pv < 0.5 and pb >= 0.5
    both_right += ok
    print(f"pair {p:5d} {kind:16s} P(err) valid {pv:.2f}  broken {pb:.2f}  {'both right' if ok else ''}", flush=True)
    if ok and kind not in chosen:
        chosen[kind] = dict(kind=kind, tv=rows[iv]["code"].split(" "), tb=rows[ib]["code"].split(" "),
                            pv=pv, pb=pb, av=av[:, show].astype(np.float16), ab=ab[:, show].astype(np.float16))
    if len(chosen) == len(want):
        break

note = (f"Demo pairs were picked because this model got BOTH twins right. "
        f"Of {tested} validation pairs tested, it got both right in {both_right}. "
        f"Overall validation accuracy of this model is about 73%.")
print(note)
meta = dict(note=note, mode=args.mode, settle=model.settle,
            pairs=[dict(kind=c["kind"], tv=c["tv"], tb=c["tb"], pv=c["pv"], pb=c["pb"]) for c in chosen.values()])
arrays = {}
for k, c in enumerate(chosen.values()):
    arrays[f"av_{k}"], arrays[f"ab_{k}"] = c["av"], c["ab"]
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, meta=json.dumps(meta), pos=pos[show], show=show,
                    cls=np.array([["cb_sensory", "visual_projection", "cb_intrinsic",
                                   "ascending_neuron", "descending_neuron"].index(s) for s in sup[show]]),
                    **arrays)
print("saved", args.out)