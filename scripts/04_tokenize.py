import json, re
from collections import Counter
from pathlib import Path
import numpy as np

rows = [json.loads(l) for l in Path("data/processed/c_syntax.jsonl").read_text().splitlines()]

FNAME = re.compile(r"^(add|sub|mul|calc|step|update|score|mix)\d$")
TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|\+\+|--|[^\sA-Za-z_\d]")

def tokenize(code):
    out = []
    for t in TOKEN.findall(code):
        if t[0].isdigit():
            out.append("<num>")        # numbers don't affect syntax
        elif FNAME.match(t):
            out.append("<fname>")      # all function names look the same
        else:
            out.append(t)
    return out

toks = [tokenize(r["code"]) for r in rows]

train_vocab = Counter()
for r, t in zip(rows, toks):
    if r["split"] == "train":
        train_vocab.update(t)

vocab = ["<pad>", "<unk>"] + sorted(train_vocab)
stoi = {w: i for i, w in enumerate(vocab)}

lens = np.array([len(t) for t in toks])
L = int(lens.max())
ids = np.zeros((len(rows), L), dtype=np.int16)
for i, t in enumerate(toks):
    ids[i, :len(t)] = [stoi.get(w, 1) for w in t]

split_id = {"train": 0, "val": 1, "test": 2}
labels = np.array([r["label"] for r in rows], dtype=np.int64)
split = np.array([split_id[r["split"]] for r in rows], dtype=np.int64)

np.savez("data/processed/c_tokens.npz", ids=ids, lengths=lens, labels=labels, split=split)
Path("data/processed/vocab.json").write_text(json.dumps(vocab))

print("vocab size:", len(vocab))
print("tokens per snippet: mean %.0f, median %.0f, max %d" % (lens.mean(), np.median(lens), L))
print("\nexample tokens:", toks[0][:30], "...")

print("\nSnippets containing an unseen token (<unk>), by kind:")
unk = (ids == 1).any(axis=1)
by = {}
for r, u in zip(rows, unk):
    by.setdefault(r["kind"], [0, 0])
    by[r["kind"]][0] += 1
    by[r["kind"]][1] += int(u)
for k, (n, u) in sorted(by.items()):
    print(f"  {k:30s} {u:4d} / {n}")