"""Dataset v2: every broken snippet has a valid twin with IDENTICAL tokens in a
different order, so counting tokens cannot beat 50%. gcc is the ground truth."""
import json, random, re, subprocess, tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from tqdm import tqdm

random.seed(1)
N_PAIRS = 3600
OUT = Path("data/processed")

VARS = ["a", "b", "c", "x", "y", "z", "n", "sum", "total", "count", "val"]
FUNCS = ["add", "sub", "mul", "calc", "step", "update", "score", "mix"]
TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|\+\+|--|[^\sA-Za-z_\d]")
FNAME = re.compile(r"^(add|sub|mul|calc|step|update|score|mix)\d$")


# ---------- generate valid C (same generator as dataset v1) ----------
def arith(vs, d=0):
    if d >= 2 or random.random() < 0.45:
        return random.choice(vs) if random.random() < 0.7 else str(random.randint(0, 99))
    return f"{arith(vs, d + 1)} {random.choice('+-*')} {arith(vs, d + 1)}"

def cond(vs):
    return f"{arith(vs, 1)} {random.choice(['<', '>', '==', '!=', '<=', '>='])} {arith(vs, 1)}"

def stmt(vs, depth=0):
    r = random.random()
    if r < 0.35 or depth >= 2:
        return f"{random.choice(vs)} = {arith(vs)};"
    if r < 0.6:
        s = f"if ({cond(vs)}) {{ {stmt(vs, depth + 1)} }}"
        if random.random() < 0.3:
            s += f" else {{ {stmt(vs, depth + 1)} }}"
        return s
    if r < 0.8:
        return f"for (int k = 0; k < {random.randint(2, 20)}; k++) {{ {stmt(vs + ['k'], depth + 1)} }}"
    return f"while ({cond(vs)}) {{ {stmt(vs, depth + 1)} }}"

def function(name):
    params = random.sample(VARS, random.randint(1, 3))
    locs = random.sample([v for v in VARS if v not in params], random.randint(1, 2))
    vs = params + locs
    lines = [f"int {l} = {arith(params)};" for l in locs]
    lines += [stmt(vs) for _ in range(random.randint(1, 3))]
    lines.append(f"return {arith(vs)};")
    sig = ", ".join(f"int {p}" for p in params)
    return f"int {name}({sig}) {{\n" + "\n".join("    " + l for l in lines) + "\n}\n"

def make_valid():
    names = random.sample(FUNCS, 2 if random.random() < 0.2 else 1)
    return "\n".join(function(f"{n}{random.randint(0, 9)}") for n in names)


# ---------- count-preserving mutations on the token list ----------
def move(tokens, chars, kind):
    pos = [i for i, t in enumerate(tokens) if t in chars]
    if not pos:
        return None
    i = random.choice(pos)
    t = tokens[i]
    rest = tokens[:i] + tokens[i + 1:]
    d = random.randint(2, 10) * random.choice([-1, 1])
    j = min(max(i + d, 0), len(rest))
    if abs(j - i) < 2:
        return None
    return rest[:j] + [t] + rest[j:], kind

def swap_adjacent(tokens):
    cand = [i for i in range(len(tokens) - 1) if tokens[i] != tokens[i + 1]]
    i = random.choice(cand)
    out = tokens[:]
    out[i], out[i + 1] = out[i + 1], out[i]
    return out, "swap_adjacent"

MUTS = [
    lambda t: move(t, {";"}, "moved_semicolon"),
    lambda t: move(t, {"{", "}"}, "moved_brace"),
    lambda t: move(t, {"(", ")"}, "moved_paren"),
    swap_adjacent,
]


def gcc_ok(code):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.c"
        p.write_text(code)
        r = subprocess.run(["gcc", "-fsyntax-only", "-w", str(p)], capture_output=True, text=True)
    return r.returncode == 0


def collapse(t):
    if t[0].isdigit():
        return "<num>"
    return "<fname>" if FNAME.match(t) else t


# ---------- build pairs ----------
cands, seen = [], set()
while len(cands) < int(N_PAIRS * 1.6):
    toks = TOKEN.findall(make_valid())
    key = " ".join(toks)
    if key in seen:
        continue
    seen.add(key)
    out = random.choice(MUTS)(toks)
    if out is None or out[0] == toks:
        continue
    cands.append((toks, out[0], out[1]))

with ThreadPoolExecutor(8) as ex:
    oks = list(tqdm(ex.map(lambda c: gcc_ok(" ".join(c[1])), cands), total=len(cands), desc="gcc"))

pairs, harmless = [], Counter()
for (valid, broken, kind), ok in zip(cands, oks):
    if ok:
        harmless[kind] += 1          # mutation happened to stay valid: drop the pair
        continue
    pairs.append((valid, broken, kind))
pairs = pairs[:N_PAIRS]
random.shuffle(pairs)

rows = []
for i, (valid, broken, kind) in enumerate(pairs):
    split = "train" if i < 0.8 * len(pairs) else ("val" if i < 0.9 * len(pairs) else "test")
    two = [{"toks": valid, "label": 0, "kind": "none"}, {"toks": broken, "label": 1, "kind": kind}]
    random.shuffle(two)
    for r in two:
        rows.append({"code": " ".join(r["toks"]), "label": r["label"], "kind": r["kind"],
                     "split": split, "pair": i, "ctoks": [collapse(t) for t in r["toks"]]})

# twins must have identical token counts
by_pair = {}
for r in rows:
    by_pair.setdefault(r["pair"], []).append(Counter(r["ctoks"]))
assert all(a == b for a, b in by_pair.values()), "twin token counts differ!"

# ---------- tokenize ----------
vc = Counter()
for r in rows:
    if r["split"] == "train":
        vc.update(r["ctoks"])
vocab = ["<pad>", "<unk>"] + sorted(vc)
stoi = {w: i for i, w in enumerate(vocab)}
lens = np.array([len(r["ctoks"]) for r in rows])
ids = np.zeros((len(rows), int(lens.max())), dtype=np.int16)
for i, r in enumerate(rows):
    ids[i, :lens[i]] = [stoi.get(w, 1) for w in r["ctoks"]]
sid = {"train": 0, "val": 1, "test": 2}
np.savez(OUT / "v2_tokens.npz", ids=ids, lengths=lens,
         labels=np.array([r["label"] for r in rows], dtype=np.int64),
         split=np.array([sid[r["split"]] for r in rows], dtype=np.int64))
(OUT / "v2_vocab.json").write_text(json.dumps(vocab))
(OUT / "v2_syntax.jsonl").write_text("\n".join(
    json.dumps({k: r[k] for k in ("code", "label", "kind", "split", "pair")}) for r in rows))

print(f"\nSaved {len(rows)} snippets = {len(pairs)} twin pairs  -> {OUT}/v2_*")
print("Label balance:", dict(Counter(r["label"] for r in rows)))
print("Broken kinds:", dict(Counter(r["kind"] for r in rows if r["label"] == 1)))
print("Dropped (mutation stayed valid):", dict(harmless))
print("Splits:", dict(Counter(r["split"] for r in rows)))
print("Tokens per snippet: mean %.0f, max %d | vocab %d | twin token counts verified identical"
      % (lens.mean(), lens.max(), len(vocab)))
v = next(r for r in rows if r["label"] == 0)
b = next(r for r in rows if r["pair"] == v["pair"] and r["label"] == 1)
print("\n--- valid ---\n" + v["code"] + f"\n--- broken twin ({b['kind']}) ---\n" + b["code"])