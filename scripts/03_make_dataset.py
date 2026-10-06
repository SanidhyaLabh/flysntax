import json, random, re, subprocess, tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

random.seed(0)
N_TOTAL = 6000
OUT = Path("data/processed/c_syntax.jsonl")

VARS = ["a", "b", "c", "x", "y", "z", "n", "sum", "total", "count", "val"]
FUNCS = ["add", "sub", "mul", "calc", "step", "update", "score", "mix"]

# ---------- generate valid C ----------
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

# ---------- break it ----------
def delete_char(ch, kind):
    def op(s):
        idx = [i for i, c in enumerate(s) if c == ch]
        if not idx:
            return None
        i = random.choice(idx)
        return s[:i] + s[i + 1:], kind
    return op

def keyword_typo(s):
    ms = list(re.finditer(r"\b(return|int|if|while|for|else)\b", s))
    if not ms:
        return None
    m = random.choice(ms)
    w = m.group(0)
    j = random.randrange(len(w))
    return s[:m.start()] + w[:j] + w[j + 1:] + s[m.end():], "keyword_typo"

def semicolon_to_comma(s):
    idx = [i for i, c in enumerate(s) if c == ";"]
    if not idx:
        return None
    i = random.choice(idx)
    return s[:i] + "," + s[i + 1:], "semicolon_to_comma"

def wrong_bracket(s):
    idx = [i for i, c in enumerate(s) if c == ")"]
    if not idx:
        return None
    i = random.choice(idx)
    return s[:i] + "]" + s[i + 1:], "wrong_bracket"

def dup_op(s):
    idx = [m.start() for m in re.finditer(r" [+\-*] ", s)]
    if not idx:
        return None
    i = random.choice(idx)
    return s[:i + 2] + s[i + 1] + " " + s[i + 2:], "dup_op"

def undeclared_var(s):
    start = s.find("{")
    ms = [m for m in re.finditer(r"\b(" + "|".join(VARS) + r")\b", s[start:])]
    if not ms:
        return None
    m = random.choice(ms)
    a, b = start + m.start(), start + m.end()
    return s[:a] + "ghost" + s[b:], "undeclared_var"

OPS = [
    delete_char(";", "missing_semicolon"),
    delete_char("}", "missing_close_brace"),
    delete_char("{", "missing_open_brace"),
    delete_char(")", "missing_close_paren"),
    delete_char("(", "missing_open_paren"),
    keyword_typo, semicolon_to_comma, wrong_bracket, dup_op, undeclared_var,
]

def mutate(s):
    ops = OPS[:]
    random.shuffle(ops)
    for op in ops:
        out = op(s)
        if out:
            return out
    return None

# ---------- ground truth from gcc ----------
def gcc_ok(code):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.c"
        p.write_text(code)
        r = subprocess.run(["gcc", "-fsyntax-only", "-w", str(p)],
                           capture_output=True, text=True)
    return r.returncode == 0

# ---------- build ----------
items, seen = [], set()
while len(items) < N_TOTAL:
    code = make_valid()
    kind = "none"
    if len(items) % 2 == 1:
        out = mutate(code)
        if out:
            code, kind = out
    if code in seen:
        continue
    seen.add(code)
    items.append({"code": code, "kind": kind})

with ThreadPoolExecutor(8) as ex:
    oks = list(tqdm(ex.map(lambda it: gcc_ok(it["code"]), items), total=len(items), desc="gcc"))

rows, bad_valid = [], 0
for it, ok in zip(items, oks):
    if it["kind"] == "none" and not ok:
        bad_valid += 1          # generator bug: drop these
        continue
    rows.append({"code": it["code"], "label": 0 if ok else 1,
                 "kind": it["kind"] if not ok else ("none" if it["kind"] == "none" else "harmless_" + it["kind"])})

random.shuffle(rows)
for i, r in enumerate(rows):
    r["split"] = "train" if i < 0.8 * len(rows) else ("val" if i < 0.9 * len(rows) else "test")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(json.dumps(r) for r in rows))

print(f"\nSaved {len(rows)} snippets to {OUT} (dropped {bad_valid} bad 'valid' ones)")
print("Labels (0=valid, 1=error):", dict(Counter(r["label"] for r in rows)))
print("Kinds:", dict(Counter(r["kind"] for r in rows)))
print("Splits:", dict(Counter(r["split"] for r in rows)))
print("Chars per snippet: mean %.0f, max %d" % (
    sum(len(r["code"]) for r in rows) / len(rows), max(len(r["code"]) for r in rows)))
print("\n--- example valid ---\n" + next(r["code"] for r in rows if r["label"] == 0))
e = next(r for r in rows if r["label"] == 1)
print(f"--- example broken ({e['kind']}) ---\n" + e["code"])