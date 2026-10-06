import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("data/raw/fly-connectome-49k")

pos = np.fromfile(root / "neurons/soma_position.i32", dtype=np.int32).reshape(-1, 3)
valid = np.fromfile(root / "neurons/soma_valid.u8", dtype=np.uint8).astype(bool)
print("positions:", pos.shape, "| with valid position:", valid.sum())
print("min xyz:", pos[valid].min(0), "| max xyz:", pos[valid].max(0))

sup = json.loads((root / "neurons/superclass.json").read_text())
print("\nsuperclass type:", type(sup).__name__, "| length:", len(sup))
if isinstance(sup, list):
    print(Counter(sup).most_common(15))
else:
    print(list(sup.items())[:10])

for name in ["in_index", "out_index"]:
    a = np.fromfile(root / f"interface/{name}.i32", dtype=np.int32)
    print(f"\n{name}: shape {a.shape}, first 10 {a[:10]}, min {a.min()}, max {a.max()}")

man = json.loads((root / "manifest.json").read_text())
print("\nmanifest keys:", list(man.keys()))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(pos[valid, 0], pos[valid, 2], s=0.3, alpha=0.4)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_title("Fly central brain: soma positions")
Path("runs").mkdir(exist_ok=True)
fig.savefig("runs/brain_preview.png", dpi=150)
print("\nSaved runs/brain_preview.png")