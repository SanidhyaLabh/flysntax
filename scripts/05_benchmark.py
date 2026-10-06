import json, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

root = Path("data/raw/fly-connectome-49k")
g = root / "graph"
offsets = np.fromfile(g / "edges_offsets.i32", dtype=np.int32).astype(np.int64)
source = np.fromfile(g / "edges_source.u16", dtype=np.uint16).astype(np.int64)
weight = np.fromfile(g / "edges_weight.f32", dtype=np.float32)
N = len(offsets) - 1

W = torch.sparse_csr_tensor(torch.from_numpy(offsets), torch.from_numpy(source),
                            torch.from_numpy(weight), size=(N, N))

print("torch threads:", torch.get_num_threads())
for B in [1, 16, 64]:
    X = torch.randn(N, B)
    for _ in range(2):
        torch.relu(W @ X)                      # warm-up
    t = time.perf_counter()
    for _ in range(10):
        torch.relu(W @ X)
    ms = (time.perf_counter() - t) / 10 * 1000
    print(f"batch {B:3d}: {ms:8.1f} ms per recurrent step")

sup = json.loads((root / "neurons/superclass.json").read_text())
inn = np.fromfile(root / "interface/in_index.i32", dtype=np.int32)
print("\nin_index neurons by superclass:", dict(Counter(sup[i] for i in inn)))