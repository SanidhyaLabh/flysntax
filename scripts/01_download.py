from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from scipy.sparse import csr_matrix

root = Path(snapshot_download(
    "fernandofernandes/fly-connectome-49k",
    repo_type="dataset",
    local_dir="data/raw/fly-connectome-49k",
))

print("\nFiles in the dataset:")
for p in sorted(root.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(root)}  ({p.stat().st_size/1e6:.1f} MB)")

g = root / "graph"
offsets = np.fromfile(g / "edges_offsets.i32", dtype=np.int32)
source  = np.fromfile(g / "edges_source.u16", dtype=np.uint16).astype(np.int32)
weight  = np.fromfile(g / "edges_weight.f32", dtype=np.float32)

N = len(offsets) - 1
W = csr_matrix((weight, source, offsets), shape=(N, N))

print(f"\nNeurons: {N}")
print(f"Synapses (edges): {W.nnz}")
print(f"Excitatory (+): {(weight > 0).sum()}  Inhibitory (-): {(weight < 0).sum()}")
print(f"Weight range: {weight.min():.3f} to {weight.max():.3f}")