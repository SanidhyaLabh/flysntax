---
license: cc-by-4.0
language:
  - en
tags:
  - connectomics
  - neuroscience
  - drosophila
  - connectome
  - graph
pretty_name: Fly Connectome 49k — the exact MaleCNS central-brain graph used by Fly LLM
size_categories:
  - 1M<n<10M
---

# Fly Connectome 49k

The **49,393-neuron / 9,050,172-edge** central-brain graph that
[`ngxson/fly-llm-hf`](https://huggingface.co/ngxson/fly-llm-hf) uses as its recurrent layer,
packaged as plain typed arrays and joined to the **MaleCNS body IDs, cell types,
superclasses and soma positions** it was derived from.

Until now this graph was reachable only by loading a 284 MB model checkpoint and reading its
frozen buffers. This dataset is that graph, verifiable on its own, so you can build on the
wiring without adopting anyone's model.

**Looking for the whole fly?** This is a central-brain *subset*. The complete MaleCNS v1.0
connectome — 166,700 neurons, 25,582,938 edges, 124,177,617 synaptic contacts, raw signed
weights — is
[fly-connectome-malecns-166k](https://huggingface.co/datasets/fernandofernandes/fly-connectome-malecns-166k),
in the same conventions so the two compose.

> **This is measured biological connectivity, not a trained artifact.** No weight here was
> learned. It is a scaled copy of the MaleCNS v1.0 connectome.

## Contents — 76.13 MB

| File | Shape | dtype | What it is |
|---|---|---|---|
| `graph/edges_offsets.i32` | 49,394 | int32 | CSR row offsets; **row = destination neuron** |
| `graph/edges_source.u16` | 9,050,172 | uint16 | presynaptic neuron index |
| `graph/edges_weight.f32` | 9,050,172 | float32 | **canonical** signed weight |
| `graph/edges_weight.i16` | 9,050,172 | int16 | fixed-point weight for browser payloads |
| `interface/in_index.i32` | 14,064 | int32 | neurons the model injects input into |
| `interface/out_index.i32` | 49,393 | int32 | neurons the readout observes |
| `neurons/body_id.i64` | 49,393 | int64 | **MaleCNS body IDs** |
| `neurons/type_index.i32` + `type_labels.json` | 49,393 / 9,161 | int32 | cell-type groups |
| `neurons/cell_type.json` | 49,393 | string | per-neuron cell type |
| `neurons/superclass.json`, `side.json` | 49,393 | string | superclass and body side |
| `neurons/soma_position.i32` | 49,393 × 3 | int32 | soma location in 8 nm voxels |
| `neurons/soma_valid.u8` | 49,393 | uint8 | 1 where a soma location exists |
| `manifest.json` | — | — | shapes, dtypes, SHA-256 of every file, provenance |
| `groups/groups.npz` | — | — | the cell-type grouping the trainer loads, so training needs no 1 GB source rebuild |

6,061,699 edges are positive, 2,988,473 negative, none zero. 44,279 of 49,393 neurons have a
measured soma location; the remaining 5,114 are flagged in `soma_valid` and **their
coordinates are zeros, not estimates**. 8,156 neurons carry an official cell type; the 1,005
without one each get their own group rather than being merged.

## Load it

```python
import json, numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(snapshot_download("fernandofernandes/fly-connectome-49k", repo_type="dataset"))
manifest = json.loads((root / "manifest.json").read_text())

offsets = np.fromfile(root / "graph/edges_offsets.i32", dtype=np.int32)   # 49,394
source  = np.fromfile(root / "graph/edges_source.u16",  dtype=np.uint16)  # 9,050,172
weight  = np.fromfile(root / "graph/edges_weight.f32",  dtype=np.float32) # 9,050,172

# Incoming synapses of neuron 12345, as (presynaptic index, weight) pairs:
lo, hi = offsets[12345], offsets[12346]
incoming = list(zip(source[lo:hi], weight[lo:hi]))
```

As a SciPy sparse matrix, `W[destination, source]`:

```python
from scipy.sparse import csr_matrix
W = csr_matrix((weight, source.astype(np.int32), offsets), shape=(49393, 49393))
```

The int16 encoding is `value = code * manifest["weights"]["quantised"]["scale"]`. It
preserves every sign, with maximum relative error **1.53e-05**. Use `.f32` for anything
numerical; `.i16` exists to halve a browser download.

## Verify it

Every file's SHA-256 is in `manifest.json`, and the manifest also carries the
**frozen-buffer digests recorded inside the trained model checkpoints** — so you can prove
this package is the same graph those models ran on, without downloading them.
[`verify_connectome_package.py`](https://github.com/fernando-neto-ai/fly-wordbrain/blob/main/scripts/verify_connectome_package.py)
runs 18 such checks with nothing but NumPy:

```
All 18 checks passed: 49,393 neurons, 9,050,172 edges, 44,279 positioned somata, 9,161 cell-type groups.
```

## Training against it

`groups/groups.npz` carries the verified neuron→cell-type grouping bound to this exact graph
(`node_type_index`, body IDs, labels, and the frozen-buffer digests). It exists so that
training a model on this connectome does not require re-deriving the grouping from the
~1 GB of upstream MaleCNS feather files. The
[fly-wordbrain pipeline](https://github.com/fernando-neto-ai/fly-wordbrain) fetches it
straight from here:

```bash
python scripts/run_pipeline.py --quick     # one command, Apple Silicon
```

## Provenance, and one honest gap

Derived from **MaleCNS v1.0 central brain** — superclasses `cb_sensory`,
`visual_projection`, `cb_intrinsic`, `ascending_neuron`, `descending_neuron` — via the
pinned reference checkpoint at revision `65c677b3d566a2e9793d5f72999cdb441c6c0a9f`.

Every edge was matched back to its source endpoints, sign and globally scaled magnitude,
with **maximum absolute discrepancy 2.98e-08**. The global scale from source synapse counts
to checkpoint weights is `9.529943345114589e-04`.

**The gap:** the induced source subgraph over these same 49,393 neurons contains
**9,679,074** edges, but the checkpoint stores **9,050,172**. The **628,902** missing edges
were not restored, and *we do not know why the original packaging omitted them*. This
dataset reproduces the checkpoint's topology exactly, because that is what the models were
trained on — it is **not** a complete induced subgraph of MaleCNS. If you need completeness,
go to the source data, not to this file.

Weights are scaled synapse counts with a transmitter-derived sign. They are not conductances
or measured physiological strengths.

## Licence and attribution

**CC BY 4.0.** Connectome data: FlyEM / HHMI Janelia Research Campus, University of
Cambridge, MRC Laboratory of Molecular Biology, and Google Research. Graph packaging and
interface layout follow [`ngxson/fly-llm-hf`](https://huggingface.co/ngxson/fly-llm-hf)
(CC BY 4.0). Packaging code is MIT.

Used by [fly-wordbrain](https://github.com/fernando-neto-ai/fly-wordbrain).
