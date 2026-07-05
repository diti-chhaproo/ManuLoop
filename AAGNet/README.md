# AAGNet on MFCAD++ — ManuLoop Setup

This is a working fork of [AAGNet](https://github.com/miles629/AAGNet) (machining feature
recognition via graph neural networks), configured and trained on the MFCAD++ dataset
for the ManuLoop pipeline. This README documents the environment, the changes made to
get it running on this machine, and how to use the trained model for inference.

## Environment

Conda env name: `pyocc`. Built from `environment_trimmed.yaml` (a pared-down version of
the upstream `environment.yaml`) plus a few packages installed via pip after the conda
solver repeatedly hung on the full multi-channel dependency set.

Key version pins that matter and why:
- `numpy==1.24.4` — newer numpy (2.x) breaks the ABI that this PyTorch/DGL build expects.
- `libdeflate==1.8` — newer libdeflate builds rename their shared library from
  `libdeflate.dll` to `deflate.dll`, which silently breaks `pythonocc-core`'s DLL
  loading chain (`TKService.dll` → `FreeImage.dll` → `tiff.dll` → `libdeflate.dll`).
  This one cost real time to track down — see git history / conversation log if the
  DLL error resurfaces after any `conda update`.
- `occt=7.5.1`, `pythonocc-core=7.5.1`, `occwl=2.0.2` — must stay together; these were
  matched against the same builds the original AAGNet authors used.

To recreate:
```
conda create -n pyocc python=3.10 -y
conda install -n pyocc --override-channels -c pytorch -c nvidia -c conda-forge pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -y
conda install -n pyocc --override-channels -c dglteam/label/cu118 -c pytorch -c nvidia -c conda-forge dgl=1.1.0 -y
conda install -n pyocc --override-channels -c conda-forge -c pytorch -c nvidia "numpy=1.24.*" -y
conda install -n pyocc --override-channels -c lambouj -c conda-forge occt=7.5.1 pythonocc-core=7.5.1 occwl=2.0.2 -y
conda install -n pyocc --override-channels -c conda-forge -c pytorch -c nvidia "numpy=1.24.*" "libdeflate=1.8" -y
conda run -n pyocc pip install scipy==1.10.1 numba==0.58.1 h5py==3.9.0 scikit-learn==1.2.2 pandas==2.0.3
conda run -n pyocc pip install timm==0.9.2 wandb==0.15.5 torchmetrics==0.11.4 torch-ema==0.3 opencv-python==4.8.0.76 ijson
conda run -n pyocc pip install "setuptools<81"   # wandb needs pkg_resources, removed in newer setuptools
```

**Note on `conda install`**: keep each install scoped to only the channels it needs
(`--override-channels -c <2-3 channels>`). Combining all of `pytorch`, `nvidia`,
`lambouj`, `dglteam`, and `conda-forge` in one solve reliably hung indefinitely on this
machine; splitting into small scoped solves resolved in under a minute each.

## Data extraction

STEP files live outside this repo at `C:\manuloop\MFCAD_dataset\MFCAD++_dataset\step\{train,val,test}`.
Extraction produces, per split, a folder (`gaag/aag_train`, `gaag/aag_val`, `gaag/aag_test`)
containing `graphs.json`, `attr_stat.json`, and (if any files failed) `failed_extractions.txt`.

```
python -m dataset.AAGExtractor --step_path <step/train> --output <gaag/aag_train> --num_workers 8
python dataset/run_label_extraction.py --step_path <step/train> --output <gaag/labels> --num_workers 10
```

`AAGExtractor.py` was patched to catch per-file exceptions instead of crashing the whole
multiprocessing batch on the first malformed STEP file — failures are logged to
`failed_extractions.txt` (filename + error) in the output directory instead. Across the
full MFCAD++ dataset (59,665 files), 210 failed (0.35%), all legitimate STEP topology
defects (non-manifold shapes, open shells, duplicate coedges) — inspect the
`failed_extractions.txt` in each `gaag/aag_*` folder for specifics.

**Important architectural note:** `dataloader/base.py` and `dataloader/mfcad2.py` were
rewritten to stream-parse `graphs.json` via `ijson` instead of `json.load()`-ing the
whole file, and to load each split from its own `aag_<split>/` folder rather than a
single shared combined file. The extracted MFCAD++ corpus is ~22GB of JSON text; a plain
`json.load()` would balloon several-fold into Python objects and exceed this machine's
16GB RAM. Each split's `MFCAD2Dataset` always normalizes using `aag_train/attr_stat.json`
regardless of which split it's loading (train-only normalization stats, avoiding
val/test leakage). If you ever go back to a single merged `graphs.json`, you will hit
`MemoryError` again — don't merge, keep the per-split streaming setup.

## Training

`engine/seg_trainer.py`'s dataset path (`wandb.config["dataset"]`) points at
`C:\manuloop\MFCAD_dataset\MFCAD++_dataset\gaag` (the folder containing `aag_train/`,
`aag_val/`, `aag_test/`, `labels/`, `train.txt`, `val.txt`, `test.txt`).

```
python -m engine.seg_trainer
```

Trains for 100 epochs (~80-90 sec/epoch on an RTX 4050, ~2h15m total including ~30min
initial data loading). Logs and checkpoints are written to `output/<timestamp>/`.
`output/` is gitignored (large binaries) — see below for the current best checkpoint.

### Latest run results (2026-07-05)

- Run directory: `output/2026_07_05_12_06_51/`
- **Best checkpoint: `output/2026_07_05_12_06_51/weight_89-epoch.pth`** (highest val IoU)
- Final test set: `test_seg_acc: 0.9924` (99.24%), `test_seg_iou: 0.9861`, `test_loss: 0.0166`
- Best val: `val_seg_acc: 0.9929` (99.29%), `val_seg_iou: 0.9865` (epoch 89)

## Inference

`infer.py` runs the full pipeline (STEP file → gAAG extraction → trained model → per-face
predictions) on a single STEP file.

```
python infer.py path/to/part.step
```

```python
from infer import load_model, load_norm_stat, load_feature_schema, infer_step_file

model = load_model()              # defaults to the best checkpoint above
stat = load_norm_stat()            # aag_train/attr_stat.json
schema = load_feature_schema()      # feature_lists/all.json

result = infer_step_file("part.step", model, stat, schema, device="cuda")
# {0: {"predicted_class": 24, "confidence": 0.9999}, 1: {...}, ...}
```

**Output format**: a dict keyed by `face_id` (an integer). Each value is
`{"predicted_class": int, "confidence": float}`.

- `predicted_class` is 0-24, see `feature_lists/all.json`'s sibling class list or
  `feature_labels.txt` in the dataset folder for the name mapping (0=Chamfer,
  1=Through hole, ..., 24=Stock/non-feature).
- `face_id` is the node index assigned by occwl's `face_adjacency` graph construction
  for that specific shape — it is deterministic per STEP file but is **not** the same
  as the STEP file's internal `ADVANCED_FACE` entity numbering.
- Bad input (missing file, wrong extension) raises a clean `FileNotFoundError` /
  `ValueError` rather than crashing — do not skip that validation if you modify
  `infer_step_file`, since a missing/invalid STEP file previously caused a hard
  segmentation fault deep inside `pythonocc-core`'s native STEP reader.
