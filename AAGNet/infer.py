# -*- coding: utf-8 -*-
"""
Inference entry point for AAGNet machining feature classification.

Given a single STEP file, runs the same B-Rep -> gAAG extraction used during
training (dataset.AAGExtractor) followed by the trained AAGNetSegmentor, and
returns a per-face predicted class + confidence.

"Face ID" in the returned dict is the node index assigned by occwl's
face_adjacency graph construction for this shape (dataset.AAGExtractor,
which is the same indexing used to build every training sample's node
labels). It is deterministic for a given STEP file but is not necessarily
the same order as the ADVANCED_FACE entity numbering inside the STEP text.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopoDS import TopoDS_Solid, topods_Solid
from OCC.Extend.TopologyUtils import TopologyExplorer
from occwl.solid import Solid as OccwlSolid

from dataset.AAGExtractor import AAGExtractor
from models.segmentors import AAGNetSegmentor
from utils.data_utils import load_one_graph, load_statistics, standardization, center_and_scale


REPO_ROOT = Path(__file__).parent

# Must match engine/seg_trainer.py's wandb.config used to train the checkpoint.
MODEL_CONFIG = dict(
    arch="AAGNetGraphEncoder",
    edge_attr_dim=12,
    node_attr_dim=10,
    edge_attr_emb=64,
    node_attr_emb=64,
    edge_grid_dim=0,
    node_grid_dim=7,
    edge_grid_emb=0,
    node_grid_emb=64,
    num_layers=3,
    delta=2,
    mlp_ratio=2,
    drop=0.25,
    drop_path=0.25,
    head_hidden_dim=64,
    conv_on_edge=False,
    use_uv_gird=True,
    use_edge_attr=True,
    use_face_attr=True,
)
NUM_CLASSES = 25

DEFAULT_WEIGHTS = REPO_ROOT / "output" / "2026_07_05_12_06_51" / "weight_89-epoch.pth"
DEFAULT_STAT_PATH = Path(r"C:\manuloop\MFCAD_dataset\MFCAD++_dataset\gaag\aag_train\attr_stat.json")
DEFAULT_FEATURE_SCHEMA_PATH = REPO_ROOT / "feature_lists" / "all.json"


def load_model(weights_path=DEFAULT_WEIGHTS, device="cuda"):
    """Instantiate AAGNetSegmentor and load trained weights."""
    model = AAGNetSegmentor(num_classes=NUM_CLASSES, **MODEL_CONFIG)
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_feature_schema(path=DEFAULT_FEATURE_SCHEMA_PATH):
    with open(path, "r") as f:
        return json.load(f)


def load_norm_stat(path=DEFAULT_STAT_PATH):
    return load_statistics(path)


def build_graph(step_path, stat, feature_schema, device="cuda"):
    """
    Run the B-Rep -> gAAG extraction (dataset.AAGExtractor) and the same
    standardization/center-and-scale preprocessing used at training time,
    returning a DGL graph ready to feed into AAGNetSegmentor.

    Raises FileNotFoundError/ValueError on bad input rather than letting
    pythonocc-core's native STEP reader fail silently on an invalid path -
    a missing/malformed STEP file previously caused a hard segmentation
    fault deep inside its native code instead of a clean Python exception.
    """
    step_path = Path(step_path)
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    if step_path.suffix.lower() not in (".step", ".stp"):
        raise ValueError(f"Expected a .step/.stp file, got: {step_path}")

    extractor = AAGExtractor(step_path, feature_schema)
    graph_data = extractor.process()

    sample = load_one_graph(step_path.stem, graph_data)
    sample = standardization(sample, stat)
    sample = center_and_scale(sample)

    return sample["graph"].to(device)


def infer_step_file(step_path, model, stat, feature_schema, device="cuda"):
    """
    Run AAGNet feature classification on a single STEP file.

    Returns:
        dict[int, dict]: face_id -> {"predicted_class": int (0-24), "confidence": float}
    """
    graph = build_graph(step_path, stat, feature_schema, device)

    with torch.no_grad():
        logits = model(graph)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)

    results = {}
    for face_id in range(graph.num_nodes()):
        results[face_id] = {
            "predicted_class": int(predictions[face_id].item()),
            "confidence": float(confidences[face_id].item()),
        }
    return results


def build_graph_from_shape(shape, label, stat, feature_schema, device="cuda"):
    """
    Same as build_graph, but takes an already-loaded TopoDS_Solid directly
    instead of a STEP file path - used by infer_multibody_step_file to feed
    solids extracted from an assembly straight into AAGExtractor.

    Feeding the shape directly (rather than writing it out to its own
    standalone STEP file and reading that back in) matters: that round-trip
    was found to segfault - with zero Python traceback - inside OCCT's
    native STEP transfer code (STEPControl_Reader.TransferRoots()) for some
    solids extracted out of multi-body assemblies, even though both the
    original assembly file and the in-memory shape are fine on their own.
    Working with the shape in memory avoids that path entirely.
    """
    extractor = AAGExtractor(label, feature_schema, preloaded_shape=shape)
    graph_data = extractor.process()

    sample = load_one_graph(label, graph_data)
    sample = standardization(sample, stat)
    sample = center_and_scale(sample)

    return sample["graph"].to(device)


def infer_shape(shape, label, model, stat, feature_schema, device="cuda"):
    """
    Same as infer_step_file, but for an already-loaded TopoDS_Solid rather
    than a STEP file path. See build_graph_from_shape for why this exists.

    Returns:
        dict[int, dict]: face_id -> {"predicted_class": int (0-24), "confidence": float}
    """
    graph = build_graph_from_shape(shape, label, stat, feature_schema, device)

    with torch.no_grad():
        logits = model(graph)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)

    results = {}
    for face_id in range(graph.num_nodes()):
        results[face_id] = {
            "predicted_class": int(predictions[face_id].item()),
            "confidence": float(confidences[face_id].item()),
        }
    return results


def split_compound_to_solids(step_path):
    """
    Load a STEP file's top-level shape and split it into individual solid
    bodies. Real-world STEP files (unlike MFCAD++'s single-solid-per-file
    synthetic parts) are frequently multi-body assemblies, which load as a
    TopoDS_Compound rather than a single TopoDS_Solid - the shape type
    AAGExtractor.process() asserts on. Confirmed via testing on real GrabCAD
    files: 7 of 11 real assembly STEP files failed on exactly that assertion.

    Each returned solid has its placement transform normalized to identity
    (occwl.shape.Shape.set_transform_to_identity()). Solids pulled out of an
    assembly retain their placement relative to the assembly's root
    coordinate system - entirely normal STEP behavior - but occwl's Face
    API (used throughout AAGExtractor) explicitly refuses to compute
    surface geometry for a transformed face and raises a clean, catchable
    AssertionError rather than silently giving wrong results. This is safe
    to do unconditionally: AAGExtractor recenters/rescales every solid to
    a unit box regardless of its starting position, so which coordinate
    frame a solid starts in doesn't matter to the resulting classification.

    Returns:
        list[TopoDS_Solid]: one entry per solid body found. If the file's
        top-level shape is already a single solid, returns a single-item
        list containing it, so callers don't need two separate code paths
        for single-body vs. multi-body files.
    """
    step_path = Path(step_path)
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    if step_path.suffix.lower() not in (".step", ".stp"):
        raise ValueError(f"Expected a .step/.stp file, got: {step_path}")

    reader = STEPControl_Reader()
    reader.ReadFile(str(step_path))
    reader.TransferRoots()
    shape = reader.OneShape()

    if isinstance(shape, TopoDS_Solid):
        solids = [shape]
    else:
        solids = []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solids.append(topods_Solid(explorer.Current()))
            explorer.Next()

    normalized = []
    for solid in solids:
        wrapped = OccwlSolid(solid, allow_compound=True)
        wrapped.set_transform_to_identity()
        normalized.append(wrapped.topods_shape())
    return normalized


def infer_multibody_step_file(step_path, model, stat, feature_schema, device="cuda", min_face_ratio=0.05):
    """
    Run AAGNet feature classification on a STEP file that may contain
    multiple solid bodies (a real-world assembly), rather than the single
    solid body MFCAD++ (and AAGExtractor) assume.

    Each solid is passed directly to AAGExtractor in memory (infer_shape /
    build_graph_from_shape) - this function does not change how a single
    body is extracted or classified, it only adds a preprocessing step that
    splits a multi-body file into several single-body solids first.

    Bodies far smaller than the largest one in the file (by face count) are
    treated as likely hardware (fasteners, washers, bearings) rather than
    the part(s) actually being manufactured, and are excluded from
    classification - but always reported in "filtered_out", never silently
    dropped.

    Args:
        min_face_ratio: a solid is filtered out if its face count is below
            this fraction of the largest solid's face count in the same
            file. Default 0.05 (5%).

    Returns:
        dict: {
            "body_0": {face_id: {"predicted_class": int, "confidence": float}, ...},
            "body_1": {...},
            ...,
            "filtered_out": [
                {"body_index": int, "num_faces": int, "reason": str}, ...
            ],
            "failed_bodies": [
                {"body_index": int, "num_faces": int, "error": str}, ...
            ],
        }
        Body indices in the keys/filtered_out/failed_bodies entries refer to
        the order solids were found in the file, which is not necessarily
        meaningful assembly structure (e.g. part names) - STEP compounds
        don't always preserve that in a way OCCT exposes simply.

        A body lands in "failed_bodies" (rather than crashing the whole
        call) when it hits a topology defect AAGExtractor's checks reject.
        One bad body must not discard already-successful predictions for
        the file's other bodies.
    """
    step_path = Path(step_path)
    solids = split_compound_to_solids(step_path)

    face_counts = [TopologyExplorer(s).number_of_faces() for s in solids]
    max_faces = max(face_counts) if face_counts else 0

    results = {}
    filtered_out = []
    failed_bodies = []

    for i, (solid, n_faces) in enumerate(zip(solids, face_counts)):
        if max_faces > 0 and n_faces < min_face_ratio * max_faces:
            reason = (
                f"{n_faces} faces is below {min_face_ratio:.0%} of the "
                f"largest body's {max_faces} faces - likely hardware "
                f"(fastener/washer/bearing), not classified"
            )
            filtered_out.append({"body_index": i, "num_faces": n_faces, "reason": reason})
            print(f"[infer_multibody_step_file] filtered body_{i}: {reason}")
            continue

        try:
            label = f"{step_path.stem}_body_{i}"
            results[f"body_{i}"] = infer_shape(solid, label, model, stat, feature_schema, device)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            failed_bodies.append({"body_index": i, "num_faces": n_faces, "error": error})
            print(f"[infer_multibody_step_file] body_{i} failed: {error}")

    results["filtered_out"] = filtered_out
    results["failed_bodies"] = failed_bodies
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step_path", type=str, help="Path to the STEP file to classify")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--stat", type=str, default=str(DEFAULT_STAT_PATH))
    parser.add_argument("--feature_schema", type=str, default=str(DEFAULT_FEATURE_SCHEMA_PATH))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--multibody", action="store_true",
                         help="Treat the STEP file as a possible multi-body assembly "
                              "(splits into individual solids first; see infer_multibody_step_file)")
    args = parser.parse_args()

    model = load_model(args.weights, args.device)
    stat = load_norm_stat(args.stat)
    feature_schema = load_feature_schema(args.feature_schema)

    if args.multibody:
        predictions = infer_multibody_step_file(args.step_path, model, stat, feature_schema, args.device)
    else:
        predictions = infer_step_file(args.step_path, model, stat, feature_schema, args.device)
    print(json.dumps(predictions, indent=2))
