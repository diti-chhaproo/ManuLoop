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


def infer_step_file(step_path, model, stat, feature_schema, device="cuda"):
    """
    Run AAGNet feature classification on a single STEP file.

    Returns:
        dict[int, dict]: face_id -> {"predicted_class": int (0-24), "confidence": float}
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

    graph = sample["graph"].to(device)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step_path", type=str, help="Path to the STEP file to classify")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--stat", type=str, default=str(DEFAULT_STAT_PATH))
    parser.add_argument("--feature_schema", type=str, default=str(DEFAULT_FEATURE_SCHEMA_PATH))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.weights, args.device)
    stat = load_norm_stat(args.stat)
    feature_schema = load_feature_schema(args.feature_schema)

    predictions = infer_step_file(args.step_path, model, stat, feature_schema, args.device)
    print(json.dumps(predictions, indent=2))
