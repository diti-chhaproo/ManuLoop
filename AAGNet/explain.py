# -*- coding: utf-8 -*-
"""
Explainability layer for AAGNet machining feature classification.

For each face, produces a gradient-based saliency attribution: how much that
face's own input attributes and its 1-hop neighbors' input attributes drove
its predicted class. This approach (rather than reading off an internal
attention/edge-weight tensor) was chosen because the trained checkpoint's
architecture, AAGNetGraphEncoder (models/encoders.py), passes messages
through NodeMPNN/EdgeMPNN (models/layers.py), whose actual neighbor
aggregation happens in PNAConvTower (models/pnaconv.py) via plain sum/max
pooling over per-edge MLP messages - there is no dgl.nn.functional.edge_softmax,
learned edge gate, or attention coefficient anywhere in that path to hook.
Gradient saliency is architecture-agnostic and works regardless.

"Face ID" has the same meaning as in infer.py: the node index assigned by
occwl's face_adjacency graph construction for this shape, not the STEP
file's internal ADVANCED_FACE entity numbering.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from infer import (
    load_model,
    load_norm_stat,
    load_feature_schema,
    build_graph,
    DEFAULT_WEIGHTS,
    DEFAULT_STAT_PATH,
    DEFAULT_FEATURE_SCHEMA_PATH,
)


def _one_hop_neighbors(graph):
    """dict[int, set[int]]: node id -> set of adjacent node ids (undirected)."""
    src, dst = graph.edges()
    src = src.tolist()
    dst = dst.tolist()
    neighbors = {i: set() for i in range(graph.num_nodes())}
    for s, d in zip(src, dst):
        neighbors[s].add(d)
        neighbors[d].add(s)
    return neighbors


def compute_attributions(model, graph, predictions):
    """
    Gradient-based saliency: for each face, backprop its predicted-class
    logit to the input node attributes (graph.ndata["x"]) and read off the
    gradient magnitude at that face's own row and at its 1-hop neighbors'
    rows. This measures how sensitive the prediction is to each face's
    (and its neighbors') input attributes - a standard saliency-map
    attribution, chosen because the model has no internal attention
    weights to read directly (see module docstring).

    Returns:
        dict[int, dict]: face_id -> {
            "attribution_score": float in [0, 1], min-max normalized across
                all faces in this part (self + neighbor gradient magnitude),
            "top_contributing_faces": list of up to 3
                {"face_id": int, "contribution_weight": float}, sorted
                descending by gradient magnitude among 1-hop neighbors,
        }
    """
    num_nodes = graph.num_nodes()
    neighbors = _one_hop_neighbors(graph)

    # Make node attributes a leaf tensor we can backprop to.
    graph.ndata["x"] = graph.ndata["x"].clone().detach().requires_grad_(True)
    logits = model(graph)  # (num_nodes, num_classes)

    self_grad_norm = torch.zeros(num_nodes)
    neighbor_grad_norms = [dict() for _ in range(num_nodes)]

    for i in range(num_nodes):
        pred_class = predictions[i].item()
        scalar_logit = logits[i, pred_class]
        grad = torch.autograd.grad(scalar_logit, graph.ndata["x"], retain_graph=True)[0]
        grad_norms = grad.norm(dim=1)  # (num_nodes,)

        self_grad_norm[i] = grad_norms[i].item()
        for n in neighbors[i]:
            neighbor_grad_norms[i][n] = grad_norms[n].item()

    raw_scores = torch.tensor([
        self_grad_norm[i].item() + sum(neighbor_grad_norms[i].values())
        for i in range(num_nodes)
    ])
    min_s, max_s = raw_scores.min(), raw_scores.max()
    norm_scores = (raw_scores - min_s) / (max_s - min_s).clamp(min=1e-8)

    results = {}
    for i in range(num_nodes):
        top = sorted(neighbor_grad_norms[i].items(), key=lambda kv: kv[1], reverse=True)[:3]
        results[i] = {
            "attribution_score": float(norm_scores[i].item()),
            "top_contributing_faces": [
                {"face_id": int(face_id), "contribution_weight": float(weight)}
                for face_id, weight in top
            ],
        }
    return results


def explain_step_file(step_path, model, stat, feature_schema, device="cuda"):
    """
    Run AAGNet feature classification on a single STEP file with per-face
    gradient saliency attribution.

    Returns:
        dict[int, dict]: face_id -> {
            "predicted_class": int (0-24),
            "confidence": float,
            "attribution_score": float in [0, 1],
            "top_contributing_faces": list of up to 3
                {"face_id": int, "contribution_weight": float},
        }
    """
    graph = build_graph(step_path, stat, feature_schema, device)

    logits = model(graph)
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)

    attributions = compute_attributions(model, graph, predictions)

    results = {}
    for face_id in range(graph.num_nodes()):
        results[face_id] = {
            "predicted_class": int(predictions[face_id].item()),
            "confidence": float(confidences[face_id].item()),
            **attributions[face_id],
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step_path", type=str, help="Path to the STEP file to classify and explain")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--stat", type=str, default=str(DEFAULT_STAT_PATH))
    parser.add_argument("--feature_schema", type=str, default=str(DEFAULT_FEATURE_SCHEMA_PATH))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.weights, args.device)
    stat = load_norm_stat(args.stat)
    feature_schema = load_feature_schema(args.feature_schema)

    predictions = explain_step_file(args.step_path, model, stat, feature_schema, args.device)
    print(json.dumps(predictions, indent=2))
