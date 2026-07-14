# -*- coding: utf-8 -*-
"""
Multi-face DFM rules for cored holes:
    Rule 11 - depth-to-diameter (L/D) ratio
    Rule 12 - blind-hole bottom thickness
    Rule 13 - hole-to-edge / hole-to-hole distance

Builds on geometry_rules.py's single-face hole-diameter measurement,
feature_grouping.py (pairs a blind hole's cylindrical wall with its flat
bottom face - same predicted class, connected), wall_thickness_rule.py
(bottom-thickness reuses the wall-thickness primitive directly), and
boundary_sampling.py (proximity checks).
"""
from pathlib import Path

import numpy as np
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.STEPControl import STEPControl_Reader
from occwl.solid import Solid
from occwl.graph import face_adjacency

from wall_thickness_rule import MIN_WALL_THICKNESS_MM, find_nearest_opposing_face
from boundary_sampling import sample_group_boundary_points, min_pairwise_distance
from feature_grouping import group_faces_by_class_and_adjacency

HOLE_CLASSES = {1, 12}  # Through hole, Blind hole

BLIND_LD_MAX_RATIO = 2.5
THROUGH_LD_MAX_RATIO_SMALL = 4.0    # diameter < THROUGH_LD_DIAMETER_BREAKPOINT_MM
THROUGH_LD_MAX_RATIO_LARGE = 10.0   # diameter >= THROUGH_LD_DIAMETER_BREAKPOINT_MM
THROUGH_LD_DIAMETER_BREAKPOINT_MM = 12.5

BOTTOM_THICKNESS_MIN_RATIO = 0.20   # Rule 12: 20% of the hole's own diameter, not a fixed mm value

HOLE_SPACING_MIN_RATIO = 1.5        # Rule 13: x wall thickness


def _circle_edge_centers(face):
    """[(edge, center_point_3d), ...] for each circular boundary edge of a face."""
    centers = []
    for edge in face.edges():
        if edge.curve_type() == "circle":
            circ = edge.specific_curve()
            center = np.array(circ.Location().Coord())
            centers.append((edge, center))
    return centers


def measure_hole_depth(cylindrical_face, bottom_face=None):
    """
    Rule 11 support: measure a hole's depth along its axis.

    For a blind hole (bottom_face given): distance from the bottom
    face's centroid to the center of the cylindrical wall's OTHER
    (opening) circular boundary edge - i.e. not the one coincident with
    the bottom face.

    For a through hole (bottom_face=None): distance between the
    cylindrical wall's two circular boundary edge centers (its two
    openings at the part's outer surface).

    Returns:
        float or None: depth in mm, or None if the cylindrical face
        doesn't have exactly two circular boundary edges (not a
        well-formed simple hole - e.g. one intersected by another
        feature may not have clean circular edges left).
    """
    centers = _circle_edge_centers(cylindrical_face)
    if len(centers) != 2:
        return None

    if bottom_face is None:
        return float(np.linalg.norm(centers[0][1] - centers[1][1]))

    props = GProp_GProps()
    brepgprop_SurfaceProperties(bottom_face.topods_shape(), props)
    bottom_center = np.array(props.CentreOfMass().Coord())

    d0 = np.linalg.norm(centers[0][1] - bottom_center)
    d1 = np.linalg.norm(centers[1][1] - bottom_center)
    opening_center = centers[0][1] if d0 > d1 else centers[1][1]
    return float(np.linalg.norm(opening_center - bottom_center))


def check_ld_ratio_rule(depth_mm, diameter_mm, is_blind):
    """Rule 11 - depth-to-diameter (L/D) ratio."""
    if depth_mm is None or diameter_mm is None or diameter_mm <= 0:
        return None
    ratio = depth_mm / diameter_mm
    if is_blind:
        limit = BLIND_LD_MAX_RATIO
    else:
        limit = (THROUGH_LD_MAX_RATIO_SMALL if diameter_mm < THROUGH_LD_DIAMETER_BREAKPOINT_MM
                 else THROUGH_LD_MAX_RATIO_LARGE)
    if ratio > limit:
        return {
            "rule": "11 - depth-to-diameter (L/D) ratio",
            "measured": round(ratio, 3),
            "limit": limit,
            "violation": f"L/D ratio {ratio:.3f} exceeds maximum {limit} "
                         f"(depth={depth_mm:.3f}mm, diameter={diameter_mm:.3f}mm)",
        }
    return None


def check_bottom_thickness_rule(bottom_face, diameter_mm, candidate_faces):
    """
    Rule 12 - blind-hole bottom thickness. Finds the nearest opposing
    face to the blind hole's bottom (wall_thickness_rule.find_nearest_opposing_face)
    and flags if that measured thickness is below 20% of the hole's own
    diameter (this threshold scales with the hole, not a fixed mm value).
    """
    _, thickness_mm = find_nearest_opposing_face(bottom_face, candidate_faces)
    if thickness_mm is None:
        return None
    limit = BOTTOM_THICKNESS_MIN_RATIO * diameter_mm
    if thickness_mm < limit:
        return {
            "rule": "12 - blind-hole bottom thickness",
            "measured": round(thickness_mm, 3),
            "limit": round(limit, 3),
            "violation": f"bottom thickness {thickness_mm:.3f}mm is below 20% of hole diameter ({limit:.3f}mm)",
        }
    return None


def check_hole_spacing_rule(hole_group_faces, other_boundary_points, wall_thickness_mm=MIN_WALL_THICKNESS_MM,
                             rule_label="hole-to-edge"):
    """
    Rule 13 - hole-to-edge / hole-to-hole distance.

    wall_thickness_mm: the reference wall thickness the 1.5x multiplier
    applies to. Defaults to MIN_WALL_THICKNESS_MM (the global minimum) -
    pass an actual locally-measured thickness for that area instead when
    one is available (e.g. from a nearby wall_thickness_rule measurement),
    since the true local wall is more correct than a global constant; the
    global minimum is used as a conservative fallback when no local
    measurement is available, which is the common case for a plain
    hole-to-outer-edge check where there isn't an obvious "opposing face"
    to measure a local wall against.
    """
    hole_points = sample_group_boundary_points(hole_group_faces)
    min_dist = min_pairwise_distance(hole_points, other_boundary_points)
    if min_dist is None:
        return None
    limit = HOLE_SPACING_MIN_RATIO * wall_thickness_mm
    if min_dist < limit:
        return {
            "rule": f"13 - {rule_label} distance",
            "measured": round(min_dist, 3),
            "limit": round(limit, 3),
            "violation": f"distance {min_dist:.3f}mm is below {HOLE_SPACING_MIN_RATIO}x "
                         f"wall thickness ({limit:.3f}mm)",
        }
    return None


def check_hole_features(step_path, model, stat, feature_schema, device="cuda"):
    """
    Run the multi-face hole rules (11, 12, 13) on a single-body STEP file.

    Groups faces via feature_grouping (same predicted class, connected),
    keeps groups predicted as a hole class (Through hole / Blind hole),
    and for each: measures diameter (its cylindrical wall), depth (Rule
    11's L/D ratio input), checks bottom thickness for blind holes
    (Rule 12), and hole-to-edge / hole-to-hole spacing (Rule 13).

    Returns:
        dict: {
            "holes": [
                {
                    "face_ids": [int, ...],
                    "type": "through" | "blind",
                    "diameter_mm": float,
                    "depth_mm": float or None,
                    "violations": list[dict],
                }, ...
            ]
        }
    """
    from infer import infer_step_file  # local import: avoid a hard dependency for callers who only need the rule functions

    step_path = Path(step_path)
    reader = STEPControl_Reader()
    reader.ReadFile(str(step_path))
    reader.TransferRoots()
    body = reader.OneShape()

    graph = face_adjacency(Solid(body))
    faces_by_id = {i: graph.nodes[i]["face"] for i in graph.nodes}

    predictions = infer_step_file(step_path, model, stat, feature_schema, device)
    predicted_classes = {fid: info["predicted_class"] for fid, info in predictions.items()}

    groups = group_faces_by_class_and_adjacency(predicted_classes, graph)

    hole_groups = []
    for group in groups:
        cls = predicted_classes[group[0]]
        if cls not in HOLE_CLASSES:
            continue

        cyl_ids = [fid for fid in group if faces_by_id[fid].surface_type() == "cylinder"]
        planar_ids = [fid for fid in group if faces_by_id[fid].surface_type() == "plane"]
        if not cyl_ids:
            continue  # no cylindrical wall in this group - can't measure a hole diameter geometrically

        cyl_face = faces_by_id[cyl_ids[0]]
        diameter_mm = 2.0 * float(cyl_face.specific_surface().Radius())

        is_blind = (cls == 12)
        bottom_face = faces_by_id[planar_ids[0]] if (is_blind and planar_ids) else None
        depth_mm = measure_hole_depth(cyl_face, bottom_face)

        violations = []
        v = check_ld_ratio_rule(depth_mm, diameter_mm, is_blind)
        if v:
            violations.append(v)

        if is_blind and bottom_face is not None:
            other_faces = [f for fid, f in faces_by_id.items() if fid not in group]
            v = check_bottom_thickness_rule(bottom_face, diameter_mm, other_faces)
            if v:
                violations.append(v)

        hole_groups.append({
            "face_ids": group,
            "type": "blind" if is_blind else "through",
            "diameter_mm": round(diameter_mm, 3),
            "depth_mm": round(depth_mm, 3) if depth_mm is not None else None,
            "violations": violations,
        })

    # Rule 13: hole-to-edge (against every other face in the part, excluding
    # the hole group's own immediate (1-hop) neighbors - those necessarily
    # share a boundary edge with the hole's cylindrical wall (e.g. the
    # stock face a through-hole pierces), which would trivially measure a
    # distance of 0 to that shared edge and isn't the "is this hole too
    # close to some OTHER, separate boundary" question Rule 13 is actually
    # asking).
    undirected_graph = graph.to_undirected(as_view=True)
    for hg in hole_groups:
        group_ids = set(hg["face_ids"])
        hole_faces = [faces_by_id[fid] for fid in hg["face_ids"]]

        neighbor_ids = set()
        for fid in group_ids:
            neighbor_ids.update(undirected_graph.neighbors(fid))
        excluded_ids = group_ids | neighbor_ids

        other_points = sample_group_boundary_points(
            [f for fid, f in faces_by_id.items() if fid not in excluded_ids]
        )
        v = check_hole_spacing_rule(hole_faces, other_points, rule_label="hole-to-edge")
        if v:
            hg["violations"].append(v)

    # Rule 13: hole-to-hole (every pair of holes found in the part)
    for i in range(len(hole_groups)):
        for j in range(i + 1, len(hole_groups)):
            faces_i = [faces_by_id[fid] for fid in hole_groups[i]["face_ids"]]
            faces_j = [faces_by_id[fid] for fid in hole_groups[j]["face_ids"]]
            points_j = sample_group_boundary_points(faces_j)
            v = check_hole_spacing_rule(faces_i, points_j, rule_label="hole-to-hole")
            if v:
                hole_groups[i]["violations"].append(v)
                hole_groups[j]["violations"].append(v)

    return {"holes": hole_groups}
