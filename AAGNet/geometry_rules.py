# -*- coding: utf-8 -*-
"""
Single-face geometry measurement + DFM rule checking for ManuLoop.

Scope for this pass (Week 6, first cut):
    Rule 3/4  - internal/external fillet radius
    Rule 10   - minimum cored-hole diameter

Explicitly OUT of scope for this pass (deferred):
    Rule 15   - draft angle: requires a pull-direction assumption that
                can't be verified from MFCAD++ geometry alone.
    Rule 11   - hole depth-to-diameter (L/D) ratio: depth measurement
                is closer to a multi-face problem than a clean single-face
                property; pushed to the relational-rules pass.

Each rule function takes an occwl Face and returns None if the face
passes (or isn't the relevant surface type), or a dict describing the
violation if it fails.
"""
from pathlib import Path

from occwl.face import Face
from occwl.solid import Solid
from occwl.graph import face_adjacency
from OCC.Core.STEPControl import STEPControl_Reader

# ---------------------------------------------------------------------------
# Thresholds from the Week 1 DFM rule table (aluminum HPDC)
# ---------------------------------------------------------------------------
MIN_FILLET_RADIUS_MM = 0.5   # Rule 3/4: practical minimum, never 0
MIN_HOLE_DIAMETER_MM = 6.0   # Rule 10: NADCA cored-hole floor

# ---------------------------------------------------------------------------
# Which AAGNet predicted classes each rule applies to (0-24, see
# feature_labels.txt in the MFCAD++ dataset root for the full class list).
# Geometry alone can't tell a hole-forming cylinder apart from a boss, pin,
# or other cylindrical feature - AAGNet's predicted class is what narrows a
# rule down to the faces it actually means to check. Extend this dict (not
# the rule functions) when adding new rules in later passes.
# ---------------------------------------------------------------------------
RULE_APPLICABLE_CLASSES = {
    "fillet_radius": {23},      # Round
    "hole_diameter": {1, 12},   # Through hole, Blind hole
}


def check_fillet_rule(face: Face, predicted_class: int = None):
    """
    Rule 3/4 - internal/external fillet radius.

    Fillets are rounded transitions between two flat faces, geometrically
    a circular arc swept along a straight edge - which is a cylindrical
    surface. Toroidal surfaces are also included since a fillet wrapping
    around a curved edge (rather than a straight one) is a torus.

    Args:
        predicted_class: AAGNet's predicted class (0-24) for this face, if
            known. When given, the rule only fires if predicted_class is in
            RULE_APPLICABLE_CLASSES["fillet_radius"] - narrows this from
            "any cylindrical/toroidal face" down to faces AAGNet actually
            classified as a fillet (Round). When None (default), falls
            back to geometry-only gating (surface type alone) - kept for
            standalone geometry testing without running inference.

    Returns None if the face isn't cylindrical/toroidal, isn't the right
    predicted class (when given), or its radius meets the minimum. Returns
    a violation dict otherwise.
    """
    if predicted_class is not None and predicted_class not in RULE_APPLICABLE_CLASSES["fillet_radius"]:
        return None

    surf_type = face.surface_type()
    if surf_type not in ("cylinder", "torus"):
        return None

    # occwl's Face has no .radius() accessor (verified against the installed
    # occwl 2.0.2: Face.specific_surface() is what exposes the raw OCCT
    # geometry primitive - gp_Cylinder for "cylinder", gp_Torus for "torus").
    # For a torus, the fillet's actual rounding radius is MinorRadius() (the
    # tube radius) - MajorRadius() is the distance from the revolution axis
    # to the tube center and is unrelated to how sharp the transition is.
    surf = face.specific_surface()
    if surf_type == "cylinder":
        radius_mm = float(surf.Radius())
    else:
        radius_mm = float(surf.MinorRadius())

    if radius_mm < MIN_FILLET_RADIUS_MM:
        return {
            "rule": "3/4 - fillet radius",
            "measured": round(radius_mm, 3),
            "limit": MIN_FILLET_RADIUS_MM,
            "violation": f"radius {radius_mm:.3f}mm is below minimum {MIN_FILLET_RADIUS_MM}mm",
        }
    return None


def check_hole_diameter_rule(face: Face, predicted_class: int = None):
    """
    Rule 10 - minimum cored-hole diameter.

    Args:
        predicted_class: AAGNet's predicted class (0-24) for this face, if
            known. When given, the rule only fires if predicted_class is in
            RULE_APPLICABLE_CLASSES["hole_diameter"] (Through hole, Blind
            hole) - this is what actually distinguishes a real hole from
            any other cylindrical face (boss, pin, fillet-adjacent
            cylinder, ...), which geometry alone cannot. When None
            (default), falls back to geometry-only gating (surface type
            alone) - kept for standalone geometry testing without running
            inference, but not trustworthy on its own for that reason.

    Returns None if the face isn't cylindrical, isn't the right predicted
    class (when given), or its diameter meets the minimum. Returns a
    violation dict otherwise.
    """
    if predicted_class is not None and predicted_class not in RULE_APPLICABLE_CLASSES["hole_diameter"]:
        return None

    if face.surface_type() != "cylinder":
        return None

    # See check_fillet_rule for why specific_surface() is used instead of a
    # (nonexistent) Face.radius() accessor.
    diameter_mm = 2.0 * float(face.specific_surface().Radius())

    if diameter_mm < MIN_HOLE_DIAMETER_MM:
        return {
            "rule": "10 - minimum cored-hole diameter",
            "measured": round(diameter_mm, 3),
            "limit": MIN_HOLE_DIAMETER_MM,
            "violation": f"diameter {diameter_mm:.3f}mm is below minimum {MIN_HOLE_DIAMETER_MM}mm",
        }
    return None


def check_face(face: Face, predicted_class: int = None):
    """
    Runs every rule in scope for this pass on one occwl Face.
    Returns a list of violation dicts - empty if the face passes
    everything applicable to it.
    """
    violations = []

    v = check_fillet_rule(face, predicted_class)
    if v:
        violations.append(v)

    v = check_hole_diameter_rule(face, predicted_class)
    if v:
        violations.append(v)

    return violations


def check_part(step_path, model, stat, feature_schema, device="cuda"):
    """
    Run AAGNet inference on a single-body STEP file and check every face
    against the DFM rules in scope for this pass, gated by AAGNet's
    predicted class for that face (see RULE_APPLICABLE_CLASSES).

    Measurements are taken on the ORIGINAL (unscaled) geometry, not the
    unit-box-scaled shape AAGNetSegmentor actually classifies - fillet
    radius and hole diameter need real physical mm, not the network's
    normalized input scale. The face_id -> Face mapping is built
    separately from the original body via the same face_adjacency call
    AAGExtractor uses internally; verified empirically (same file, scaled
    vs. unscaled) that this indexing is identical either way since it's
    purely topological (occwl's EntityMapper), not geometric - scaling a
    shape can't add faces, remove faces, or change adjacency, so it can't
    change which index a face gets.

    Returns:
        dict[int, dict]: face_id -> {
            "predicted_class": int (0-24),
            "violations": list[dict] (empty if the face passes everything
                applicable to it),
        }
    """
    from infer import infer_step_file  # local import: avoid a hard dependency for callers who only need check_face

    step_path = Path(step_path)
    reader = STEPControl_Reader()
    reader.ReadFile(str(step_path))
    reader.TransferRoots()
    body = reader.OneShape()

    graph = face_adjacency(Solid(body))
    faces_by_id = {face_idx: graph.nodes[face_idx]["face"] for face_idx in graph.nodes}

    predictions = infer_step_file(step_path, model, stat, feature_schema, device)

    results = {}
    for face_id, face in faces_by_id.items():
        predicted_class = predictions[face_id]["predicted_class"]
        results[face_id] = {
            "predicted_class": predicted_class,
            "violations": check_face(face, predicted_class),
        }
    return results
