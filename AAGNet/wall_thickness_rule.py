# -*- coding: utf-8 -*-
"""
Wall thickness measurement for ManuLoop DFM rules.

Rule 1 (minimum wall thickness) directly, and a shared primitive
(measure_wall_thickness) reused by Rule 12 (blind-hole bottom thickness)
and Rule 5 (rib base thickness), since all three are geometrically the
same problem: the distance between two roughly-parallel, opposite-facing
faces - verified point by point so the measurement can't "cheat" by
measuring to a point that lies past the opposing face's actual trimmed
edge (which would understate how thin the wall really is near a boundary).

Scoped to planar faces only, consistent with the same non-curved scoping
decision applied to ribs elsewhere in this pass - a curved "wall" (e.g. a
constant-thickness shell over a cylindrical boss) is a materially harder
problem (the opposing point isn't a simple plane-ray intersection) and is
out of scope here.
"""
import numpy as np
from occwl.face import Face

MIN_WALL_THICKNESS_MM = 1.5  # Rule 1: table's "functional min"; typical range is 2.0-3.5mm
PARALLEL_ANGLE_TOL_DEG = 10.0  # how anti-parallel two normals must be to call faces "opposite sides of a wall"


def is_point_inside_face(face: Face, point_3d) -> bool:
    """
    Test whether a 3D point (assumed to lie on or near face's underlying
    surface) falls within the face's actual trimmed/bounded region, not
    just its infinite underlying surface.

    Reuses occwl's Face.inside(), which wraps BRepTopAdaptor_FClass2d - the
    standard OCCT tool for exactly this (2D point-in-wire classification
    in a face's parametric domain, i.e. does this UV point fall inside
    the face's outer wire and outside any inner/hole wires). Face's own
    point_to_parameter() projects the 3D point onto the face's surface to
    get the UV coordinate .inside() needs.
    """
    uv = face.point_to_parameter(np.asarray(point_3d, dtype=float))
    return face.inside(uv)


def _sample_interior_uv_points(face: Face, grid_n=7, max_points=8):
    """
    Sample a handful of (uv, point_3d) pairs confirmed to lie inside
    face's actual trimmed region (not just its parametric bounding box,
    which can include area outside the face for non-rectangular/trimmed
    faces, e.g. a rectangular plate face with a hole cut through it).

    Walks a grid_n x grid_n grid across the face's UV bounds and keeps
    points where Face.inside() is True, stopping once max_points are
    found. This evenly samples area only for planar faces (the only case
    this module handles); a curved face's UV grid does not evenly sample
    its area, but that's out of scope here.
    """
    bounds = face.uv_bounds()
    umin, vmin = bounds.min_point()
    umax, vmax = bounds.max_point()

    samples = []
    for i in range(grid_n):
        for j in range(grid_n):
            u = umin + (umax - umin) * (i + 0.5) / grid_n
            v = vmin + (vmax - vmin) * (j + 0.5) / grid_n
            if face.inside((u, v)):
                samples.append(((u, v), face.point((u, v))))
                if len(samples) >= max_points:
                    return samples
    return samples


def are_parallel_opposite(face_a: Face, face_b: Face, angle_tol_deg=PARALLEL_ANGLE_TOL_DEG) -> bool:
    """
    Check whether two faces are (roughly) parallel planes with opposite-
    facing normals - i.e. plausibly the two sides of one wall. Scoped to
    planar faces only (see module docstring); returns False for anything
    else, including two parallel-but-same-facing planes (that's not a
    wall, that's two faces pointing the same way).
    """
    if face_a.surface_type() != "plane" or face_b.surface_type() != "plane":
        return False

    samples_a = _sample_interior_uv_points(face_a, max_points=3)
    samples_b = _sample_interior_uv_points(face_b, max_points=3)
    if not samples_a or not samples_b:
        return False

    normal_a = np.mean([face_a.normal(uv) for uv, _ in samples_a], axis=0)
    normal_b = np.mean([face_b.normal(uv) for uv, _ in samples_b], axis=0)
    norm_a_len = np.linalg.norm(normal_a)
    norm_b_len = np.linalg.norm(normal_b)
    if norm_a_len < 1e-9 or norm_b_len < 1e-9:
        return False
    normal_a = normal_a / norm_a_len
    normal_b = normal_b / norm_b_len

    cos_angle = np.clip(np.dot(normal_a, -normal_b), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg <= angle_tol_deg


def measure_wall_thickness(face_a: Face, face_b: Face, grid_n=7, max_samples=8):
    """
    Measure the wall thickness between two faces presumed to be opposite
    sides of a wall (roughly parallel planes, opposite-facing normals).

    For each confirmed-interior sample point on face_a, casts a ray along
    face_a's normal at that point and intersects it with face_b's
    (infinite) plane, then verifies the hit point actually lies within
    face_b's trimmed region via is_point_inside_face() - without this
    check the measurement could silently land past face_b's actual edge
    and understate how thin the wall really is near a boundary.

    Returns:
        float: the MINIMUM valid thickness found across all sampled
        points, in mm - a wall's failure point is its thinnest point, not
        its average. None if the faces aren't a valid parallel-plane pair
        (see are_parallel_opposite) or no sample point on face_a had a
        valid opposing point on face_b.
    """
    if not are_parallel_opposite(face_a, face_b):
        return None

    plane_b = face_b.specific_surface()  # gp_Pln, guaranteed by are_parallel_opposite's plane check
    origin_b = np.array(plane_b.Location().Coord())
    axis_b_dir = plane_b.Axis().Direction()
    normal_b_geom = np.array([axis_b_dir.X(), axis_b_dir.Y(), axis_b_dir.Z()])

    samples_a = _sample_interior_uv_points(face_a, grid_n=grid_n, max_points=max_samples)
    if not samples_a:
        return None

    thicknesses = []
    for uv_a, point_a in samples_a:
        normal_a = face_a.normal(uv_a)
        denom = np.dot(normal_a, normal_b_geom)
        if abs(denom) < 1e-6:
            continue  # ray parallel to face_b's plane - shouldn't happen given are_parallel_opposite passed
        t = np.dot(origin_b - point_a, normal_b_geom) / denom
        hit_point = point_a + t * normal_a
        if is_point_inside_face(face_b, hit_point):
            thicknesses.append(abs(t))  # normal_a is unit length, so |t| is the physical distance

    if not thicknesses:
        return None
    return min(thicknesses)


def find_nearest_opposing_face(face: Face, candidate_faces, angle_tol_deg=PARALLEL_ANGLE_TOL_DEG):
    """
    Among candidate_faces, find the one that is a valid parallel-opposite
    partner to `face` (see are_parallel_opposite) with the smallest
    measured wall thickness to it. Used e.g. to find "the outer face on
    the other side of a blind hole's bottom" without knowing in advance
    which face in the part that is.

    Returns:
        (Face, float) or (None, None): the best-matching face and its
        measured thickness, or (None, None) if no candidate qualifies.
    """
    best_face = None
    best_thickness = None
    for candidate in candidate_faces:
        if candidate is face:
            continue
        thickness = measure_wall_thickness(face, candidate)
        if thickness is None:
            continue
        if best_thickness is None or thickness < best_thickness:
            best_thickness = thickness
            best_face = candidate
    return best_face, best_thickness


def check_wall_thickness_rule(face_a: Face, face_b: Face):
    """
    Rule 1 - minimum wall thickness, checked between an explicit pair of
    faces presumed to be opposite sides of one wall.

    Returns None if the faces aren't a valid wall pair (see
    are_parallel_opposite) or the thickness meets the minimum. Returns a
    violation dict otherwise.
    """
    thickness_mm = measure_wall_thickness(face_a, face_b)
    if thickness_mm is None:
        return None
    if thickness_mm < MIN_WALL_THICKNESS_MM:
        return {
            "rule": "1 - minimum wall thickness",
            "measured": round(thickness_mm, 3),
            "limit": MIN_WALL_THICKNESS_MM,
            "violation": f"thickness {thickness_mm:.3f}mm is below minimum {MIN_WALL_THICKNESS_MM}mm",
        }
    return None
