# -*- coding: utf-8 -*-
"""
Shared boundary-point sampling + minimum-distance measurement for
ManuLoop's proximity-style DFM rules (Rule 13 hole-to-edge/hole-to-hole,
Rule 7 rib spacing, Rule 9 boss spacing) - all of these are really the
same underlying question: how close does one feature's boundary get to
another's (or to the rest of the part)?
"""
import numpy as np


def sample_face_boundary_points(face, n_per_edge=8):
    """
    Sample 3D points along every edge bounding a face (e.g. for a hole,
    this is its bounding circle(s); for a planar face, its outline).
    """
    points = []
    for edge in face.edges():
        if not edge.has_curve():
            continue
        bounds = edge.u_bounds()
        umin, umax = bounds.a, bounds.b
        for i in range(n_per_edge):
            u = umin + (umax - umin) * i / max(n_per_edge - 1, 1)
            points.append(edge.point(u))
    return points


def sample_group_boundary_points(faces, n_per_edge=8):
    """Sample boundary points across every face in a feature group."""
    points = []
    for face in faces:
        points.extend(sample_face_boundary_points(face, n_per_edge))
    return points


def min_pairwise_distance(points_a, points_b):
    """
    Minimum Euclidean distance between any point in points_a and any
    point in points_b. Brute-force O(n*m) - fine for the small point
    counts (tens, not thousands) these DFM proximity checks use.

    Returns:
        float or None: the minimum distance, or None if either point set
        is empty.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return None
    a = np.asarray(points_a)
    b = np.asarray(points_b)
    diffs = a[:, None, :] - b[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return float(dists.min())
