# -*- coding: utf-8 -*-
"""
Shared feature-grouping utility for ManuLoop's multi-face DFM rules
(rib, boss, blind-hole bottom+wall pairing, etc).

Several rules need to treat multiple faces as one feature instance before
measuring anything - e.g. a blind hole is a cylindrical wall face plus a
flat bottom face; a rib is (at least) two side faces plus a top face. The
signal used to group them is the one validated back in Week 5's
explainability testing: faces belonging to the same feature instance
share the same AAGNet predicted class AND are mutually adjacent (e.g.
faces 47/49/51, same class, mutual gradient attribution to each other).

Same class elsewhere in the part with no adjacency path between them is
NOT one feature instance - e.g. two separate, unconnected holes of the
same class must not be merged into one group.
"""


def group_faces_by_class_and_adjacency(predicted_classes, graph):
    """
    Partition a part's faces into connected components: faces sharing the
    same predicted class AND connected to each other via a chain of
    same-class adjacent faces in the face-adjacency graph.

    Args:
        predicted_classes: dict[int, int] - face_id -> predicted class
            (e.g. infer.py's infer_step_file() output, reduced to just the
            predicted_class field).
        graph: the occwl face_adjacency graph (a networkx DiGraph, as
            returned by occwl.graph.face_adjacency) covering (at least)
            every face_id in predicted_classes.

    Returns:
        list[list[int]]: one list of face_ids per connected component. A
        feature made of a single face (e.g. an isolated fillet with no
        same-class neighbor) still produces a one-element group.
    """
    # as_view=True: a read-only undirected view sharing the original graph's
    # data, rather than a deep copy - graph nodes hold occwl Face objects
    # (SWIG-wrapped OCCT geometry) which can't be deep-copied/pickled.
    undirected = graph.to_undirected(as_view=True)

    visited = set()
    groups = []
    for face_id in predicted_classes:
        if face_id in visited:
            continue

        component = []
        stack = [face_id]
        visited.add(face_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in undirected.neighbors(current):
                if neighbor in visited:
                    continue
                if predicted_classes.get(neighbor) == predicted_classes.get(current):
                    visited.add(neighbor)
                    stack.append(neighbor)

        groups.append(component)
    return groups
