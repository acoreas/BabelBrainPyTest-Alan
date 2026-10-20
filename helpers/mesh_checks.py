"""
Mesh checks for tests that don't have truth meshes to compare against.

points_inside() tells whether points are inside a triangle mesh without trimesh's contains(), which
needs rtree (not in BabelBrain's environments) and is very slow on the ~500k face charm meshes.
check_intersection() checks the properties any correct intersection of two solids has.
"""
import numpy as np


def points_inside(mesh, points):
    """True for points inside `mesh`: majority vote of the ray-crossing parity along +x, +y and +z.

    Only triangles whose projected bounding box contains the ray are tested, which keeps it fast on
    large meshes, and the vote keeps it robust to small holes (e.g. charm meshes that aren't closed)."""
    tri = np.asarray(mesh.triangles, np.float64)
    points = np.asarray(points, np.float64)
    votes = np.zeros(len(points), int)
    for axis in range(3):
        u, v = [d for d in range(3) if d != axis]
        lo, hi = tri[:, :, [u, v]].min(axis=1), tri[:, :, [u, v]].max(axis=1)
        for i, p in enumerate(points):
            cand = np.nonzero((lo[:, 0] <= p[u]) & (hi[:, 0] >= p[u]) & (lo[:, 1] <= p[v]) & (hi[:, 1] >= p[v]))[0]
            if cand.size == 0:
                continue
            a, b, c = tri[cand, 0], tri[cand, 1], tri[cand, 2]
            # Barycentric coordinates of the ray's footprint in each projected triangle
            e1, e2, w = (b - a)[:, [u, v]], (c - a)[:, [u, v]], p[[u, v]] - a[:, [u, v]]
            den = e1[:, 0]*e2[:, 1] - e2[:, 0]*e1[:, 1]
            ok = np.abs(den) > 1e-12
            safe = np.where(ok, den, 1.0)
            l1 = np.where(ok, (w[:, 0]*e2[:, 1] - e2[:, 0]*w[:, 1]) / safe, -1.0)
            l2 = np.where(ok, (e1[:, 0]*w[:, 1] - w[:, 0]*e1[:, 1]) / safe, -1.0)
            hit = ok & (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
            depth = a[:, axis] + l1*(b - a)[:, axis] + l2*(c - a)[:, axis]
            votes[i] += int(np.count_nonzero(hit & (depth > p[axis]))) % 2
    return votes >= 2


def check_intersection(result, mesh1, mesh2, n_points=300, min_agreement=0.99, seed=0):
    """Problems (list of strings, empty when fine) with `result` as the intersection of two solids:
    - it is a closed surface
    - its bounding box is inside both inputs' bounding boxes
    - its volume is no larger than either closed input's volume
    - random points around it are inside `result` exactly when they are inside both inputs
      (at least `min_agreement` of them; points within a facet of a surface can go either way)
    Also returns the measured point agreement."""
    problems = []
    if not result.is_watertight:
        problems.append("result is not a closed surface")

    size = np.ptp(result.bounds, axis=0).max()
    tol = 1e-6 + 1e-3*size
    for name, mesh in (('mesh1', mesh1), ('mesh2', mesh2)):
        if np.any(result.bounds[0] < mesh.bounds[0] - tol) or np.any(result.bounds[1] > mesh.bounds[1] + tol):
            problems.append(f"result extends outside the bounding box of {name}")
        if mesh.is_volume and result.volume > mesh.volume*(1 + 1e-3):
            problems.append(f"result volume {result.volume:.6g} is larger than {name} volume {mesh.volume:.6g}")

    lo, hi = result.bounds
    margin = 0.05*(hi - lo)
    points = np.random.default_rng(seed).uniform(lo - margin, hi + margin, (n_points, 3))
    expected = points_inside(mesh1, points) & points_inside(mesh2, points)
    agreement = float(np.mean(points_inside(result, points) == expected))
    if agreement < min_agreement:
        problems.append(f"only {100*agreement:.1f}% of sample points agree with inside(mesh1) and inside(mesh2)")
    return problems, agreement
