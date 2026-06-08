"""
[DEPRECATED / SUPERSEDED — not used by current work, kept for reference only]
Legacy, unvalidated. Superseded by build_periodic_tf_mesh() in
breakdown/pbc_dg_analysis.py. Canonical terminology lives there:
  ḡ = reference metric;  g = total metric = ḡ + Δg;  Δg = global strain = g − ḡ;
  g_s = local per-triangle metric;  δg_s = g_s − g = W_s · Δg  (non-affine).

Periodic TF (triangulate-first) triangular mesh on a rhombic torus.

Regular N×N triangular lattice with periodicity vectors L1=[1,0], L2=[0.5,√3/2].
All N² vertices perturbed independently by eta*[cosθ, sinθ].
Topology fixed from regular lattice (TF = triangulate-first).
"""
import numpy as np


def make_periodic_tf_mesh(N, eta, seed=0):
    """Build periodic TF triangular mesh.

    Returns dict with:
        pts         : (N², 2)   perturbed vertex positions
        simplices   : (2N², 3)  triangle vertex indices
        tri_images  : (2N², 3, 2) image offsets (p,q) per vertex per triangle
        edge_vecs   : (2N², 3, 2) unwrapped edge vectors (v1-v0, v2-v0, v2-v1)
        actual_len2 : (2N², 3)  squared rest lengths from unwrapped vecs
        areas       : (2N²,)    triangle areas
        kkt_arrays  : (s1, s2, q_arr) for ALL 3N² edges (all interior on torus)
        L1, L2      : periodicity vectors
        N, eta
    """
    rng = np.random.default_rng(seed)

    L1 = np.array([1.0, 0.0])
    L2 = np.array([0.5, np.sqrt(3) / 2])

    # Reference positions on regular lattice
    n_arr = np.arange(N)
    m_arr = np.arange(N)
    nn, mm = np.meshgrid(n_arr, m_arr, indexing='ij')  # (N, N)
    ref_pts = nn.ravel()[:, None] * L1 + mm.ravel()[:, None] * L2  # (N², 2)

    # Perturbations
    angles = rng.uniform(0, 2 * np.pi, N * N)
    perturb = eta * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    pts = ref_pts + perturb  # (N², 2)

    def idx(n, m):
        return (n % N) * N + (m % N)

    # Build triangles + image offsets
    # For each rhombus cell (n,m), two triangles:
    #   T1: v0=idx(n,m), v1=idx(n+1,m), v2=idx(n,m+1)
    #   T2: v0=idx(n+1,m), v1=idx(n+1,m+1), v2=idx(n,m+1)
    n_tri = 2 * N * N
    simplices = np.zeros((n_tri, 3), dtype=np.int64)
    tri_images = np.zeros((n_tri, 3, 2), dtype=np.int64)  # (p, q) image offsets

    ti = 0
    for n in range(N):
        for m in range(N):
            # T1
            verts = [idx(n, m), idx(n + 1, m), idx(n, m + 1)]
            simplices[ti] = verts
            # Image offsets: node (n+dn, m+dm) mod N has offset (dn//N, dm//N) relative to (n,m)
            # v0=(n,m): offset (0,0)
            # v1=(n+1,m): p offset = 1 if n+1>=N else 0
            # v2=(n,m+1): q offset = 1 if m+1>=N else 0
            tri_images[ti, 0] = [0, 0]
            tri_images[ti, 1] = [1 if (n + 1) >= N else 0, 0]
            tri_images[ti, 2] = [0, 1 if (m + 1) >= N else 0]
            ti += 1

            # T2
            verts = [idx(n + 1, m), idx(n + 1, m + 1), idx(n, m + 1)]
            simplices[ti] = verts
            tri_images[ti, 0] = [1 if (n + 1) >= N else 0, 0]
            tri_images[ti, 1] = [1 if (n + 1) >= N else 0, 1 if (m + 1) >= N else 0]
            tri_images[ti, 2] = [0, 1 if (m + 1) >= N else 0]
            ti += 1

    # Box-level periodicity vectors: image offset (p,q) shifts by p*BL1 + q*BL2
    BL1 = N * L1   # total box vector in L1 direction
    BL2 = N * L2   # total box vector in L2 direction

    # Unwrapped positions per triangle
    v0_pos = pts[simplices[:, 0]] + tri_images[:, 0, 0:1] * BL1 + tri_images[:, 0, 1:2] * BL2
    v1_pos = pts[simplices[:, 1]] + tri_images[:, 1, 0:1] * BL1 + tri_images[:, 1, 1:2] * BL2
    v2_pos = pts[simplices[:, 2]] + tri_images[:, 2, 0:1] * BL1 + tri_images[:, 2, 1:2] * BL2

    # Edge vectors: [v1-v0, v2-v0, v2-v1] (3 per triangle)
    e01 = v1_pos - v0_pos
    e02 = v2_pos - v0_pos
    e12 = v2_pos - v1_pos
    edge_vecs = np.stack([e01, e02, e12], axis=1)  # (2N², 3, 2)

    actual_len2 = (edge_vecs ** 2).sum(axis=2)  # (2N², 3)

    # Areas
    cross = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]
    areas = 0.5 * np.abs(cross)

    # ── Build KKT arrays: all 3N² bonds on torus ─────────────────────────────
    # Each bond is shared by exactly 2 triangles.
    # Canonical bond key: (u, v, dp0, dp1) with canonical direction.
    # For edge (va, vb) in triangle ti with image offset (pa-pb):
    #   da = tri_images[ti, ka] - tri_images[ti, kb]  (image of va relative to vb)
    # Canonical: min of (va, da[0], da[1]) vs (vb, -da[0], -da[1])

    bond_map = {}  # canonical_key -> [(ti, ka, kb, sign), ...]

    # Edges within each triangle: (local vertex indices)
    edge_pairs = [(0, 1), (0, 2), (1, 2)]

    for ti in range(n_tri):
        for (ka, kb) in edge_pairs:
            va = simplices[ti, ka]
            vb = simplices[ti, kb]
            # Image offset of va relative to vb (integer lattice units)
            ia = tri_images[ti, ka]
            ib = tri_images[ti, kb]
            dp = ia - ib  # relative image offset (va side)

            # Canonical direction: compare (va, dp0, dp1) vs (vb, -dp0, -dp1)
            fwd = (va, int(dp[0]), int(dp[1]))
            rev = (vb, int(-dp[0]), int(-dp[1]))
            if fwd <= rev:
                key = (va, vb, int(dp[0]), int(dp[1]))
                # unwrapped edge vec = pts[va]+ia*L1+ia*L2 - (pts[vb]+ib*L1+ib*L2)
                # = e_{ka} - e_{kb} ... easier: use edge_vecs directly
                # For local pair (ka,kb): the edge vector from vb to va
                # = (pts[va]+ia) - (pts[vb]+ib)
                # edge_vecs[ti, local_edge_idx] depends on edge pair
                if (ka, kb) == (0, 1):
                    evec = edge_vecs[ti, 0]   # e01 = v1-v0, so vb=v0, va=v1 → nope
                    # e01 = v1_pos - v0_pos; here va=v0(ka=0), vb=v1(kb=1)
                    # so vec from va to vb = e01, and canonical is fwd=(v0,...) meaning
                    # we store the vec from va to vb as -e01? Let me be consistent:
                    # We always store q = [dx², 2dx·dy, dy²] where dx,dy is the edge vec
                    # direction doesn't matter for q (q is symmetric in dx→-dx)
                    evec = edge_vecs[ti, 0]
                elif (ka, kb) == (0, 2):
                    evec = edge_vecs[ti, 1]   # e02 = v2-v0
                else:  # (1, 2)
                    evec = edge_vecs[ti, 2]   # e12 = v2-v1
                bond_map[key] = {'vecs': [evec], 'tris': [(ti, +1)]}
            else:
                key = (vb, va, int(-dp[0]), int(-dp[1]))
                if (ka, kb) == (0, 1):
                    evec = edge_vecs[ti, 0]
                elif (ka, kb) == (0, 2):
                    evec = edge_vecs[ti, 1]
                else:
                    evec = edge_vecs[ti, 2]
                if key not in bond_map:
                    bond_map[key] = {'vecs': [evec], 'tris': [(ti, -1)]}
                else:
                    bond_map[key]['vecs'].append(evec)
                    bond_map[key]['tris'].append((ti, -1))
                continue

            # Already added above for fwd case; handle rev additions:
            # (this branch is unreachable — rev handled in else)

    # Second pass: add the reverse-direction entries
    # The above code has a logic bug; let me redo with cleaner approach.

    bond_map = {}
    for ti in range(n_tri):
        for ei, (ka, kb) in enumerate(edge_pairs):
            va = int(simplices[ti, ka])
            vb = int(simplices[ti, kb])
            ia = tri_images[ti, ka]
            ib = tri_images[ti, kb]
            dp = (int(ia[0] - ib[0]), int(ia[1] - ib[1]))

            fwd = (va, dp[0], dp[1])
            rev = (vb, -dp[0], -dp[1])
            if fwd <= rev:
                key = (va, vb, dp[0], dp[1])
                evec = edge_vecs[ti, ei]
            else:
                key = (vb, va, -dp[0], -dp[1])
                evec = -edge_vecs[ti, ei]

            if key not in bond_map:
                bond_map[key] = {'evec': evec, 'tris': []}
            bond_map[key]['tris'].append(ti)

    s1_list, s2_list, q_list = [], [], []
    for key, val in bond_map.items():
        tris = val['tris']
        assert len(tris) == 2, f"Bond {key} has {len(tris)} triangles, expected 2"
        evec = val['evec']
        dx, dy = evec[0], evec[1]
        s1_list.append(tris[0])
        s2_list.append(tris[1])
        q_list.append([dx * dx, 2.0 * dx * dy, dy * dy])

    s1 = np.array(s1_list, dtype=np.int64)
    s2 = np.array(s2_list, dtype=np.int64)
    q_arr = np.array(q_list, dtype=np.float64)

    return {
        'pts': pts,
        'simplices': simplices,
        'tri_images': tri_images,
        'edge_vecs': edge_vecs,
        'actual_len2': actual_len2,
        'areas': areas,
        'kkt_arrays': (s1, s2, q_arr),
        'L1': L1,
        'L2': L2,
        'BL1': BL1,  # N*L1: box periodicity vector
        'BL2': BL2,  # N*L2: box periodicity vector
        'N': N,
        'eta': eta,
    }


def build_bare_tensors(mesh, rigidities=None, rest_len2=None):
    """Compute bare elastic tensors (N_tri, 5) from mesh edge vectors.

    factor[ti, ei] = k[ti,ei] / l0²[ti,ei] / 16
    bare[ti, :] = [Σ factor·vx⁴, Σ factor·vx³vy, Σ factor·vx²vy², ...]

    Note: edge_vecs[ti] = [e01, e02, e12] but the bare tensor sums over all
    3 edges of each triangle. Edge pairing matches forward_solver_torch convention:
    edges = [(v0,v1), (v0,v2), (v1,v2)] but the code sums all 3 edges' contributions.
    """
    edge_vecs = mesh['edge_vecs']    # (N_tri, 3, 2)
    actual_len2 = mesh['actual_len2']  # (N_tri, 3)
    N_tri = len(mesh['simplices'])

    if rigidities is None:
        k = np.ones((N_tri, 3))
    else:
        k = np.asarray(rigidities)

    if rest_len2 is None:
        l2 = actual_len2
    else:
        l2 = np.asarray(rest_len2)

    factor = k / np.maximum(l2, 1e-30) / 16.0  # (N_tri, 3)

    vx = edge_vecs[:, :, 0]
    vy = edge_vecs[:, :, 1]

    bare = np.stack([
        (factor * vx ** 4).sum(axis=1),
        (factor * vx ** 3 * vy).sum(axis=1),
        (factor * vx ** 2 * vy ** 2).sum(axis=1),
        (factor * vx * vy ** 3).sum(axis=1),
        (factor * vy ** 4).sum(axis=1),
    ], axis=1)  # (N_tri, 5)

    return bare
