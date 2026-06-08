"""
Cluster (local-patch) response — attack the single-site uniform-stress overshoot.

For a central triangle c, relax a patch of node-radius d around it: free the nodes within d
hops of c's vertices, hold everything else at the affine position (u=0), solve the SAME
spring equilibrium K_ff u_f = -f_aff_f, and read c's non-affine metric change
  dg_c = (F_c^T F_c - I) - Delta_g,  F_c = E_def @ E_ref^-1.
As d grows the clamp recedes and dg_c -> the full simulation (overshoot->1, corr->1).
The single-site uniform-stress MF (stored W_Std, overshoot ~1.4) is the reference.

This brackets the truth: small cluster = clamped (under-shoot), single-site MF = over-shoot.
The question: how big a cluster recovers the simulation's per-triangle dg?
"""
import os, sys
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
import pbc_dg_analysis as pda  # _assemble_K_and_faff, macroscopic_dg

DATA = os.path.join(HERE, 'dg_analysis_data')
DS = [1, 2, 3, 5, 8]
ETAS = [0.1, 0.2, 0.3, 0.4, 0.5]
NTRI_SAMPLE = 120
NTRIAL = 2


def fro(a, b):
    return (a * b).sum((1, 2))


def main():
    rng = np.random.default_rng(0)
    res = {eta: {'corr': {d: [] for d in DS}, 'over': {d: [] for d in DS},
                 'mf_corr': [], 'mf_over': []} for eta in ETAS}

    for eta in ETAS:
        for t in range(NTRIAL):
            f = os.path.join(DATA, f'sample_eta{eta:.2f}_trial{t}.npz')
            if not os.path.exists(f):
                continue
            s = np.load(f, allow_pickle=True)
            F = s['F']; Dg = pda.macroscopic_dg(F)
            sx = s['simplices']; ev = s['edge_vecs']; pts = s['pts']
            n_node = len(pts); n_tri = len(sx)
            mesh = {'bond_u': s['bond_u'], 'bond_v': s['bond_v'], 'bond_R': s['bond_R'],
                    'pts': pts}
            K, faff = pda._assemble_K_and_faff(mesh, F)
            faff = faff.reshape(n_node, 2)

            # node adjacency
            adj = [set() for _ in range(n_node)]
            for u, v in zip(s['bond_u'], s['bond_v']):
                adj[int(u)].add(int(v)); adj[int(v)].add(int(u))

            def ring(seed, d):
                seen = set(seed); fr = set(seed)
                for _ in range(d):
                    nxt = set()
                    for x in fr:
                        nxt |= adj[x]
                    nxt -= seen; seen |= nxt; fr = nxt
                return seen

            sim = s['dg_sim']; mf = s['dg_Std']
            cells = rng.choice(n_tri, size=min(NTRI_SAMPLE, n_tri), replace=False)

            # accumulate per-cell dg for each d, plus sim and mf
            dgd = {d: [] for d in DS}; sims = []; mfs = []
            e01r = ev[:, 0]; e02r = ev[:, 1]
            for c in cells:
                n0, n1, n2 = int(sx[c, 0]), int(sx[c, 1]), int(sx[c, 2])
                a_ref, b_ref = e01r[c], e02r[c]
                Eref = np.array([[a_ref[0], b_ref[0]], [a_ref[1], b_ref[1]]])
                if abs(np.linalg.det(Eref)) < 1e-12:
                    continue
                ok = True
                cell_dg = {}
                for d in DS:
                    free = np.array(sorted(ring([n0, n1, n2], d)), dtype=np.int64)
                    fdof = np.concatenate([2 * free, 2 * free + 1])
                    fdof.sort()
                    try:
                        Kff = K[fdof][:, fdof].tocsc()
                        uf = spla.spsolve(Kff, -faff.ravel()[fdof])
                    except Exception:
                        ok = False; break
                    u = np.zeros((n_node, 2))
                    u.ravel()[fdof] = uf
                    a_def = a_ref @ F.T + (u[n1] - u[n0])
                    b_def = b_ref @ F.T + (u[n2] - u[n0])
                    Edef = np.array([[a_def[0], b_def[0]], [a_def[1], b_def[1]]])
                    Fc = Edef @ np.linalg.inv(Eref)
                    g = Fc.T @ Fc - np.eye(2) - Dg
                    cell_dg[d] = g
                if not ok:
                    continue
                for d in DS:
                    dgd[d].append(cell_dg[d])
                sims.append(sim[c]); mfs.append(mf[c])

            sims = np.array(sims); mfs = np.array(mfs)
            def cc(A):
                a = np.stack([A[:, 0, 0], A[:, 0, 1], A[:, 1, 1]], 1).ravel()
                b = np.stack([sims[:, 0, 0], sims[:, 0, 1], sims[:, 1, 1]], 1).ravel()
                m = np.isfinite(a) & np.isfinite(b)
                return np.corrcoef(a[m], b[m])[0, 1]
            def ov(A):
                return np.median(np.sqrt(fro(A, A)) / np.maximum(np.sqrt(fro(sims, sims)), 1e-30))
            for d in DS:
                A = np.array(dgd[d])
                res[eta]['corr'][d].append(cc(A)); res[eta]['over'][d].append(ov(A))
            res[eta]['mf_corr'].append(cc(mfs)); res[eta]['mf_over'].append(ov(mfs))

    # report
    print("Cluster (patch radius d) per-triangle dg vs simulation:")
    print(f"{'eta':>5} | " + " ".join(f"d={d:<2}" for d in DS) + " | single-site MF")
    print("  corr:")
    for eta in ETAS:
        row = " ".join(f"{np.mean(res[eta]['corr'][d]):.2f}" for d in DS)
        print(f"{eta:>5} |  {row}  |  {np.mean(res[eta]['mf_corr']):.2f}")
    print("  overshoot ||dg||/||dg_sim|| (median):")
    for eta in ETAS:
        row = " ".join(f"{np.mean(res[eta]['over'][d]):.2f}" for d in DS)
        print(f"{eta:>5} |  {row}  |  {np.mean(res[eta]['mf_over']):.2f}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(ETAS)))
    for k, eta in enumerate(ETAS):
        axes[0].plot(DS, [np.mean(res[eta]['over'][d]) for d in DS], '-o', color=cmap[k], label=f'η={eta}')
        axes[0].plot([DS[-1] + 1], [np.mean(res[eta]['mf_over'])], 'x', color=cmap[k], ms=9)
        axes[1].plot(DS, [np.mean(res[eta]['corr'][d]) for d in DS], '-o', color=cmap[k], label=f'η={eta}')
        axes[1].plot([DS[-1] + 1], [np.mean(res[eta]['mf_corr'])], 'x', color=cmap[k], ms=9)
    axes[0].axhline(1, color='gray', lw=0.6, ls=':')
    axes[0].set_xlabel('cluster radius d  (× = single-site MF)'); axes[0].set_ylabel('overshoot ||dg||/||dg_sim||')
    axes[0].set_title('Magnitude vs cluster size'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axhline(1, color='gray', lw=0.6, ls=':')
    axes[1].set_xlabel('cluster radius d  (× = single-site MF)'); axes[1].set_ylabel('corr with sim')
    axes[1].set_title('Direction vs cluster size'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle('Cluster (affine-clamped patch) response converging to the simulation', fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_cluster_response.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('\nsaved', p)


if __name__ == '__main__':
    main()
