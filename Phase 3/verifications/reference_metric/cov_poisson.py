"""
Covariant elastic moduli, contracting the compliance with ḡ-orthonormal directions:
    ν(m) = − S(n,n,m,m) / S(m,m,m,m),        ḡ(m,m)=ḡ(n,n)=1, ḡ(m,n)=0
    E(m) = 1 / [ √det ḡ · S(m,m,m,m) ]       (√det ḡ = common physical volume)
Check the two descriptions of the SAME material collapse onto one curve:
    (a) elongated network, ḡ=I;   (b) regular network, full ḡ=diag(1,ψ²).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import _common as C
from forward_solver_dgbar import forward_dgbar


def affine(geo, M):
    """Apply affine map M to every geometric field of a periodic geo (topology preserved)."""
    g = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in geo.items()}
    for key in ('pts', 'edge_vecs', 'bond_R', 'centroids', 'tri_verts'):
        if key in g:
            g[key] = geo[key] @ M.T
    g['actual_len2'] = (g['edge_vecs'] ** 2).sum(-1)
    g['areas'] = geo['areas'] * abs(np.linalg.det(M))
    g['BL1'] = M @ geo['BL1']; g['BL2'] = M @ geo['BL2']
    return g


def _gbar_frame(gbar):
    g = np.asarray(gbar, float)
    e1 = np.array([1.0, 0.0]); e1 = e1 / np.sqrt(e1 @ g @ e1)
    e2 = np.array([0.0, 1.0]); e2 = e2 - (e2 @ g @ e1) * e1; e2 = e2 / np.sqrt(e2 @ g @ e2)
    return e1, e2, np.linalg.det(g)


def nu_cov(C6, gbar, thetas):
    """Covariant ν(θ): m,n ḡ-orthonormal (e1 along x, e2 = ḡ-orthonormalized y)."""
    Sc = C._compliance_tensor(C6)
    e1, e2, _ = _gbar_frame(gbar)
    out = []
    for th in thetas:
        m = np.cos(th) * e1 + np.sin(th) * e2
        n = -np.sin(th) * e1 + np.cos(th) * e2
        Smm = np.einsum('ijkl,i,j,k,l', Sc, m, m, m, m)
        Snn = np.einsum('ijkl,i,j,k,l', Sc, n, n, m, m)
        out.append(-Snn / Smm)
    return np.array(out)


def E_cov(C6, gbar, thetas):
    """Covariant E(θ) = 1/(√det ḡ · S(m,m,m,m)), m ḡ-unit."""
    Sc = C._compliance_tensor(C6)
    e1, e2, detg = _gbar_frame(gbar)
    out = []
    for th in thetas:
        m = np.cos(th) * e1 + np.sin(th) * e2
        Smm = np.einsum('ijkl,i,j,k,l', Sc, m, m, m, m)
        out.append(1.0 / (np.sqrt(detg) * Smm))
    return np.array(out)


def _demo():
    psi = 1.3
    F = np.array([[1.0, 0.0], [0.0, psi]]); gbar_full = F.T @ F
    reg = C.make_lattice(1.0, 1.0, half=6.0); n = len(reg['simplices'])
    elong = affine(reg, F); ne = len(elong['simplices'])

    he = forward_dgbar(elong, 1.0, np.tile(np.eye(2), (ne, 1, 1)))        # (a) ḡ=I
    rf = forward_dgbar(reg, 1.0, np.tile(gbar_full, (n, 1, 1)))           # (b) full ḡ

    th = np.linspace(0.0, np.pi, 361)
    nu_e = nu_cov(he['elastic_tensor'], np.eye(2), th);  nu_r = nu_cov(rf['elastic_tensor'], gbar_full, th)
    E_e = E_cov(he['elastic_tensor'], np.eye(2), th);    E_r = E_cov(rf['elastic_tensor'], gbar_full, th)

    print(f"ψ={psi}   ḡ_full=diag(1,{psi**2:.2f})")
    print(f"covariant ν: max |elongated − regular| = {np.max(np.abs(nu_e - nu_r)):.2e}")
    print(f"covariant E: max |elongated − regular| = {np.max(np.abs(E_e - E_r)):.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax[0].plot(np.degrees(th), nu_e, color='#1f77b4', lw=2.6, label='elongated network, ḡ=I')
    ax[0].plot(np.degrees(th), nu_r, color='#d62728', lw=1.6, ls='--', label='regular network, full ḡ')
    ax[0].axhline(1 / 3, color='0.7', lw=.6); ax[0].set_ylabel('ν_cov(θ)'); ax[0].legend()
    ax[1].plot(np.degrees(th), E_e, color='#1f77b4', lw=2.6, label='elongated network, ḡ=I')
    ax[1].plot(np.degrees(th), E_r, color='#d62728', lw=1.6, ls='--', label='regular network, full ḡ')
    ax[1].set_ylabel('E_cov(θ)'); ax[1].legend()
    for a in ax:
        a.set_xlabel('θ (deg)'); a.set_xlim(0, 180); a.set_xticks(range(0, 181, 30)); a.grid(alpha=.3)
    fig.suptitle(f'Covariant ν(θ) and E(θ): the two descriptions collapse   ψ={psi}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(HERE, 'cov_poisson.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"saved {os.path.basename(p)}")


if __name__ == '__main__':
    _demo()
