"""
Local elastic response OF THE PATCH REGIONS, measured from the SIMULATION, two ways:
  A) region-averaged per-triangle tensor  ⟨C(s)⟩   (C(s)=(I+W)ᵀA(I+W) from PH.relax) — the design's
     definition; ν,E via region_phys_C6.
  B) field average: from the 3 macro modes, area-weighted ⟨ε⟩ and ⟨σ⟩ over the region → effective
     Voigt tensor Cv=⟨σ⟩⟨ε⟩⁻¹ → ν,E.  (σ=A:ε; stress in relative units, so ν is meaningful and E is
     up to the same constant everywhere.)
Validated on the whole cell (B must reproduce the homogenised value). Regions: R_E (stiff), R_ν (aux).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'verification_tools'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'Phase 2'))
import _common as C
import physical_homog as PH
import test_cluster_rigidity as TR
import test_cluster_Ceff as CE

TOPOS = ['regular', 'disorder_hi']


def augment(geo):
    tv = np.asarray(geo['tri_verts']); p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    geo['edge_vecs'] = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    geo['actual_len2'] = (geo['edge_vecs'] ** 2).sum(2)
    geo['simplices'] = np.asarray(geo['simplices'])
    return geo


def mode_fields(geo):
    """Per-triangle strain ε_k and stress σ_k (2x2) for the 3 unit macro modes, from the simulation."""
    ev, sx = geo['edge_vecs'], geo['simplices']
    u = PH.relax(geo, np.arange(2, 2 * len(geo['pts'])), TR.assemble_K_faff)
    eps = [CE.tri_metric_change(ev, sx, PH.Fk[k], u[k]) / PH.DELTA for k in range(3)]
    bare = TR.bare_tensor(geo)
    A0, A1, A2, A3, A4 = (bare[:, i] for i in range(5))
    sig = []
    for e in eps:
        exx, eyy, exy = e[:, 0, 0], e[:, 1, 1], e[:, 0, 1]
        s = np.zeros_like(e)
        s[:, 0, 0] = A0 * exx + 2 * A1 * exy + A2 * eyy
        s[:, 0, 1] = s[:, 1, 0] = A1 * exx + 2 * A2 * exy + A3 * eyy
        s[:, 1, 1] = A2 * exx + 2 * A3 * exy + A4 * eyy
        sig.append(s)
    return eps, sig


def field_nuE(eps, sig, areas, idx):
    """Method B: area-weighted ⟨ε⟩,⟨σ⟩ over `idx`; Cv=⟨σ⟩⟨ε⟩⁻¹ (engineering Voigt) → ν(θ),E(θ)."""
    w = areas[idx]; W = w.sum()
    def vgt(t, eng):                       # [xx, yy, (2)xy] area-weighted mean over idx
        return np.array([(w * t[idx, 0, 0]).sum(), (w * t[idx, 1, 1]).sum(),
                         (eng * w * t[idx, 0, 1]).sum()]) / W
    E3 = np.stack([vgt(e, 2.0) for e in eps], 1)      # strain engineering (γ=2ε)
    S3 = np.stack([vgt(s, 1.0) for s in sig], 1)      # stress
    Cv = S3 @ np.linalg.inv(E3)                       # 3x3 Voigt
    c6 = [Cv[0, 0], Cv[0, 2], Cv[0, 1], Cv[2, 2], Cv[1, 2], Cv[1, 1]]
    nu, Eth = C.nu_E_theta(np.array(c6), C.ANG)
    return float(nu.mean()), float(Eth.mean())


def main():
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for col, topo in enumerate(TOPOS):
        geo, k, C6, meta = C.load_network(os.path.join(HERE, 'networks', f'patch__{topo}.npz'))
        augment(geo); areas = np.asarray(geo['areas']); cen = np.asarray(geo['centroids'])
        RE, RN = meta['region']
        regions = {'whole': np.arange(len(areas)),
                   'R_E (stiff)': np.where(((cen - RE['center']) ** 2).sum(1) < RE['radius'] ** 2)[0],
                   'R_nu (aux)': np.where(((cen - RN['center']) ** 2).sum(1) < RN['radius'] ** 2)[0]}
        eps, sig = mode_fields(geo)

        def magf(t):
            return np.sqrt(t[:, 0, 0] ** 2 + 2 * t[:, 0, 1] ** 2 + t[:, 1, 1] ** 2)
        FORC = [('dilation', (1, 1, 0)), ('pure shear', (1, -1, 0)), ('simple shear', (0, 0, 2))]
        for fname, (c0, c1, c2) in FORC:                          # mean |strain|,|stress| per region
            e = c0 * eps[0] + c1 * eps[1] + c2 * eps[2]
            s = c0 * sig[0] + c1 * sig[1] + c2 * sig[2]
            me, ms = magf(e), magf(s)
            for name, idx in regions.items():
                print(f"  [{topo:11s} {fname:12s} {name:11s}] <|strain|>={me[idx].mean():.3f} "
                      f"<|stress|>={ms[idx].mean():.4f}", flush=True)

        labels, nuA, nuB = [], [], []
        for name, idx in regions.items():
            nA, EA = C.c6_nuE(C.region_phys_C6(geo, C6, idx))     # Method A (per-triangle tensor avg)
            nB, EB = field_nuE(eps, sig, areas, idx)              # Method B (field average sig vs eps)
            rows.append((topo, name, f'{nA:+.3f}', f'{EA:.3f}', f'{nB:+.3f}', f'{EB:.3f}'))
            labels.append(name); nuA.append(nA); nuB.append(nB)
            print(f"  {topo:12s} {name:11s} | A(tensor) nu={nA:+.3f} E={EA:.3f} | "
                  f"B(field) nu={nB:+.3f} E={EB:.3f}", flush=True)
        x = np.arange(len(labels))
        ax[col].bar(x - 0.2, nuA, 0.4, label='A: ⟨C(s)⟩ (per-tri)', color='#1f77b4')
        ax[col].bar(x + 0.2, nuB, 0.4, label='B: ⟨σ⟩/⟨ε⟩ (field)', color='#d62728')
        ax[col].axhline(0, color='k', lw=.5); ax[col].set_xticks(x); ax[col].set_xticklabels(labels)
        ax[col].set_ylabel('local ν'); ax[col].set_title(f'{topo}: local ν from the simulation')
        ax[col].legend(fontsize=8); ax[col].grid(alpha=.3, axis='y')
    C.write_csv(os.path.join(HERE, 'local_response.csv'),
                ['topology', 'region', 'nu_tensoravg', 'E_tensoravg', 'nu_field', 'E_field'], rows)
    fig.suptitle('Local response of the patch regions from the SIMULATION — two independent estimates',
                 fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'local_response.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved local_response.png + local_response.csv')


if __name__ == '__main__':
    main()
