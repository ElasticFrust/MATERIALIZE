"""
anisotropy / crystal_pointy — SHOW POINTY, and settle "is the response only 4th-order?".

A strongly sheared crystal (make_crystal, k=1, W=0 exact) whose nu(theta) is a razor-sharp spike. We
compute, on a fine theta grid, the Fourier spectrum of THREE directional functions and overlay them:

  * 1/E(theta) = S11-bar(theta)        -- a COMPLIANCE PROJECTION, LINEAR in the tensor
  * nu/E(theta) = -S12-bar(theta)      -- the other compliance projection, LINEAR in the tensor
  * nu(theta) = -S12-bar/S11-bar       -- a RATIO of the two

The point: the LINEAR compliance projections are exactly band-limited to angular harmonics {0,2,4}
(a rank-4 tensor contracted 4 times -> a quartic form -> harmonics up to 4; this is the real content
of "4th order"). But E(theta) and nu(theta) are a RECIPROCAL and a RATIO of those, so they carry ALL
harmonics 6,8,10,... -- which is exactly why they can be arbitrarily sharp/pointy/nasty. The spectrum
below shows 1/E and nu/E dying identically after n=4, while nu(theta) has a long high-order tail.
Output crystal_pointy.png (network + nu(theta) line + nu polar + Fourier spectrum).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'anisotropy'
PHI, PSI, HALF = 4.0, 1.0, 4.0                  # strong shear -> pointy nu(theta)
NTH = 1440                                       # fine theta grid over one period [0,pi)


def harmonic_spectrum(f):
    """|Fourier coeff| vs angular-harmonic order n for a pi-periodic signal sampled over [0,pi).
    FFT bin m corresponds to cos/sin(2m*theta), i.e. harmonic order n = 2m."""
    A = np.abs(np.fft.rfft(f)) / len(f)
    A[1:] *= 2                                    # one-sided amplitude
    n = 2 * np.arange(len(A))
    return n, A


def main():
    geo = C.make_crystal(PHI, PSI, half=HALF)
    geo['bond_k'] = np.ones(len(geo['bond_R'])); geo['tri_k'] = geo['bond_k'][geo['tri_bond']]
    C6 = C.sim_region_C6(geo)

    th = np.linspace(0.0, np.pi, NTH, endpoint=False)
    nu, E = C.nu_E_theta(C6, th)
    invE = 1.0 / E                                # S11-bar  (compliance projection, linear)
    nuE = nu / E                                  # -S12-bar (compliance projection, linear)

    n_nu, A_nu = harmonic_spectrum(nu)
    n_iE, A_iE = harmonic_spectrum(invE)
    n_nE, A_nE = harmonic_spectrum(nuE)

    # how much of nu's power sits ABOVE harmonic 4 (the "more than 4th order" number)
    frac_hi = (A_nu[n_nu > 4] ** 2).sum() / (A_nu[1:] ** 2).sum()
    print(f"  [crystal_pointy] phi={PHI} psi={PSI}  nu in [{nu.min():+.3f},{nu.max():+.3f}]  "
          f"E in [{E.min():.3f},{E.max():.3f}]", flush=True)
    print(f"    compliance 1/E: |a_n| beyond n=4 = {A_iE[n_iE > 4].max():.2e}  (== 0 => band-limited {{0,2,4}})",
          flush=True)
    print(f"    compliance nu/E: |a_n| beyond n=4 = {A_nE[n_nE > 4].max():.2e}", flush=True)
    print(f"    nu(theta): |a_n| beyond n=4 = {A_nu[n_nu > 4].max():.2e}  -> "
          f"{100*frac_hi:.1f}% of nu's oscillation power is ABOVE 4th order", flush=True)

    fig = plt.figure(figsize=(20, 5))
    a0 = fig.add_subplot(1, 4, 1)
    C.draw_network(a0, geo, geo['bond_k'], cmap='viridis', lw_scale=3.0)
    a0.set_title(f'crystal φ={PHI:g} ψ={PSI:g} (k=1)', fontsize=11)

    a1 = fig.add_subplot(1, 4, 2)
    a1.plot(np.degrees(th), nu, color='#d62728', lw=1.8); a1.axhline(0, color='0.6', lw=.6)
    a1.set_xlabel('θ (deg)'); a1.set_ylabel('ν(θ)'); a1.set_xlim(0, 180)
    a1.set_xticks(range(0, 181, 45)); a1.grid(alpha=.3)
    a1.set_title(f'ν(θ) — pointy  [{nu.min():+.2f}, {nu.max():+.2f}]', fontsize=11)

    a2 = fig.add_subplot(1, 4, 3, projection='polar')
    th2 = np.concatenate([th, th + np.pi])
    a2.plot(th2, np.concatenate([nu, nu]), color='#d62728', lw=1.6)
    a2.plot(th2, np.zeros_like(th2), color='0.6', lw=.5); a2.set_rorigin(min(0.0, nu.min()) * 1.05)
    a2.set_title('ν(θ) polar', fontsize=11, pad=12)

    a3 = fig.add_subplot(1, 4, 4)
    a3.semilogy(n_iE[:20], A_iE[:20] + 1e-16, 'o-', color='#1f77b4', ms=4, label='1/E(θ)  (compliance, linear)')
    a3.semilogy(n_nE[:20], A_nE[:20] + 1e-16, 's--', color='#7f7f7f', ms=3, label='ν/E(θ)  (compliance, linear)')
    a3.semilogy(n_nu[:20], A_nu[:20] + 1e-16, 'o-', color='#d62728', ms=4, label='ν(θ)  (ratio)')
    a3.axvline(4, color='k', ls=':', lw=1); a3.text(4.2, a3.get_ylim()[1] * 0.3, 'rank-4\ncutoff', fontsize=8)
    a3.set_xlabel('angular harmonic order n'); a3.set_ylabel('|Fourier amplitude|')
    a3.set_xticks(range(0, 21, 2)); a3.grid(alpha=.3, which='both'); a3.legend(fontsize=8)
    a3.set_title('spectra: compliance band-limited {0,2,4}, ν is NOT', fontsize=11)

    fig.suptitle(f'{CASE} / crystal_pointy — a k=1 crystal with a razor-sharp ν(θ); '
                 f'{100*frac_hi:.0f}% of its power is above 4th order', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(C.savedir(CASE), 'crystal_pointy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


if __name__ == '__main__':
    main()
