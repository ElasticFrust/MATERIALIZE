"""Benchmark: Original solver vs Woodbury-optimized solver.

Generates networks with generate_foam_points(size=(7,7), eta)
for 10 eta values from 0.1 to 0.49, 7 trials each.
Compares correctness and timing.
"""
import numpy as np
import time
import copy
import Disc_2_Cont_original as orig
import Disc_2_Cont_optimized as opt


def run_benchmark():
    eta_values = np.linspace(0.1, 0.49, 10)
    n_trials = 7
    size = (7, 7)

    results = []

    print(f"{'eta':>6s} | {'trial':>5s} | {'N_tri':>5s} | "
          f"{'t_orig (s)':>10s} | {'t_opt (s)':>10s} | {'speedup':>8s} | "
          f"{'tensor_err':>10s} | {'Y_err':>10s} | {'nu_err':>10s}")
    print("-" * 100)

    for eta in eta_values:
        for trial in range(n_trials):
            seed = int(eta * 10000) + trial
            np.random.seed(seed)

            # Generate network
            DT_orig = orig.generate_foam_points(size, eta)
            DT_opt = copy.deepcopy(DT_orig)
            n_tri = len(DT_orig.simplices)

            # Time original solver
            t0 = time.perf_counter()
            orig.analyze_elastic_struct(DT_orig)
            t_orig = time.perf_counter() - t0

            # Time optimized solver
            t0 = time.perf_counter()
            opt.analyze_elastic_struct(DT_opt)
            t_opt = time.perf_counter() - t0

            # Compare results
            tensor_err = np.max(np.abs(
                DT_orig.totalElasticTensor - DT_opt.totalElasticTensor
            ) / (np.abs(DT_orig.totalElasticTensor) + 1e-30))

            y_rel_err = abs(DT_orig.YoungsModulus - DT_opt.YoungsModulus) / (abs(DT_orig.YoungsModulus) + 1e-30)
            nu_rel_err = abs(DT_orig.PoissonsRatio - DT_opt.PoissonsRatio) / (abs(DT_orig.PoissonsRatio) + 1e-30)

            speedup = t_orig / t_opt if t_opt > 0 else float('inf')

            results.append({
                'eta': eta,
                'trial': trial,
                'n_tri': n_tri,
                't_orig': t_orig,
                't_opt': t_opt,
                'speedup': speedup,
                'tensor_err': tensor_err,
                'y_err': y_rel_err,
                'nu_err': nu_rel_err,
            })

            print(f"{eta:6.3f} | {trial:5d} | {n_tri:5d} | "
                  f"{t_orig:10.4f} | {t_opt:10.4f} | {speedup:8.1f}x | "
                  f"{tensor_err:10.2e} | {y_rel_err:10.2e} | {nu_rel_err:10.2e}")

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    speedups = [r['speedup'] for r in results]
    tensor_errs = [r['tensor_err'] for r in results]
    y_errs = [r['y_err'] for r in results]
    nu_errs = [r['nu_err'] for r in results]
    t_origs = [r['t_orig'] for r in results]
    t_opts = [r['t_opt'] for r in results]
    n_tris = [r['n_tri'] for r in results]

    print(f"\nNumber of triangles: {min(n_tris)} - {max(n_tris)} (mean {np.mean(n_tris):.0f})")
    print(f"\nOriginal solver time:  {np.mean(t_origs):.4f}s mean, {np.min(t_origs):.4f}s min, {np.max(t_origs):.4f}s max")
    print(f"Optimized solver time: {np.mean(t_opts):.4f}s mean, {np.min(t_opts):.4f}s min, {np.max(t_opts):.4f}s max")
    print(f"Speedup: {np.mean(speedups):.1f}x mean, {np.min(speedups):.1f}x min, {np.max(speedups):.1f}x max")
    print(f"\nMax relative errors across ALL {len(results)} runs:")
    print(f"  Elastic tensor: {np.max(tensor_errs):.2e}")
    print(f"  Young's modulus: {np.max(y_errs):.2e}")
    print(f"  Poisson's ratio: {np.max(nu_errs):.2e}")

    # Per-eta summary
    print(f"\n{'eta':>6s} | {'mean_speedup':>12s} | {'mean_t_orig':>11s} | {'mean_t_opt':>10s} | {'max_tensor_err':>14s}")
    print("-" * 65)
    for eta in eta_values:
        eta_results = [r for r in results if abs(r['eta'] - eta) < 1e-6]
        mean_sp = np.mean([r['speedup'] for r in eta_results])
        mean_to = np.mean([r['t_orig'] for r in eta_results])
        mean_tn = np.mean([r['t_opt'] for r in eta_results])
        max_te = np.max([r['tensor_err'] for r in eta_results])
        print(f"{eta:6.3f} | {mean_sp:12.1f}x | {mean_to:10.4f}s | {mean_tn:10.4f}s | {max_te:14.2e}")

    all_passed = all(r['tensor_err'] < 1e-8 for r in results)
    print(f"\nAll {len(results)} tests passed (rel error < 1e-8): {'YES' if all_passed else 'NO'}")
    if not all_passed:
        failures = [r for r in results if r['tensor_err'] >= 1e-8]
        print(f"  Failures: {len(failures)}")
        for f in failures:
            print(f"    eta={f['eta']:.3f}, trial={f['trial']}, tensor_err={f['tensor_err']:.2e}")


if __name__ == '__main__':
    run_benchmark()
