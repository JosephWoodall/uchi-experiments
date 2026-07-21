"""Scaling-law fitting: the reusable answer to "simulate large scale
cheaply" (Kaplan et al. 2020, arXiv:2001.08361; Hoffmann et al./Chinchilla
2022, arXiv:2203.15556) -- train several small sizes, fit L(N) as a power
law, extrapolate to sizes not actually trained. This is what
tasks/todo.md's original Phase A-D plan set out to build before Ducky's
own architecture questions took over; it was never finished, and existing
runs/*_base_{xs,s,m,l} checkpoints turned out to be an inconsistent basis
for it (different vocab generations, step budgets -- checked directly, not
assumed) -- this is the first clean version.

Fits L(N) = a * N^(-alpha) via a log-log linear regression
(numpy.polyfit, degree 1) -- the same simple form Kaplan et al. use, no
scipy.optimize dependency needed. With only 2-3 size points this is a
secant slope, not a robust regression -- an honest limitation, not hidden:
call get_fit_quality() to see how many points went in before trusting an
extrapolation.
"""
import math


def fit_power_law(points: list[tuple[float, float]]) -> dict:
    """points: [(N, L), ...], N=params (or any capacity proxy), L=loss.
    Returns {"a": a, "alpha": alpha, "n_points": k} for L(N) = a * N**(-alpha).
    Requires at least 2 points (a line needs 2 points; this is documented
    as a real limitation of a 2-point fit, not glossed over).
    """
    import numpy as np

    if len(points) < 2:
        raise ValueError("fit_power_law needs at least 2 (N, L) points")
    log_n = np.array([math.log(n) for n, _ in points])
    log_l = np.array([math.log(l) for _, l in points])
    slope, intercept = np.polyfit(log_n, log_l, deg=1)
    alpha = -slope
    a = math.exp(intercept)
    return {"a": a, "alpha": alpha, "n_points": len(points)}


def extrapolate(fit: dict, target_n: float) -> float:
    """L(target_n) = a * target_n**(-alpha), using a fit from fit_power_law."""
    return fit["a"] * target_n ** (-fit["alpha"])


def compare_architectures(points_by_arch: dict, target_n: float) -> dict:
    """points_by_arch: {arch_name: [(N, L), ...]}. Fits each architecture's
    curve and reports the extrapolated loss at target_n for each, plus
    which one wins (lower loss). Does NOT average across domains -- call
    this once per domain and compare the verdicts, per the lesson
    tie_layers already taught this session (a single-domain toy result
    doesn't generalize; averaging across domains would hide exactly the
    kind of domain-dependence that mattered there).
    """
    results = {}
    for arch, points in points_by_arch.items():
        fit = fit_power_law(points)
        results[arch] = {**fit, "extrapolated_loss": extrapolate(fit, target_n)}
    winner = min(results, key=lambda a: results[a]["extrapolated_loss"])
    return {"target_n": target_n, "by_architecture": results, "winner": winner}


if __name__ == "__main__":
    # Sanity check against a synthetic KNOWN power law before trusting this
    # on real data -- does the fit actually recover a=10, alpha=0.3?
    true_a, true_alpha = 10.0, 0.3
    synthetic = [(n, true_a * n ** (-true_alpha)) for n in [1e4, 1e5, 1e6, 1e7]]
    fit = fit_power_law(synthetic)
    print(f"true: a={true_a}, alpha={true_alpha}")
    print(f"fit:  a={fit['a']:.4f}, alpha={fit['alpha']:.4f}")
    assert abs(fit["a"] - true_a) < 1e-6 and abs(fit["alpha"] - true_alpha) < 1e-6, (
        "fit_power_law failed to recover a known synthetic power law -- do not trust it on real data"
    )
    print("sanity check passed: fit exactly recovers a known synthetic power law")
