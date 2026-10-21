"""Profile loading, seeded perturbation and the samplers built from a profile.

A profile holds aggregate statistics only (see profiles/). Every distribution
the generator draws from is read from it, so swapping a placeholder for a real
profile is a one-path change in config.json.
"""
import copy
import hashlib
import json
import re

import numpy as np

QUANTILE_KEY = re.compile(r"^p\d+(\.\d+)?$")
NOT_PERTURBED = {"window_days", "cross_shard_lookback", "n_clusters", "ks", "alpha",
                 "xmin", "mu", "sigma", "fit", "sample", "ratios"}
UNIT_INTERVAL = re.compile(r"(rate|frac|mass|reveal|^p_)")


def load(path):
    with open(path) as f:
        return json.load(f)


def overlay(base, over):
    """`over` (e.g. a real profile) on top of `base` (the synthetic placeholder).

    Dicts merge recursively, except statistical blocks (quantile sets, degree
    blocks, category blocks), which `over` replaces whole so stats never mix."""
    if not (isinstance(base, dict) and isinstance(over, dict)):
        return copy.deepcopy(over)
    if any(QUANTILE_KEY.match(k) for k in over) or ({"quantiles", "frac_zero"} <= set(over) and "dtype" not in over):
        return copy.deepcopy(over)  # a quantile set or a degree block
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = overlay(base[k], v) if k in base else copy.deepcopy(v)
    return out


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _factor(seed, path, lo, hi):
    h = hashlib.blake2b(f"{seed}:{path}".encode(), digest_size=16).digest()
    r = np.random.Generator(np.random.Philox(key=int.from_bytes(h, "little")))
    return 1.0 + r.choice([-1.0, 1.0]) * r.uniform(lo, hi)


def perturb(profile, seed, lo, hi):
    """Scales every numeric leaf by a seeded factor in [1-hi, 1-lo] or [1+lo, 1+hi].

    Sibling quantile anchors (p10, p50, ..., max) and their mean share one factor,
    so they stay monotone and consistent; the elements of a list (e.g. hour-of-day
    weights) share one factor; rates are clamped to [0, 1]; integers stay integers >= 1 when the
    original was >= 1. The same seed always gives the same profile.
    """
    out = copy.deepcopy(profile)

    def walk(node, path):
        items = node.items() if isinstance(node, dict) else enumerate(node)
        for k, v in list(items):
            key = str(k)
            if key in NOT_PERTURBED:
                continue
            p = f"{path}/{key}"
            if isinstance(v, (dict, list)):
                walk(v, p)
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if QUANTILE_KEY.match(key) or key == "max" or isinstance(node, list):
                group = path  # one factor for all anchors of a quantile set / all weights of a list
            elif key == "mean":  # a mean moves with its quantiles (beside or inside them)
                group = path if path.endswith("/quantiles") else f"{path}/quantiles"
            else:
                group = p
            x = v * _factor(seed, group, lo, hi)
            if UNIT_INTERVAL.search(key):
                if v > 0.5:  # perturb the complement, so e.g. 0.92 never becomes 1.0
                    x = 1.0 - (1.0 - v) * _factor(seed, group, lo, hi)
                x = min(max(x, 0.0), 1.0)
            if key.endswith("rho"):
                x = min(max(x, -0.95), 0.95)
            if isinstance(v, int):
                x = max(int(round(x)), 1 if v >= 1 else 0)
            node[k] = x

    walk(out, "")
    return out


class Quantile:
    """Inverse CDF from quantile anchors, linear in u (optionally in log1p space)."""

    def __init__(self, probs, values, log=False):
        order = np.argsort(probs, kind="stable")
        self.p = np.asarray(probs, float)[order]
        v = np.maximum.accumulate(np.asarray(values, float)[order])
        self.log = log
        self.v = np.log1p(v) if log else v

    def __call__(self, u):
        x = np.interp(u, self.p, self.v)
        return np.expm1(x) if self.log else x

    def mean(self):
        return float(self(np.linspace(0, 1, 200_001)).mean())


def _anchors(quantiles):
    out = []
    for k, v in quantiles.items():
        if QUANTILE_KEY.match(k):
            out.append((float(k[1:]) / 100.0, float(v)))
    return sorted(out)


def numeric_sampler(spec):
    """Numeric column: p1..p99 anchors, tails extrapolated a little past p1/p99."""
    a = _anchors(spec["quantiles"])
    p = [x for x, _ in a]
    v = [y for _, y in a]
    if p[0] > 0:
        span = v[min(2, len(v) - 1)] - v[0]
        lo = v[0] - 0.25 * span
        p, v = [0.0] + p, [max(lo, 0.0) if v[0] >= 0 else lo] + v
    if p[-1] < 1:
        span = v[-1] - v[max(len(v) - 3, 0)]
        p, v = p + [1.0], v + [v[-1] + span]
    return Quantile(p, v)


def length_sampler(spec):
    a = _anchors(spec)
    p = [x for x, _ in a]
    v = [y for _, y in a]
    if p[0] > 0:
        p, v = [0.0] + p, [v[0]] + v
    if p[-1] < 1:
        p, v = p + [1.0], v + [v[-1]]
    return Quantile(p, v)


def degree_mean(deg):
    """Profiles put the mean beside the quantiles or inside them."""
    m = deg.get("mean", deg.get("quantiles", {}).get("mean"))
    return None if m is None else float(m)


def tail_alpha(deg):
    """Power-law tail exponent: the profile's fit, else from the p99.9 -> p99.99 slope."""
    fit = deg.get("fit", {}).get("powerlaw_tail", {})
    if fit.get("alpha"):
        return float(fit["alpha"])
    q = deg["quantiles"]
    a, b = float(q.get("p99.9", 0)), float(q.get("p99.99", 0))
    return 1.0 + np.log(10) / np.log(b / a) if a > 0 and b > a else 3.0


def degree_sampler(deg, real_parent_rows=None, shard_parent_rows=None, nodes=None):
    """Per-node degree inverse CDF.

    u < frac_zero maps to 0; the remaining anchors are interpolated in log1p space.
    The top anchor (u = 1) is chosen per shard and never exceeds the global max:
    - hubs (real_parent_rows given) keep their share of the parent table:
      max(p99.99, global_max * shard_rows / real_rows);
    - per-node degrees of the main entity (nodes = entities per shard) extend the
      power-law tail from p99.99 to the expected max of `nodes` draws:
      p99.99 * (1e-4 * nodes) ** (1 / (alpha - 1)).
    """
    q = deg["quantiles"]
    fz = float(deg.get("frac_zero", 0.0))
    a = [(p, max(v, 1.0)) for p, v in _anchors(q) if p > fz]  # above frac_zero, degrees are >= 1
    top = max(v for _, v in _anchors(q))
    if "max" in q:
        gmax = float(q["max"])
        if real_parent_rows:
            ext = gmax * shard_parent_rows / real_parent_rows
        elif nodes:
            ext = float(q.get("p99.99", top)) * max(1e-4 * nodes, 1.0) ** (1.0 / max(tail_alpha(deg) - 1.0, 0.5))
        else:
            ext = gmax
        top = max(top, min(gmax, ext))
    probs = [fz] + [p for p, _ in a] + [1.0]
    vals = [1.0] + [v for _, v in a] + [top]
    if fz > 0:
        probs, vals = [0.0, fz - 1e-12] + probs, [0.0, 0.0] + vals
    q = Quantile(probs, vals, log=True)
    m = degree_mean(deg)
    return TailCalibrated(q, m) if m else q


class TailCalibrated:
    """Keeps the quantile function exact up to p90 and stretches the tail above it
    so the distribution mean equals the profile mean (keeps table row ratios).
    The stretch factor is clamped to [0.25, 1.25]."""

    def __init__(self, q, mean):
        self.q = q
        grid = q(np.linspace(0, 1, 200_001))
        self.q90 = float(q(np.array([0.9]))[0])
        excess = np.maximum(grid - self.q90, 0).mean()
        # stretch capped at 1.25: a mean driven by outliers beyond one shard's max must not
        # inflate p90..p99
        self.c = 1.0 if excess <= 0 else float(np.clip(1 + (mean - grid.mean()) / excess, 0.25, 1.25))

    def __call__(self, u):
        x = self.q(u)
        return np.where(x > self.q90, self.q90 + (x - self.q90) * self.c, x)

    def mean(self):
        return float(self(np.linspace(0, 1, 200_001)).mean())


def zipf_rank(u, n, s):
    """Approximate Zipf(s) rank in [0, n) via the continuous power-law inverse CDF."""
    n = int(n)
    if n <= 1:
        return np.zeros(len(u), np.int64)
    if abs(s - 1.0) < 1e-6:
        x = np.exp(u * np.log(n + 1.0))
    else:
        x = (((n + 1.0) ** (1.0 - s) - 1.0) * u + 1.0) ** (1.0 / (1.0 - s))
    return np.minimum(x.astype(np.int64) - 1, n - 1).clip(0)


def stochastic_round(x, u):
    return np.floor(x + u).astype(np.int64)


class TenureScaled:
    """Per-day activity sampler: an entity's count is d(u) * f, where f is its tenure
    relative to the mean tenure, so events per day do not depend on when it joined.

    d is solved on a grid (multiplicative deconvolution against the tenure-factor
    sample) so that the counts d(u) * f still follow the profiled degree quantiles.
    """

    def __init__(self, q, factors, iters=30):
        # dense in the upper tail, where a few nodes carry much of the mass
        self.grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1001), 1 - np.logspace(-7, -2, 300)]))
        target = q(self.grid)
        f = np.asarray(factors, float)
        d = target.copy()
        for _ in range(iters):
            prod = np.sort((d[:, None] * f[None, :]).ravel())
            # quantiles of the product of d(U) and f, on the grid (each grid cell weighs its width)
            w = np.diff(np.concatenate([[0.0], (self.grid[1:] + self.grid[:-1]) / 2, [1.0]]))
            cdf = np.cumsum(np.repeat(w / len(f), len(f))[np.argsort((d[:, None] * f[None, :]).ravel(), kind="stable")])
            achieved = prod[np.minimum(np.searchsorted(cdf, self.grid), len(prod) - 1)]
            ratio = np.where(achieved > 0, target / np.maximum(achieved, 1e-12), 2.0)
            d = np.sort(np.where(target > 0, d * np.clip(ratio, 0.5, 2.0) ** 0.7, 0.0))
        self.d = d

    def __call__(self, u):
        return np.expm1(np.interp(u, self.grid, np.log1p(self.d)))

    def mean(self):
        """Mean count (the tenure factor has mean 1)."""
        return float(self(np.linspace(0, 1, 200_001)).mean())
