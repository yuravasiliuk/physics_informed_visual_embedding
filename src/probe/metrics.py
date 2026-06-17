"""Ridge probe + GATE-B metrics (gate-b-probe.md §3, §4.3).

The "linear probe" of arm 2a is a closed-form ridge regression per (layer,
target) instead of an SGD-trained Linear head: identical hypothesis class,
deterministic, no training hyperparameters except λ — which is selected on a
group-held-out validation slice of the *train* split, so the test split is
touched exactly once.

Metrics, always against the trivial baselines (§4.3):
    v, w        aggregate R² vs the train-mean predictor.
    com_offset  mean L2 vs the centroid baseline (= predicting the zero
                offset — R7), reported as lift = baseline_L2 − probe_L2,
                plus the "L2 < ½ median bbox diagonal" pass-bar inputs.

Everything is reported per scenario slice as well: a free-fall CoM failure +
contact-rich success is the expected healthy outcome, not a gate failure
(§5 / R6).
"""

import hashlib

import numpy as np

LAMBDA_GRID = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)


def group_split(groups, test_frac=0.2, salt="gate-b"):
    """Deterministic group-level split: hash each rollout group id, send the
    lowest `test_frac` quantile of hashes to test. Hash (not modulo) so the
    assignment is stable under corpus growth — adding clips never flips an
    existing group's side, which keeps cached features comparable across runs."""
    cut = int(test_frac * 2**64)  # fixed threshold, NOT a quantile of the
    # observed keys — a quantile would move when clips are added and flip
    # existing groups across the split.
    is_test = np.array([
        int.from_bytes(hashlib.sha1(f"{salt}:{g}".encode()).digest()[:8], "big") < cut
        for g in groups
    ])
    return ~is_test, is_test


def _ridge(X_tr, y_tr, X_te, lam):
    """Closed-form ridge with train-stat standardization and intercept.
    λ is scaled by n_train so the grid is corpus-size independent."""
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-8
    Xn_tr, Xn_te = (X_tr - mu) / sd, (X_te - mu) / sd
    y_mu = y_tr.mean(axis=0)
    d = Xn_tr.shape[1]
    A = Xn_tr.T @ Xn_tr + lam * len(Xn_tr) / d * np.eye(d)
    W = np.linalg.solve(A, Xn_tr.T @ (y_tr - y_mu))
    return Xn_te @ W + y_mu


def fit_linear_probe(X_tr, y_tr, X_te, groups_tr):
    """Select λ on an inner group split of train, refit on full train, predict
    test. Returns (y_pred_test, best_lambda)."""
    in_tr, in_val = group_split(groups_tr, test_frac=0.2, salt="lambda-val")
    if in_val.sum() == 0 or in_tr.sum() == 0:  # tiny corpora: no inner split
        in_tr = in_val = np.ones(len(groups_tr), dtype=bool)
    best_lam, best_err = LAMBDA_GRID[0], np.inf
    for lam in LAMBDA_GRID:
        pred = _ridge(X_tr[in_tr], y_tr[in_tr], X_tr[in_val], lam)
        err = float(((pred - y_tr[in_val]) ** 2).sum())
        if err < best_err:
            best_lam, best_err = lam, err
    return _ridge(X_tr, y_tr, X_te, best_lam), best_lam


def r2_aggregate(y, y_pred, y_train_mean):
    """Aggregate R² over all dims: 1 − SSE/SST, SST against the *train* mean
    (the trivial-baseline ruler of §4.3, also valid on scenario slices)."""
    sse = float(((y - y_pred) ** 2).sum())
    sst = float(((y - y_train_mean) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def com_metrics(y, y_pred, bbox_diag):
    """CoM-offset block: probe L2, centroid-baseline L2 (zero offset), lift,
    and the ½-median-bbox-diagonal pass bar (§3)."""
    probe_l2 = float(np.linalg.norm(y - y_pred, axis=1).mean())
    baseline_l2 = float(np.linalg.norm(y, axis=1).mean())
    half_diag = float(np.median(bbox_diag) / 2.0)
    return {
        "probe_l2": probe_l2,
        "centroid_baseline_l2": baseline_l2,
        "lift": baseline_l2 - probe_l2,
        "half_median_bbox_diag": half_diag,
        "pass": bool(probe_l2 < half_diag and baseline_l2 - probe_l2 > 0),
    }


def spearman(a, b):
    """Spearman rank correlation (average-rank ties via argsort-of-argsort on
    stable sort; adequate for continuous values)."""
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        return r
    ra, rb = rank(np.asarray(a)), rank(np.asarray(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def pair_divergence_layer(xa, xb, x_base_pool, groups_pool, rng, n_baseline=2000):
    """Layer-wise paired-intervention divergence (the controlled-pair
    experiment companion to the linear probe).

    xa, xb       (n_pairs, D) features of base / intervened twins at one layer.
    x_base_pool  (m, D) all base samples at the same layer; baseline pairs are
                 drawn between *different rollout groups*, giving the
                 "unrelated clips" distance scale at this depth.

    Raw L2 between layers is incomparable — transformer feature norms and
    anisotropy grow with depth — so the headline number is the *ratio* of the
    twin distance to the unrelated-pair distance at the same layer:
        ratio ≈ 0  -> encoder collapses the intervention (blind at this depth);
        ratio ≈ 1  -> twins look as different as unrelated clips (divergence
                      saturated — past the chaotic fork, uninformative);
        in between -> the intervention is resolved but not saturated; the
                      depth profile of this ratio is the deliverable.
    """
    d_pair = np.linalg.norm(xa - xb, axis=1)
    cos = (xa * xb).sum(axis=1) / (
        np.linalg.norm(xa, axis=1) * np.linalg.norm(xb, axis=1) + 1e-12
    )
    m = len(x_base_pool)
    i = rng.integers(0, m, n_baseline)
    j = rng.integers(0, m, n_baseline)
    diff_group = groups_pool[i] != groups_pool[j]
    d_base = np.linalg.norm(x_base_pool[i[diff_group]] - x_base_pool[j[diff_group]], axis=1)
    base_med = float(np.median(d_base)) if len(d_base) else float("nan")
    return {
        "pair_l2_median": float(np.median(d_pair)),
        "baseline_l2_median": base_med,
        "ratio": float(np.median(d_pair) / base_med) if base_med else float("nan"),
        "cosine_median": float(np.median(cos)),
    }, d_pair


def evaluate_layer(X_tr, X_te, targets_tr, targets_te, groups_tr,
                   scenarios_te, bbox_diag_te):
    """Fit+score one layer for all targets; per-scenario slices included.

    targets_* : {"v": (n,3), "w": (n,3), "com_offset": (n,3)}
    Returns a JSON-ready dict.
    """
    out = {}
    for name in ("v", "w", "com_offset"):
        y_tr, y_te = targets_tr[name], targets_te[name]
        y_pred, lam = fit_linear_probe(X_tr, y_tr, X_te, groups_tr)
        y_mu = y_tr.mean(axis=0)
        block = {"lambda": lam, "r2": r2_aggregate(y_te, y_pred, y_mu), "n_test": len(y_te)}
        if name == "com_offset":
            block.update(com_metrics(y_te, y_pred, bbox_diag_te))
        per_scen = {}
        for scen in np.unique(scenarios_te):
            s = scenarios_te == scen
            sb = {"r2": r2_aggregate(y_te[s], y_pred[s], y_mu), "n": int(s.sum())}
            if name == "com_offset":
                sb.update(com_metrics(y_te[s], y_pred[s], bbox_diag_te[s]))
            per_scen[str(scen)] = sb
        block["per_scenario"] = per_scen
        out[name] = block
    return out
