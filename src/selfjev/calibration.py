"""Held-out temperature scaling (Guo et al. 2017) and validation-selected thresholds.

Fit temperatures only on predictions whose split is 'calibration'; select thresholds only on 'validation'.
Anything else (in particular 'test') is refused. A calibration file is tied to the model revision, adapter
and prompt that produced the predictions, and refuses to load for a different scorer.
"""
import json
import math
import time
from pathlib import Path

from .classify import sigmoid
from .data import sha256_file

MIN_N = 30  # questions (or labels, for multilabel) needed to fit a temperature for one output type
GRID = [10 ** (k / 200) for k in range(-400, 401)]  # T in [0.01, 100], ~1.2% steps


class LeakageError(ValueError):
    pass


def _softplus(x):
    return max(x, 0) + math.log1p(math.exp(-abs(x)))


def nll(items, kind, T):
    """Mean negative log-likelihood of temperature-scaled scores."""
    if kind == "multiclass":  # items: (scores, target index)
        tot = 0.0
        for s, g in items:
            z = [v / T for v in s]
            m = max(z)
            tot += m + math.log(sum(math.exp(v - m) for v in z)) - z[g]
        return tot / len(items)
    return sum(_softplus(-s / T) if y else _softplus(s / T) for s, y in items) / len(items)  # items: (score, bool)


def _items(preds, kind):
    rows = [r for r in preds if r["type"] == kind]
    if kind == "binary":
        return [(r["scores"][0], r["target"]) for r in rows]
    if kind == "multiclass":
        return [(r["scores"], r["candidate_ids"].index(r["target"])) for r in rows]
    return [(s, c in r["target"]) for r in rows for c, s in zip(r["candidate_ids"], r["scores"])]


def _load_report(path, split):
    report = json.loads(Path(path).read_text())
    bad = {r["split"] for r in report["predictions"]} - {split}
    if bad or report["meta"]["splits"] != [split]:
        raise LeakageError(f"{path}: expected only split '{split}' predictions, found {sorted(bad | set(report['meta']['splits']))}. "
                           "Temperatures fit on 'calibration', thresholds on 'validation'; test labels are never used.")
    if report["meta"].get("calibration"):
        raise ValueError(f"{path}: predictions were already calibrated; re-run eval without --calibration")
    return report


def best_threshold(items, T):
    """F1-maximising threshold on sigmoid(s / T): one sweep over distinct probabilities, cutting midway."""
    pts = sorted(((sigmoid(s / T), bool(y)) for s, y in items), reverse=True)
    pos, tp, fp, best = sum(y for _, y in pts), 0, 0, (-1.0, 0.5)
    for i, (p, y) in enumerate(pts):
        tp, fp = tp + y, fp + (not y)
        nxt = pts[i + 1][0] if i + 1 < len(pts) else 0.0
        if nxt == p:
            continue
        f1 = 2 * tp / (tp + fp + pos) if pos else 0.0
        if f1 > best[0]:
            best = (f1, (p + nxt) / 2)
    return best[1]


def fit(fit_report, threshold_report=None, out=None):
    rep = _load_report(fit_report, "calibration")
    meta = rep["meta"]
    cal = {"temperature": {}, "threshold": {}, "fit": {
        "method": "temperature scaling per output type, grid search on NLL", "min_n": MIN_N,
        "model": meta["model"], "revision": meta["revision"], "adapter": meta["adapter"],
        "adapter_sha256": meta["adapter_sha256"], "prompt_sha": meta["prompt_sha"],
        "fit_report": str(fit_report), "fit_report_sha256": sha256_file(fit_report), "fit_data": meta["data"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n": {}, "nll_before": {}, "nll_after": {}, "skipped": {}}}
    for kind in ("binary", "multiclass", "multilabel"):
        items = _items(rep["predictions"], kind)
        cal["fit"]["n"][kind] = len(items)
        if len(items) < MIN_N:
            cal["fit"]["skipped"][kind] = f"only {len(items)} items (< {MIN_N}); left uncalibrated"
            continue
        T = min(GRID, key=lambda t: nll(items, kind, t))
        cal["temperature"][kind] = T
        cal["fit"]["nll_before"][kind], cal["fit"]["nll_after"][kind] = nll(items, kind, 1.0), nll(items, kind, T)
    if threshold_report:
        vrep = _load_report(threshold_report, "validation")
        if any(vrep["meta"][k] != meta[k] for k in ("revision", "adapter_sha256", "prompt_sha")):
            raise ValueError("threshold report comes from a different model/adapter/prompt than the fit report")
        cal["fit"] |= {"threshold_report": str(threshold_report), "threshold_report_sha256": sha256_file(threshold_report),
                       "threshold_rule": "max F1 on validation after temperature scaling"}
        for kind in ("binary", "multilabel"):
            items = _items(vrep["predictions"], kind)
            if len(items) >= MIN_N and any(y for _, y in items):
                cal["threshold"][kind] = best_threshold(items, cal["temperature"].get(kind, 1.0))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(cal, indent=1))
    return cal


def load(path, scorer_meta: dict, prompt_sha: str) -> dict:
    cal = json.loads(Path(path).read_text())
    f = cal["fit"]
    want = (scorer_meta["model"], scorer_meta["revision"], scorer_meta["adapter_sha256"], prompt_sha)
    got = (f["model"], f["revision"], f["adapter_sha256"], f["prompt_sha"])
    if got != want:
        raise ValueError(f"calibration {path} was fit for (model, revision, adapter sha, prompt sha) {got}, not {want}")
    cal["fit"] = f | {"file": str(path), "file_sha256": sha256_file(path)}
    return cal
