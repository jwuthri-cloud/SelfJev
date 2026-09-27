"""Quality and calibration metrics over JSONL examples. Writes report.json (with every prediction) + report.md.

Calibration error (ECE) is computed with 10 equal-width bins over [0, 1]:
  ECE = sum_b (n_b / N) * |mean(outcome in b) - mean(confidence in b)|
  binary / multilabel: confidence = p_yes of each (question or label), outcome = target is yes (marginal reliability)
  multiclass:          confidence = max probability, outcome = argmax is correct (top-label reliability)
Brier: binary/multilabel mean (p - y)^2 per question/label; multiclass mean over questions of sum_k (p_k - y_k)^2.
"""
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers

from .classify import classify_many, run_meta
from .formatting import DEFAULT_PROMPT
from .data import load, sha256_file

BINS = 10


def _div(a, b):
    return a / b if b else None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def prf(tp, fp, fn):
    p, r = _div(tp, tp + fp) or 0.0, _div(tp, tp + fn) or 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def auroc(ys, ps):
    """Rank-based AUROC with average ranks for ties; None if only one class is present."""
    pairs = sorted(zip(ps, ys))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    npos = sum(1 for _, y in pairs if y)
    nneg = len(pairs) - npos
    if not npos or not nneg:
        return None
    return (sum(r for r, (_, y) in zip(ranks, pairs) if y) - npos * (npos + 1) / 2) / (npos * nneg)


def reliability(conf, outcome):
    bins = defaultdict(list)
    for c, o in zip(conf, outcome):
        bins[min(int(c * BINS), BINS - 1)].append((c, o))
    rows = [{"bin": f"[{b / BINS:.1f},{(b + 1) / BINS:.1f}{']' if b == BINS - 1 else ')'}", "n": len(v),
             "confidence": _mean([c for c, _ in v]), "frequency": _mean([float(o) for _, o in v])} for b, v in sorted(bins.items())]
    ece = sum(r["n"] * abs(r["frequency"] - r["confidence"]) for r in rows) / len(conf) if conf else None
    return ece, rows


def _logloss(p, y):
    p = min(max(p, 1e-15), 1 - 1e-15)
    return -math.log(p if y else 1 - p)


def binary_metrics(ys, ps, preds):
    tp = sum(1 for y, d in zip(ys, preds) if y and d)
    fp = sum(1 for y, d in zip(ys, preds) if not y and d)
    fn = sum(1 for y, d in zip(ys, preds) if y and not d)
    p, r, f1 = prf(tp, fp, fn)
    ece, rel = reliability(ps, ys)
    return {"n": len(ys), "positives": sum(ys), "accuracy": _mean([float(y == d) for y, d in zip(ys, preds)]),
            "precision": p, "recall": r, "f1": f1, "auroc": auroc(ys, ps),
            "brier": _mean([(q - y) ** 2 for q, y in zip(ps, ys)]), "log_loss": _mean([_logloss(q, y) for q, y in zip(ps, ys)]),
            "ece": ece, "reliability": rel}


def macro_f1(pairs):
    """pairs: (gold label id, predicted label id); macro over ids seen in gold or predictions."""
    ids = {g for g, _ in pairs} | {p for _, p in pairs if p is not None}
    return _mean([prf(sum(1 for g, p in pairs if g == i and p == i), sum(1 for g, p in pairs if g != i and p == i),
                      sum(1 for g, p in pairs if g == i and p != i))[2] for i in ids])


def evaluate_predictions(preds):
    """preds: rows produced by predict(). Returns the metrics part of the report."""
    by_type = defaultdict(list)
    for r in preds:
        by_type[r["type"]].append(r)
    out = {}
    if b := by_type["binary"]:
        out["binary"] = binary_metrics([r["target"] for r in b], [r["p_yes"] for r in b], [r["selected"] for r in b])
    if m := by_type["multiclass"]:
        conf = [max(r["probabilities"]) for r in m]
        ok = [r["selected"] == r["target"] for r in m]
        ece, rel = reliability(conf, ok)
        answered = [(c, o) for c, o, r in zip(conf, ok, m) if r["selected"] is not None]
        out["multiclass"] = {
            "n": len(m), "accuracy": _mean([float(o) for o in ok]), "macro_f1": macro_f1([(r["target"], r["selected"]) for r in m]),
            "log_loss": _mean([-math.log(max(r["probabilities"][r["candidate_ids"].index(r["target"])], 1e-15)) for r in m]),
            "brier": _mean([sum((p - (c == r["target"])) ** 2 for c, p in zip(r["candidate_ids"], r["probabilities"])) for r in m]),
            "ece_top_label": ece, "reliability": rel,
            "abstention": [{"abstain_below": t, "coverage": len(acc) / len(m), "error_on_accepted": _mean([1.0 - o for _, o in acc])}
                           for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for acc in [[a for a in answered if a[0] >= t]]]}
    if ml := by_type["multilabel"]:
        labels = [(c, c in r["target"], p, c in r["selected"]) for r in ml for c, p in zip(r["candidate_ids"], r["probabilities"])]
        tp = sum(1 for _, y, _, d in labels if y and d)
        fp = sum(1 for _, y, _, d in labels if not y and d)
        fn = sum(1 for _, y, _, d in labels if y and not d)
        mp, mr, mf = prf(tp, fp, fn)
        per_label = defaultdict(lambda: [0, 0, 0, 0])  # tp fp fn n_pos
        for c, y, _, d in labels:
            per_label[c][0] += y and d
            per_label[c][1] += (not y) and d
            per_label[c][2] += y and not d
            per_label[c][3] += y
        ece, rel = reliability([p for _, _, p, _ in labels], [y for _, y, _, _ in labels])
        out["multilabel"] = {
            "n": len(ml), "labels": len(labels), "exact_match": _mean([float(set(r["selected"]) == set(r["target"])) for r in ml]),
            "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
            "macro_f1": _mean([prf(*v[:3])[2] for v in per_label.values() if v[3] or v[1]]),
            "label_auroc": auroc([y for _, y, _, _ in labels], [p for _, _, p, _ in labels]),
            "brier": _mean([(p - y) ** 2 for _, y, p, _ in labels]), "log_loss": _mean([_logloss(p, y) for _, y, p, _ in labels]),
            "ece": ece, "reliability": rel,
            "per_label": {c: dict(zip(("precision", "recall", "f1"), prf(*v[:3]))) | {"positives": v[3]} for c, v in sorted(per_label.items())}}
    out["question_accuracy"] = _mean([float(r["correct"]) for r in preds])
    return out


def _slices(preds, key):
    groups = defaultdict(list)
    for r in preds:
        for k in key(r):
            groups[k].append(r)
    return {k: {"n": len(v), "question_accuracy": _mean([float(r["correct"]) for r in v]),
                **{t: sum(1 for r in v if r["type"] == t) for t in ("binary", "multiclass", "multilabel")}}
            for k, v in sorted(groups.items())}


def _length_bucket(n):
    lo = max([0, 128, 512, 2048, 8192], key=lambda b: (n >= b, b))
    hi = {0: 127, 128: 511, 512: 2047, 2048: 8191}.get(lo)
    return f"{lo:05d}-{hi:05d}" if hi else f"{lo:05d}+"


def breakdowns(preds):
    fams = defaultdict(list)
    for r in preds:
        fams[r["family"]].append(r)
    para = defaultdict(list)
    for r in preds:
        if r.get("paraphrase_group"):
            para[r["paraphrase_group"]].append(r)
    para = {k: v for k, v in para.items() if len(v) > 1}
    return {
        "by_family": {f: evaluate_predictions(v) for f, v in sorted(fams.items())},
        "by_hard_case": _slices(preds, lambda r: r["hard_cases"] or ["(none)"]),
        "by_state_tokens": _slices(preds, lambda r: [_length_bucket(r["state_tokens"])]),
        "by_candidates": _slices(preds, lambda r: [f"{len(r['candidate_ids']) or 1:02d}"]),
        "paraphrase_groups": {"groups": len(para),
                              "same_prediction": _mean([float(len({json.dumps(r["selected"], sort_keys=True) for r in v}) == 1) for v in para.values()]),
                              "all_correct": _mean([float(all(r["correct"] for r in v)) for v in para.values()])},
    }


def predict(scorer, examples, calibration=None, prompt=DEFAULT_PROMPT):
    requests = [{"state": ex["state"], "questions": [{"id": "q", **ex["question"]}]} for ex in examples]
    results, stats = classify_many(scorer, requests, calibration, prompt)
    state_tokens = [len(x) for x in scorer.tokenizer([ex["state"] for ex in examples], add_special_tokens=False)["input_ids"]]
    preds = []
    for ex, [res], n in zip(examples, results, state_tokens):
        row = {k: ex.get(k) for k in ("id", "source_id", "family", "split", "target", "paraphrase_group")}
        row |= {"type": res["type"], "hard_cases": ex.get("hard_cases", []), "state_tokens": n, "selected": res["selected"],
                "calibration": res["calibration"]}
        if res["type"] == "binary":
            row |= {"scores": [res["score"]], "p_yes": res["p_yes"], "candidate_ids": [], "probabilities": [res["p_yes"]]}
        else:
            row |= {"candidate_ids": [c["id"] for c in res["candidates"]], "scores": [c["score"] for c in res["candidates"]],
                    "probabilities": [c["probability"] for c in res["candidates"]]}
        row["correct"] = (set(row["selected"]) == set(row["target"])) if res["type"] == "multilabel" else row["selected"] == row["target"]
        preds.append(row)
    return preds, stats


def run(scorer, data_paths, splits=None, calibration=None, out_dir=None, limit=None, prompt=DEFAULT_PROMPT):
    examples = load(data_paths, set(splits) if splits else None)[:limit]
    if not examples:
        raise ValueError(f"no examples in {data_paths} for splits {splits}")
    t0 = time.perf_counter()
    preds, stats = predict(scorer, examples, calibration, prompt)
    report = {
        "meta": run_meta(scorer, calibration, prompt) | {
            "data": [{"path": str(p), "sha256": sha256_file(p)} for p in data_paths], "splits": sorted({p["split"] or "" for p in preds}),
            "n": len(preds), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": time.perf_counter() - t0, "scoring": stats,
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
            "machine": platform.machine(), "platform": platform.platform()},
        "metrics": evaluate_predictions(preds), **breakdowns(preds), "predictions": preds}
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "report.json").write_text(json.dumps(report, indent=1))
        (Path(out_dir) / "report.md").write_text(markdown(report))
        (Path(out_dir) / "reliability.svg").write_text(reliability_svg(report["metrics"]))
    return report


def rerender(report_path):
    """Rewrite report.md / reliability.svg from a saved report.json (no model; report.json is left untouched)."""
    path = Path(report_path)
    report = json.loads(path.read_text())
    report |= {"metrics": evaluate_predictions(report["predictions"]), **breakdowns(report["predictions"])}
    (path.parent / "report.md").write_text(markdown(report))
    (path.parent / "reliability.svg").write_text(reliability_svg(report["metrics"]))


def reliability_svg(m, size=180, pad=28):
    """Reliability diagram per output type: bin mean confidence (x) vs observed frequency (y); area ~ bin count."""
    panels = [(t, m[t]["reliability"]) for t in ("binary", "multiclass", "multilabel") if t in m]
    w = size + 2 * pad
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w * len(panels)}" height="{w}" font-family="sans-serif" font-size="11">',
           f'<rect width="100%" height="100%" fill="white"/>']
    for k, (t, rows) in enumerate(panels):
        x0, n = k * w + pad, sum(r["n"] for r in rows)
        out += [f'<rect x="{x0}" y="{pad}" width="{size}" height="{size}" fill="none" stroke="#888"/>',
                f'<line x1="{x0}" y1="{pad + size}" x2="{x0 + size}" y2="{pad}" stroke="#bbb" stroke-dasharray="4"/>',
                f'<text x="{x0}" y="{pad - 8}">{t} ({"top-label" if t == "multiclass" else "p_yes"}), n={n}</text>',
                f'<text x="{x0}" y="{pad + size + 16}">confidence → observed ↑</text>']
        out += [f'<circle cx="{x0 + r["confidence"] * size:.1f}" cy="{pad + (1 - r["frequency"]) * size:.1f}" '
                f'r="{2 + 9 * (r["n"] / n) ** 0.5:.1f}" fill="#2f6bd6" fill-opacity="0.65"/>' for r in rows]
    return "\n".join(out + ["</svg>"]) + "\n"


def _f(x, pct=False):
    return "—" if x is None else f"{100 * x:.1f}" if pct else f"{x:.3f}" if isinstance(x, float) else str(x)


def headline(m):
    """One-line per-type summary used in tables."""
    parts = []
    if b := m.get("binary"):
        parts.append(f"bin acc {_f(b['accuracy'], 1)} F1 {_f(b['f1'], 1)} AUROC {_f(b['auroc'])} ECE {_f(b['ece'])}")
    if c := m.get("multiclass"):
        parts.append(f"mc acc {_f(c['accuracy'], 1)} mF1 {_f(c['macro_f1'], 1)} ECE {_f(c['ece_top_label'])}")
    if ml := m.get("multilabel"):
        parts.append(f"ml EM {_f(ml['exact_match'], 1)} µF1 {_f(ml['micro_f1'], 1)} ECE {_f(ml['ece'])}")
    return "; ".join(parts)


def markdown(r):
    m, meta = r["metrics"], r["meta"]
    lines = [f"# Evaluation report", "",
             f"- model `{meta['model']}` @ `{meta['revision'][:10]}` ({meta.get('architecture', 'stock reranker pairs')}), "
             f"adapter/checkpoint `{meta['adapter']}`, prompt `{meta['prompt']}` ({meta['prompt_sha']})",
             f"- data {', '.join(d['path'] for d in meta['data'])}; splits {meta['splits']}; n={meta['n']}; calibration `{meta['calibration']}`",
             f"- {meta['device']} / {meta['dtype']}; {meta['created']}; wall {meta['wall_s']:.1f}s", "",
             "## Overall", "", f"question accuracy {_f(m['question_accuracy'], 1)}%", "", "![reliability](reliability.svg)", ""]
    for t, keys in (("binary", ("n", "positives", "accuracy", "precision", "recall", "f1", "auroc", "brier", "log_loss", "ece")),
                    ("multiclass", ("n", "accuracy", "macro_f1", "log_loss", "brier", "ece_top_label")),
                    ("multilabel", ("n", "labels", "exact_match", "micro_f1", "macro_f1", "label_auroc", "brier", "log_loss", "ece"))):
        if t in m:
            lines += [f"**{t}**: " + ", ".join(f"{k} {_f(m[t][k])}" for k in keys), ""]
    lines += ["## By family", "", "| family | n | question acc % | details |", "|---|---|---|---|"]
    lines += [f"| {f} | {sum(v.get(t, {}).get('n', 0) for t in ('binary', 'multiclass', 'multilabel'))} | {_f(v['question_accuracy'], 1)} | {headline(v)} |"
              for f, v in r["by_family"].items()]
    for name in ("by_hard_case", "by_state_tokens", "by_candidates"):
        lines += ["", f"## {name.replace('_', ' ')}", "", "| slice | n | question acc % |", "|---|---|---|"]
        lines += [f"| {k} | {v['n']} | {_f(v['question_accuracy'], 1)} |" for k, v in r[name].items()]
    pg = r["paraphrase_groups"]
    lines += ["", f"## Paraphrase groups", "", f"{pg['groups']} groups; same prediction {_f(pg['same_prediction'], 1)}%; all correct {_f(pg['all_correct'], 1)}%"]
    if "multiclass" in m:
        lines += ["", "## Multiclass abstention (coverage vs error on accepted)", "", "| abstain_below | coverage % | error % |", "|---|---|---|"]
        lines += [f"| {a['abstain_below']} | {_f(a['coverage'], 1)} | {_f(a['error_on_accepted'], 1)} |" for a in m["multiclass"]["abstention"]]
    for t, key in (("binary", "reliability"), ("multiclass", "reliability"), ("multilabel", "reliability")):
        if t in m:
            lines += ["", f"## Reliability: {t}", "", "| bin | n | mean confidence | observed frequency |", "|---|---|---|---|"]
            lines += [f"| {b['bin']} | {b['n']} | {_f(b['confidence'])} | {_f(b['frequency'])} |" for b in m[t][key]]
    return "\n".join(lines) + "\n"


COMPARE = [("question acc %", ("question_accuracy",), True), ("binary acc %", ("binary", "accuracy"), True),
           ("binary F1 %", ("binary", "f1"), True), ("binary AUROC", ("binary", "auroc"), False), ("binary ECE", ("binary", "ece"), False),
           ("multiclass acc %", ("multiclass", "accuracy"), True), ("multiclass macro-F1 %", ("multiclass", "macro_f1"), True),
           ("multiclass ECE", ("multiclass", "ece_top_label"), False), ("multilabel exact %", ("multilabel", "exact_match"), True),
           ("multilabel micro-F1 %", ("multilabel", "micro_f1"), True), ("multilabel label AUROC", ("multilabel", "label_auroc"), False),
           ("multilabel ECE", ("multilabel", "ece"), False)]
CALIBRATION = [(f"{t} {k}", (t, k), False) for t in ("binary", "multiclass", "multilabel") for k in ("brier", "log_loss")]


def _get(m, path):
    for k in path:
        m = m.get(k) if isinstance(m, dict) else None
    return m


def compare(reports, names):
    """Markdown table of headline metrics (overall and per family) for reports on the same data."""
    same = len({json.dumps([d["sha256"] for d in r["meta"]["data"]]) + str(r["meta"]["splits"]) for r in reports}) == 1
    lines = [f"# Comparison: {' vs '.join(names)}", "",
             *([] if same else ["> WARNING: these reports use different data files or splits; rows are not comparable.", ""]),
             "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
             "| adapter | " + " | ".join(str(r["meta"]["adapter"]) for r in reports) + " |",
             "| calibration | " + " | ".join("yes" if r["meta"]["calibration"] else "no" for r in reports) + " |",
             "| prompt | " + " | ".join(f"{r['meta']['prompt']} ({r['meta']['prompt_sha']})" for r in reports) + " |",
             "| n | " + " | ".join(str(r["meta"]["n"]) for r in reports) + " |", "",
             "| slice | metric | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    fams = sorted(set().union(*(r["by_family"] for r in reports)))
    for name, ms in [("overall", [r["metrics"] for r in reports])] + [(f, [r["by_family"].get(f, {}) for r in reports]) for f in fams]:
        for label, path, pct in COMPARE + (CALIBRATION if name == "overall" else []):
            vals = [_get(m, path) for m in ms]
            if any(v is not None for v in vals):
                lines.append(f"| {name} | {label} | " + " | ".join(_f(v, pct) for v in vals) + " |")
    tags = sorted(set().union(*(r["by_hard_case"] for r in reports)))
    lines += ["", "## Question accuracy % by hard-case tag", "", "| tag | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    lines += [f"| {t} | {reports[0]['by_hard_case'].get(t, {}).get('n', '—')} | "
              + " | ".join(_f(r["by_hard_case"].get(t, {}).get("question_accuracy"), True) for r in reports) + " |" for t in tags]
    return "\n".join(lines) + "\n"
