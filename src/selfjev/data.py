"""JSONL examples: one (state, question, target) per line. See README "Data format"."""
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from .schemas import ValidationError, parse_question

SPLITS = ("train", "validation", "calibration", "test", "dev")
REQUIRED = {"id", "source_id", "family", "provenance", "state", "question", "target"}
OPTIONAL = {"split", "hard_cases", "paraphrase_group", "notes", "dataset", "jev"}  # dataset, jev: data/all.jsonl.gz


def validate_example(ex: dict):
    """Returns the parsed Question; raises ValidationError naming the example."""
    where = f"example {ex.get('id', '?')!r}" if isinstance(ex, dict) else "example"
    if not isinstance(ex, dict):
        raise ValidationError(f"{where}: must be an object")
    missing, extra = REQUIRED - set(ex), set(ex) - REQUIRED - OPTIONAL
    if missing or extra:
        raise ValidationError(f"{where}: missing {sorted(missing)} / unknown {sorted(extra)}")
    for k in ("id", "source_id", "family", "provenance", "state"):
        if not isinstance(ex[k], str) or not ex[k].strip():
            raise ValidationError(f"{where}: '{k}' must be a non-empty string")
    if ex.get("split") is not None and ex["split"] not in SPLITS:
        raise ValidationError(f"{where}: split must be one of {SPLITS}")
    hc = ex.get("hard_cases", [])
    if not isinstance(hc, list) or not all(isinstance(h, str) and h for h in hc):
        raise ValidationError(f"{where}: hard_cases must be a list of strings")
    try:
        q = parse_question({"id": "q", **ex["question"]} if isinstance(ex["question"], dict) else ex["question"])
    except ValidationError as e:
        raise ValidationError(f"{where}: {e}") from None
    t, ids = ex["target"], [c.id for c in q.candidates]
    ok = {
        "binary": isinstance(t, bool),
        "multiclass": isinstance(t, str) and t in ids,
        # multilabel targets must be complete: every listed candidate not in target is a real negative.
        "multilabel": isinstance(t, list) and all(isinstance(x, str) and x in ids for x in t) and len(set(t)) == len(t),
    }[q.type]
    if not ok:
        raise ValidationError(f"{where}: bad target {t!r} for {q.type} with candidates {ids}")
    return q


SOURCE_KEYS = {"source_id", "family", "provenance", "state", "questions", "hard_cases", "split"}
QUESTION_EXTRA = {"target", "hard_cases", "paraphrase_group", "notes"}


def expand_source(src: dict) -> list[dict]:
    """Authoring format (one state, many questions) -> one example per question, ids '<source_id>-q<i>'."""
    extra = set(src) - SOURCE_KEYS
    if extra or not isinstance(src.get("questions"), list) or not src["questions"]:
        raise ValidationError(f"source {src.get('source_id')!r}: needs non-empty 'questions'; unknown {sorted(extra)}")
    out = []
    for i, q in enumerate(src["questions"]):
        if not isinstance(q, dict) or "target" not in q:
            raise ValidationError(f"source {src.get('source_id')!r} question {i}: must be an object with 'target'")
        ex = {k: src[k] for k in ("source_id", "family", "provenance", "state", "split") if k in src}
        ex |= {"id": f"{src.get('source_id')}-q{i}", "question": {k: v for k, v in q.items() if k not in QUESTION_EXTRA},
               "target": q["target"], "hard_cases": list(src.get("hard_cases", [])) + list(q.get("hard_cases", []))}
        ex |= {k: q[k] for k in ("paraphrase_group", "notes") if k in q}
        validate_example(ex)
        out.append(ex)
    return out


def read_jsonl(path) -> list[dict]:
    with (gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def load(paths, splits=None) -> list[dict]:
    """Load + validate (source-format lines are expanded); optional split filter. Duplicate ids are an error."""
    out, seen = [], set()
    for p in [paths] if isinstance(paths, (str, Path)) else paths:
        for ex in (e for row in read_jsonl(p) for e in (expand_source(row) if "questions" in row else [row])):
            validate_example(ex)
            if ex["id"] in seen:
                raise ValidationError(f"duplicate example id {ex['id']!r} in {p}")
            seen.add(ex["id"])
            if splits is None or ex.get("split") in splits:
                out.append(ex)
    return out


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_for(key: str, weights: dict[str, float], salt: str = "selfjev") -> str:
    """Deterministic split from a group key (source/paraphrase group), so a group never straddles splits."""
    u = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:12], 16) / 16**12
    total, acc = sum(weights.values()), 0.0
    for name, w in weights.items():
        acc += w / total
        if u < acc:
            return name
    return name


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check(paths):
    bad = 0
    for p in paths:
        stats = Counter()
        for n, line in enumerate(open(p), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                for ex in expand_source(row) if "questions" in row else [row]:
                    stats[(ex["family"], validate_example(ex).type)] += 1
            except (ValidationError, json.JSONDecodeError) as e:
                bad += 1
                print(f"{p}:{n}: {e}")
        print(f"{p}: {sum(stats.values())} valid", dict(sorted(stats.items())))
    return bad


if __name__ == "__main__":  # python -m selfjev.data check FILE...
    if sys.argv[1:2] != ["check"] or len(sys.argv) < 3:
        sys.exit("usage: python -m selfjev.data check FILE.jsonl...")
    sys.exit(1 if _check(sys.argv[2:]) else 0)
