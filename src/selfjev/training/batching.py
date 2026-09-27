"""Whole states per micro-batch: every question about a text trains in the same packed tree."""

from ..engine.tree import build_tree


def group_by_state(items):
    """Stable reorder so each state's questions are contiguous: leaf order then matches item order."""
    order = {}
    for it in items:
        order.setdefault(it["state"], []).append(it)
    return [it for group in order.values() for it in group]


def trees_for(items, roots):
    groups = {}
    for it in items:
        groups.setdefault(it["state"], []).append(it)
    return [build_tree(roots[s], [(it["q"], it["ids"]) for it in g]) for s, g in groups.items()]


def tree_len(items, roots):
    return len(roots[items[0]["state"]]) + sum(len(it["q"]) + sum(map(len, it["ids"])) for it in items)


def micro_batches(items, roots, max_batch_tokens, rng, bucket=256):
    """Whole states (all their questions) per micro-batch, length-bucketed; padded packed tokens (trees x longest
    tree) stay under max_batch_tokens unless one tree alone exceeds it."""
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(it["state"], []).append(i)
    keys = list(groups)
    rng.shuffle(keys)
    size = {k: tree_len([items[i] for i in groups[k]], roots) for k in keys}
    out = []
    for c in range(0, len(keys), bucket):
        cur, n, w = [], 0, 0
        for k in sorted(keys[c : c + bucket], key=size.__getitem__):
            if cur and (n + 1) * max(w, size[k]) > max_batch_tokens:
                out.append(cur)
                cur, n, w = [], 0, 0
            cur, n, w = cur + groups[k], n + 1, max(w, size[k])
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out
