"""python -m selfjev.data check FILE.jsonl...: validate examples or authoring sources; exit 1 on any bad line."""

import sys

from . import _check

if sys.argv[1:2] != ["check"] or len(sys.argv) < 3:
    sys.exit("usage: python -m selfjev.data check FILE.jsonl...")
sys.exit(1 if _check(sys.argv[2:]) else 0)
