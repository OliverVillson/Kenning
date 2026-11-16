"""How much of a text repeats itself?

Emitting a kenning again only pays off if ideas recur. This measures the
simplest version of recurrence: for each span length n, the share of tokens
that sit inside an n-token span which already appeared earlier in the same
document. Exact repeats only, so it is a lower bound: a restated idea in
different words does not count.
"""

from __future__ import annotations


def repeat_coverage(ids: list[int], n: int) -> int:
    """Number of tokens in ``ids`` covered by an n-gram that occurred earlier in ``ids``."""
    seen = set()
    covered = [False] * len(ids)
    for start in range(len(ids) - n + 1):
        gram = tuple(ids[start : start + n])
        if gram in seen:
            for i in range(start, start + n):
                covered[i] = True
        else:
            seen.add(gram)
    return sum(covered)


def repeat_table(docs: list[list[int]], lengths: list[int]) -> list[dict]:
    total = sum(len(d) for d in docs)
    rows = []
    for n in lengths:
        covered = sum(repeat_coverage(d, n) for d in docs)
        rows.append({"span_tokens": n, "repeated_share": covered / total if total else 0.0})
    return rows
