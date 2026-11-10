"""Bonus: question 12 needs two documents (E5 and E7). Split it into one sub-question per car,
search each one separately, and fuse the results with RRF.

  python bonus_q12.py                      # hybrid, like the final version
  python bonus_q12.py --mode bm25          # words only (no new vectors needed)
"""
import argparse
import json
from pathlib import Path

from eval_retrieval import build_index, is_hit, reciprocal_rank
from rag.search import search, search_multi

# One sub-question per electric model. Note "טווח נסיעה" and not "טווח הנסיעה": BM25 has no
# Hebrew prefix handling, so the words must match the document exactly. In a real system an LLM call (or the list of models
# with type: model in the metadata) would produce these; here they are written by hand.
SUB_QUESTIONS = [
    "טווח נסיעה של אופק E5 חשמלי (OF-E5)",
    "טווח נסיעה של אופק E7 חשמלי (OF-E7)",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    q = next(q for q in json.loads(Path("golden.json").read_text(encoding="utf-8")) if q["id"] == 12)
    index = build_index("headings")
    if args.mode != "bm25":
        before = search(index, q["question"], args.mode, args.k, True)
        print(f"one query:   {'✓' if is_hit(q, before) else '✗'}  MRR {reciprocal_rank(q, before):.2f}  {[c.id for c in before]}")
    after = search_multi(index, SUB_QUESTIONS, args.mode, args.k, True)
    print(f"two queries: {'✓' if is_hit(q, after) else '✗'}  MRR {reciprocal_rank(q, after):.2f}  {[c.id for c in after]}")


if __name__ == "__main__":
    main()
