#!/usr/bin/env python3
"""Task 4 (Email 1 pilot): who do the judges think wrote each letter?

Reads a Task 4 log and reports, per judge and per writer group (group labels
are joined from the redaction key; the log holds letter ids only):
  1. Valid-reply rate.
  2. Answer distributions: share answering "ai"; native language; where the
     writer grew up; mean confidence.
  3. Test-retest per judge: Fleiss' kappa over the k runs, letters as subjects.
  4. Cues: share of quoted cues found verbatim in the letter, and the most
     frequent quoted cues per AI group (human-letter quotes are not printed
     unless --show-human-cues).

There is no right answer for a persona question about an AI letter; for AI
letters the distributions show how the persona reads, not accuracy.

Usage (from the repository root):
    python3 analysis/task4_attribution.py data_private/pilot_email1/judging/<run>.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import csv

from task1_agreement import KEY_FILE, fleiss_kappa, fmt

FIELDS = ("author", "native_language", "grew_up")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--show-human-cues", action="store_true")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--key", default=str(KEY_FILE), help="redaction key (letter_id, group)")
    args = ap.parse_args()

    cells = {}
    for r in map(json.loads, open(args.log, encoding="utf-8")):
        c = (r["judge"], r["input_id"], r["run"])
        if c not in cells or r["attempt"] > cells[c]["attempt"]:
            cells[c] = r
    judges = sorted({j for j, _, _ in cells})
    runs = sorted({k for _, _, k in cells})
    key = {row["letter_id"]: row["group"] for row in csv.DictReader(open(args.key, encoding="utf-8"))}
    group = {r["input_id"]: key[r["input_id"]] for r in cells.values()}
    groups = sorted(set(group.values()))
    print(f"log: {Path(args.log).name}  |  {len(cells)} cells: {len(judges)} judges x "
          f"{len(group)} letters x {len(runs)} runs")

    print("\n1. Valid replies (status ok)")
    for j in judges:
        st = Counter(r["status"] for (jj, _, _), r in cells.items() if jj == j)
        print(f"   {j:18s} {st['ok']}/{sum(st.values())}  {dict(st)}")
    ok = [r for r in cells.values() if r["status"] == "ok"]

    print("\n2. Answers by writer group (all runs; % of valid replies; mean confidence 1-5)")
    for g in groups:
        n_letters = sum(1 for v in group.values() if v == g)
        print(f"   {g} ({n_letters} letters)")
        for j in judges:
            rs = [r["answer"] for r in ok if r["judge"] == j and group[r["input_id"]] == g]
            if not rs:
                continue
            parts = []
            for f in FIELDS:
                dist = Counter(a[f] for a in rs)
                conf = sum(a[f"{f}_confidence"] for a in rs) / len(rs)
                top = " ".join(f"{k} {v / len(rs):.0%}" for k, v in dist.most_common())
                parts.append(f"{f}: {top} (c{conf:.1f})")
            print(f"     {j.replace('claude-', ''):10s} " + " | ".join(parts))

    print(f"\n3. Test-retest (Fleiss' kappa over {len(runs)} runs; letters with all runs valid)")
    for f in FIELDS:
        row = []
        for j in judges:
            by = defaultdict(list)
            for r in ok:
                if r["judge"] == j:
                    by[r["input_id"]].append(r["answer"][f])
            table = [Counter(v) for v in by.values() if len(v) == len(runs)]
            row.append(f"{j.replace('claude-', '')} {fmt(fleiss_kappa(table))}")
        print(f"   {f:15s} " + "   ".join(row))

    print("\n4. Cues")
    for j in judges:
        found = [r["cue_quotes_found"] for r in ok if r["judge"] == j and r["cue_quotes_found"] is not None]
        n_cues = [len(r["answer"]["cues"]) for r in ok if r["judge"] == j]
        if n_cues:
            print(f"   {j.replace('claude-', ''):10s} {sum(n_cues) / len(n_cues):.1f} cues per reply, "
                  f"{sum(found) / len(found) if found else 0:.0%} quoted verbatim")
    for g in groups:
        if not g.startswith("AI-") and not args.show_human_cues:
            continue
        quotes = Counter(c["quote"].strip().lower() for r in ok if group[r["input_id"]] == g
                         for c in r["answer"]["cues"])
        print(f"   most quoted in {g}: " + "; ".join(f'"{q}" x{n}' for q, n in quotes.most_common(args.top)))


if __name__ == "__main__":
    main()
