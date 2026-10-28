#!/usr/bin/env python3
"""Task 1 (Email 1 pilot): judge reliability and the self-recognition check.

Reads a Task 1 judging log and reports, for move and focus codes separately:
  1. Valid-reply rate per judge (status ok).
  2. Test-retest reliability per judge: Fleiss' kappa over the k runs, with
     units as subjects (coding_manual.md §11).
  3. Between-judge agreement: Cohen's kappa on each judge's modal code per
     unit (a tie counts as its own category, TIE).
  4. Self-recognition check: agreement of the writer model (claude-sonnet-5-5)
     with the other judges on its own letters versus on the other letters.

There is no human gold standard in the pilot, so nothing here measures
accuracy. Group labels come from the redaction key; only aggregates are printed.

Usage (from the repository root):
    python3 analysis/task1_agreement.py data_private/pilot_email1/judging/<run>.jsonl
    python3 analysis/task1_agreement.py ai_runs/dryrun/SYNTHETIC_task1_email1_<run>.jsonl --no-key
"""

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = ROOT / "data_private" / "pilot_email1" / "redaction" / "KEY_do_not_open" / "key.csv"
WRITER = "claude-sonnet-5-5"
FIELDS = ("move", "focus")


def fleiss_kappa(table):
    """table: list of Counters, one per subject, each summing to the same n raters."""
    table = [t for t in table if sum(t.values()) > 1]
    if not table:
        return None
    n = sum(table[0].values())
    if any(sum(t.values()) != n for t in table):
        return None
    cats = sorted({c for t in table for c in t})
    N = len(table)
    p_j = {c: sum(t[c] for t in table) / (N * n) for c in cats}
    P_i = [(sum(v * v for v in t.values()) - n) / (n * (n - 1)) for t in table]
    P_bar, P_e = sum(P_i) / N, sum(p * p for p in p_j.values())
    return None if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def cohen_kappa(a, b):
    if not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def modal(codes):
    c = Counter(codes).most_common()
    return "TIE" if len(c) > 1 and c[0][1] == c[1][1] else c[0][0]


def fmt(x):
    return "  n/a" if x is None else f"{x:5.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--no-key", action="store_true", help="skip the group breakdown")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.log, encoding="utf-8")]
    # Keep the latest attempt per cell (refills append later attempts).
    cells = {}
    for r in recs:
        c = (r["judge"], r["letter_id"], r["run"])
        if c not in cells or r["attempt"] > cells[c]["attempt"]:
            cells[c] = r
    judges = sorted({j for j, _, _ in cells})
    letters = sorted({l for _, l, _ in cells})
    runs = sorted({k for _, _, k in cells})

    print(f"log: {Path(args.log).name}  |  {len(cells)} cells: {len(judges)} judges x "
          f"{len(letters)} letters x {len(runs)} runs")

    print("\n1. Valid replies (status ok)")
    for j in judges:
        st = Counter(r["status"] for (jj, _, _), r in cells.items() if jj == j)
        print(f"   {j:18s} {st['ok']}/{sum(st.values())}  {dict(st)}")

    # codes[j][(letter, unit)][field] = list over runs (ok replies only)
    codes = {j: defaultdict(lambda: defaultdict(list)) for j in judges}
    for (j, lid, _), r in cells.items():
        if r["status"] != "ok":
            continue
        for c in r["codes"]:
            for f in FIELDS:
                codes[j][(lid, c["unit"])][f].append(c[f])

    print(f"\n2. Test-retest (Fleiss' kappa over {len(runs)} runs; units with all runs valid)")
    for f in FIELDS:
        row = []
        for j in judges:
            table = [Counter(v[f]) for v in codes[j].values() if len(v[f]) == len(runs)]
            row.append(f"{j.replace('claude-', '')} {fmt(fleiss_kappa(table))}")
        print(f"   {f:6s} " + "   ".join(row))

    modal_codes = {j: {u: {f: modal(v[f]) for f in FIELDS} for u, v in codes[j].items()} for j in judges}

    def pair_kappa(j1, j2, f, unit_filter=lambda u: True):
        units = sorted(u for u in set(modal_codes[j1]) & set(modal_codes[j2]) if unit_filter(u))
        return cohen_kappa([modal_codes[j1][u][f] for u in units],
                           [modal_codes[j2][u][f] for u in units]), len(units)

    print("\n3. Between-judge agreement (Cohen's kappa on modal codes)")
    for f in FIELDS:
        print(f"   {f}")
        for j1, j2 in itertools.combinations(judges, 2):
            k, n = pair_kappa(j1, j2, f)
            print(f"     {j1.replace('claude-', ''):10s} vs {j2.replace('claude-', ''):10s} {fmt(k)}  (n={n} units)")

    if args.no_key or not KEY_FILE.exists():
        print("\n4. Self-recognition check: skipped (no key).")
        return
    group = {row["letter_id"]: row["group"] for row in csv.DictReader(open(KEY_FILE, encoding="utf-8"))}
    ai = {l for l in letters if group.get(l, "").startswith("AI-")}
    print(f"\n4. Self-recognition check: {WRITER} vs each other judge")
    print(f"   own letters = {len(ai)} AI letters written by {WRITER}; other = {len(letters) - len(ai)} human letters")
    print("   (with 2 human letters the 'other' column is a pipeline check, not a comparison)")
    for f in FIELDS:
        print(f"   {f}")
        for j in judges:
            if j == WRITER:
                continue
            k_own, n_own = pair_kappa(WRITER, j, f, lambda u: u[0] in ai)
            k_oth, n_oth = pair_kappa(WRITER, j, f, lambda u: u[0] not in ai)
            print(f"     vs {j.replace('claude-', ''):10s} own {fmt(k_own)} (n={n_own})   other {fmt(k_oth)} (n={n_oth})")

    print("\n5. Code distribution by group (all judges, modal codes)")
    for f in FIELDS:
        by = defaultdict(Counter)
        for j in judges:
            for (lid, _), v in modal_codes[j].items():
                by[group.get(lid, "?")][v[f]] += 1
        print(f"   {f}")
        for g in sorted(by):
            tot = sum(by[g].values())
            top = ", ".join(f"{c} {n / tot:.0%}" for c, n in by[g].most_common())
            print(f"     {g:9s} (n={tot:4d}) {top}")


if __name__ == "__main__":
    main()
