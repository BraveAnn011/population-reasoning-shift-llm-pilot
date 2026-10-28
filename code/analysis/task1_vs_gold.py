#!/usr/bin/env python3
"""Task 1 (Email 1 pilot): AI judges against the researcher's gold codes.

Reads the filled coding sheet and a Task 1 judging log and reports, for move
and focus codes separately (coding_manual.md §11):
  1. Primary: Cohen's kappa of each single judge run with gold, averaged over
     the k runs, plus raw agreement.
  2. Secondary: kappa of each judge's 5-run modal code with gold (TIE counts
     as disagreement).
  3. The code pairs most often confused with gold, per judge, and how many
     fall on a §5.1 boundary pair.
Separately for AI-written and human letters. Only aggregates are printed.

Usage (from the repository root):
    python3 analysis/task1_vs_gold.py data_private/pilot_email1/gold/coding_sheet_done.csv \
        data_private/pilot_email1/judging/<run>.jsonl
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from task1_agreement import FIELDS, KEY_FILE, cohen_kappa, fmt, modal

MOVES = {"OPN", "APO", "RSN", "TRG", "INQ", "NEWS", "SHR", "PLAN", "CLS", "OFF"}
FOCUS = {"SELF", "OTHER", "JOINT", "NEUTRAL"}
BOUNDARY_PAIRS = {frozenset(p) for p in (("APO", "SHR"), ("OPN", "SHR"), ("APO", "OPN"),
                                         ("SHR", "INQ"), ("RSN", "NEWS"), ("NEWS", "OFF"),
                                         ("OPN", "NEWS"), ("PLAN", "CLS"), ("CLS", "SHR"),
                                         ("NEWS", "SHR"), ("TRG", "SHR"), ("TRG", "RSN"))}


def load_gold(path):
    gold, bad = {}, []
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        u = (row["letter_id"], int(row["unit"]))
        move, focus = row["move"].strip().upper(), row["focus"].strip().upper()
        if move not in MOVES or focus not in FOCUS:
            bad.append(f"{u[0]} u{u[1]}")
        gold[u] = {"move": move, "focus": focus}
    if bad:
        raise SystemExit(f"{len(bad)} rows have a missing or unknown code, e.g. {', '.join(bad[:5])}")
    return gold


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold")
    ap.add_argument("log")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    cells = {}
    for r in map(json.loads, open(args.log, encoding="utf-8")):
        c = (r["judge"], r["letter_id"], r["run"])
        if c not in cells or r["attempt"] > cells[c]["attempt"]:
            cells[c] = r
    per_run = defaultdict(lambda: defaultdict(dict))   # judge -> run -> unit -> codes
    for (j, lid, k), r in cells.items():
        if r["status"] == "ok":
            for c in r["codes"]:
                per_run[j][k][(lid, c["unit"])] = c
    if {u for j in per_run for k in per_run[j] for u in per_run[j][k]} != set(gold):
        raise SystemExit("The coding sheet and the judging log cover different units.")
    group = {row["letter_id"]: row["group"] for row in csv.DictReader(open(KEY_FILE, encoding="utf-8"))}
    subsets = {"AI letters": lambda u: group[u[0]].startswith("AI-"),
               "human letters": lambda u: not group[u[0]].startswith("AI-")}

    print(f"gold: {Path(args.gold).name} ({len(gold)} units)  |  log: {Path(args.log).name}")
    for name, keep in subsets.items():
        units = sorted(u for u in gold if keep(u))
        print(f"\n== {name} (n={len(units)} units)")
        for f in FIELDS:
            print(f"   {f}: single run vs gold (mean kappa, mean agreement)  |  5-run modal vs gold")
            for j in sorted(per_run):
                ks, acc = [], []
                for k, codes in per_run[j].items():
                    us = [u for u in units if u in codes]
                    if not us:
                        continue
                    ks.append(cohen_kappa([codes[u][f] for u in us], [gold[u][f] for u in us]))
                    acc.append(sum(codes[u][f] == gold[u][f] for u in us) / len(us))
                ks = [x for x in ks if x is not None]
                mod = {u: modal([per_run[j][k][u][f] for k in per_run[j] if u in per_run[j][k]]) for u in units}
                km = cohen_kappa([mod[u] for u in units], [gold[u][f] for u in units])
                print(f"     {j.replace('claude-', ''):10s} {fmt(sum(ks) / len(ks) if ks else None)}  "
                      f"{sum(acc) / len(acc) if acc else 0:5.0%}   |  {fmt(km)}")
        print("   move: most frequent modal-vs-gold confusions (judge code -> gold code)")
        for j in sorted(per_run):
            conf = Counter()
            for u in units:
                m = modal([per_run[j][k][u]["move"] for k in per_run[j] if u in per_run[j][k]])
                if m != gold[u]["move"]:
                    conf[(m, gold[u]["move"])] += 1
            on_b = sum(n for (a, b), n in conf.items() if frozenset((a, b)) in BOUNDARY_PAIRS)
            top = ", ".join(f"{a}->{b} {n}" for (a, b), n in conf.most_common(4))
            print(f"     {j.replace('claude-', ''):10s} {sum(conf.values())} wrong, {on_b} on a §5.1 pair: {top}")


if __name__ == "__main__":
    main()
