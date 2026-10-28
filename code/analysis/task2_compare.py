#!/usr/bin/env python3
"""Task 2 (Email 1 pilot): judges' estimated group differences vs the coded ones.

Reads a Task 2 log, the Task 1 log and the frozen segmentation. For each
comparison, field and judge it reports the judge's mean estimate per group
(Set 1/Set 2 mapped back to groups), the estimated difference A - B, and the
difference in the Task 1 consensus codes (all judges and runs, modal code per
unit, TIE when the top codes tie). The ratio estimated / coded difference
(> 1: the judge sees a bigger difference than the codes show) is a pilot
check against the judges' own codes, not the DESIGN §7 amplification index,
which needs human codes and human letters.

"Coded" values: apology, other-first, plan and closing are % of letters;
self, other and off are % of move units (the prompt asks per sentence, so
these are approximate); mean_sentences counts . ! ? 。！？ in the letter.

Usage (from the repository root):
    python3 analysis/task2_compare.py <task2 log> <task1 log> <frozen segmentation> [--show-text]
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from task1_agreement import KEY_FILE, modal

FIELDS = ["apology_pct", "other_first_pct", "plan_pct", "closing_pct",
          "self_pct", "other_pct", "off_pct", "mean_sentences"]


def coded_values(task1_log, frozen_path):
    frozen = json.load(open(frozen_path, encoding="utf-8"))["letters"]
    votes = defaultdict(lambda: defaultdict(list))
    cells = {}
    for r in map(json.loads, open(task1_log, encoding="utf-8")):
        c = (r["judge"], r["letter_id"], r["run"])
        if c not in cells or r["attempt"] > cells[c]["attempt"]:
            cells[c] = r
    for r in cells.values():
        if r["status"] == "ok":
            for c in r["codes"]:
                votes[r["letter_id"]][c["unit"]].append((c["move"], c["focus"]))
    per_letter = {}
    for lid, units in votes.items():
        mv = [modal([m for m, _ in units[u]]) for u in sorted(units)]
        fo = [modal([f for _, f in units[u]]) for u in sorted(units)]
        first = lambda code: next((i for i, m in enumerate(mv) if m == code), len(mv))
        text = " ".join(frozen[lid]["units"])
        per_letter[lid] = {
            "apology_pct": 100 * ("APO" in mv), "other_first_pct": 100 * (first("INQ") < first("NEWS")),
            "plan_pct": 100 * ("PLAN" in mv), "closing_pct": 100 * ("CLS" in mv),
            "self_pct": 100 * fo.count("SELF") / len(fo), "other_pct": 100 * fo.count("OTHER") / len(fo),
            "off_pct": 100 * mv.count("OFF") / len(mv),
            "mean_sentences": len(re.findall(r"[.!?。！？]+", text)),
        }
    return per_letter


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("task1_log")
    ap.add_argument("frozen")
    ap.add_argument("--show-text", action="store_true", help="print one Part A per judge per comparison")
    ap.add_argument("--key", default=str(KEY_FILE), help="redaction key (letter_id, group)")
    args = ap.parse_args()

    cells = {}
    for r in map(json.loads, open(args.log, encoding="utf-8")):
        c = (r["judge"], r["input_id"], r["run"])
        if c not in cells or r["attempt"] > cells[c]["attempt"]:
            cells[c] = r
    judges = sorted({j for j, _, _ in cells})
    key = {row["letter_id"]: row["group"] for row in csv.DictReader(open(args.key, encoding="utf-8"))}
    for r in cells.values():  # the log holds letter ids only; groups come from the key
        if r.get("set_1_ids"):
            r["set_1_group"], r["set_2_group"] = key[r["set_1_ids"][0]], key[r["set_2_ids"][0]]
    coded = coded_values(args.task1_log, args.frozen)
    print(f"log: {Path(args.log).name}  |  {len(cells)} cells")
    for j in judges:
        st = Counter(r["status"] for (jj, _, _), r in cells.items() if jj == j)
        print(f"   {j:18s} {st['ok']}/{sum(st.values())} ok  {dict(st)}")

    for cid in sorted({i for _, i, _ in cells}):
        a, b = cid.split("__vs__")
        any_rec = next(r for (_, i, _), r in cells.items() if i == cid)
        ids = {any_rec["set_1_group"]: any_rec["set_1_ids"], any_rec["set_2_group"]: any_rec["set_2_ids"]}
        missing = [l for g in ids for l in ids[g] if l not in coded]
        if missing:
            print(f"   (no Task 1 codes for {len(missing)} letters; they are left out of the coded values)")
        ids = {g: [l for l in v if l in coded] for g, v in ids.items()}
        mean = lambda g, f: sum(coded[l][f] for l in ids[g]) / len(ids[g]) if ids[g] else float("nan")
        print(f"\n== {a} (A, {len(ids[a])} letters) vs {b} (B, {len(ids[b])} letters)")
        print(f"   {'field':16s} {'coded A':>8s} {'coded B':>8s} {'A-B':>7s} | judge estimates A / B, A-B (ratio to coded A-B)")
        for f in FIELDS:
            ca, cb = mean(a, f), mean(b, f)
            cells_txt = []
            for j in judges:
                ests = defaultdict(list)
                for (jj, i, _), r in cells.items():
                    if jj == j and i == cid and r["status"] == "ok":
                        ests[r["set_1_group"]].append(r["answer"]["set_1"][f])
                        ests[r["set_2_group"]].append(r["answer"]["set_2"][f])
                if not ests[a] or not ests[b]:
                    cells_txt.append(f"{j.replace('claude-', '')} n/a")
                    continue
                ea, eb = sum(ests[a]) / len(ests[a]), sum(ests[b]) / len(ests[b])
                # The ratio is shown only when the coded difference is not tiny.
                ratio = f"{(ea - eb) / (ca - cb):.1f}x" if abs(ca - cb) >= (0.5 if f == "mean_sentences" else 5) \
                    else "coded A-B ~ 0"
                cells_txt.append(f"{j.replace('claude-', '')} {ea:.0f}/{eb:.0f}, {ea - eb:+.0f} ({ratio})")
            print(f"   {f:16s} {ca:8.1f} {cb:8.1f} {ca - cb:+7.1f} | " + "; ".join(cells_txt))
        if args.show_text:
            print("   Part A (run 1 of each judge; Set 1/Set 2 named by group):")
            for j in judges:
                r = cells.get((j, cid, 1))
                if r and r["part_a"]:
                    t = r["part_a"].replace("Set 1", f"[{r['set_1_group']}]").replace("Set 2", f"[{r['set_2_group']}]")
                    print(f"   -- {j}:\n      " + t.replace("\n", "\n      "))


if __name__ == "__main__":
    main()
