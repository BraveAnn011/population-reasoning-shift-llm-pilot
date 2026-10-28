#!/usr/bin/env python3
"""Task 1 (Email 1 pilot): AI judges code each frozen, segmented letter.

Each call gives one judge the coding manual (§3 without §3.1, §5, §6; the
current draft, logged by hash) and one letter already split into the frozen
units, and asks for a move code and a focus code per unit
(ai_runs/prompts/judge/task1_code_email1.txt; coding_manual.md §11).

Judges: claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5, claude-fable-5-1.
k = 5 runs per judge per letter, each a fresh single-turn request, no system
prompt, no sampling or effort parameters sent. Call order is shuffled.

Real runs read only a FROZEN_segmentation file whose letters still match the
researcher's redaction sign-off, and only from committed code and prompts.
All real outputs go to data_private/.

Usage (from the repository root):
    python3 ai_runs/judge_letters.py --dry-run                 # synthetic, mock judges
    python3 ai_runs/judge_letters.py --smoke                   # synthetic, real API, 1 run, cost projection
    python3 ai_runs/judge_letters.py --real --frozen FROZEN    # frozen real letters, k runs
    python3 ai_runs/judge_letters.py --real --frozen FROZEN --fill-missing FILE
                                                               # re-run cells whose attempts all errored
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from generate_writers import (ANTHROPIC_MAX_TOKENS, NOT_SENT, RESULT_KEYS, ROOT,
                              call_anthropic, provenance, sdk_versions, sha256)
import segment_letters as SEG

TEMPLATE_FILE = ROOT / "prompts" / "judge" / "task1_code_email1.txt"
MANUAL_FILE = ROOT.parent / "prereg" / "coding_manual.md"

JUDGES = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"]
K_RUNS = 5
MOVES = {"OPN", "APO", "RSN", "TRG", "INQ", "NEWS", "SHR", "PLAN", "CLS", "OFF"}
FOCUS = {"SELF", "OTHER", "JOINT", "NEUTRAL"}

# List prices per million tokens (input, output), from the claude-api reference
# cached 2026-10-06; used only for the smoke-test cost projection.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-opus-5-5": (4.0, 20.0), "claude-fable-5-1": (10.0, 50.0)}

OUT = {"synthetic": ROOT / "dryrun",
       "real": ROOT.parent / "data_private" / "pilot_email1" / "judging"}
PREFIX = {"synthetic": "SYNTHETIC_", "real": ""}
# Writer-only result keys (Email label check) that a judge record leaves out.
WRITER_ONLY_KEYS = {"label_ok", "body"}


def manual_sections():
    """§3 up to §3.1, §5 and §6 of the coding manual, verbatim."""
    m = MANUAL_FILE.read_text(encoding="utf-8")
    def between(start, end):
        i, j = m.index(start), m.index(end)
        return m[i:j].strip()
    return "\n\n".join([
        between("## 3. Segmentation into move units", "### 3.1 Pilot deviation"),
        between("## 5. Move codes: Email 1", "## 6. Focus sub-code"),
        between("## 6. Focus sub-code", "## 7. Derived letter-level features"),
    ])


def numbered(units):
    # A line break inside a unit is shown as a space so each unit stays on one line.
    return "\n".join(f"[{n}] {u.replace(chr(10), ' ')}" for n, u in enumerate(units, 1))


def extract_array(text):
    """The first JSON array of objects in the reply: a fenced json block first,
    then a raw decode starting at each '[' in turn (so prose mentioning [3] or
    [name] does not break parsing)."""
    text = text or ""
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S)] + [text]
    dec = json.JSONDecoder()
    for cand in candidates:
        for i, ch in enumerate(cand):
            if ch != "[":
                continue
            try:
                obj, _ = dec.raw_decode(cand[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
                return obj
    return None


def parse_codes(text, n_units):
    """Return (codes, problem). codes is a list of {unit, move, focus} in unit order."""
    items = extract_array(text)
    if items is None:
        return None, "no_json"
    by_unit = {}
    for x in items:
        u = x.get("unit")
        if type(u) is not int or u in by_unit:
            return None, "wrong_units"
        by_unit[u] = x
    if sorted(by_unit) != list(range(1, n_units + 1)):
        return None, "wrong_units"
    codes = []
    for u in range(1, n_units + 1):
        move = str(by_unit[u].get("move", "")).strip().upper()
        focus = str(by_unit[u].get("focus", "")).strip().upper()
        if move not in MOVES or focus not in FOCUS:
            return None, "invalid_codes"
        codes.append({"unit": u, "move": move, "focus": focus})
    return codes, None


MOCK_FAIL = os.environ.get("JUDGE_MOCK_FAIL")  # tests only: bad_json, wrong_units, invalid, error


def call_mock(model, prompt):
    n = len(re.findall(r"^\[\d+\] ", prompt.split("<email>", 1)[1], re.M))
    base = {"raw_response": {"mock": True}, "model_returned": f"{model}-MOCK",
            "request_id": "req_SYNTHETIC", "response_id": "msg_SYNTHETIC",
            "stop_reason": "end_turn", "status": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}
    if MOCK_FAIL == "error":
        raise RuntimeError("SYNTHETIC mock API error")
    if MOCK_FAIL == "bad_json":
        return {**base, "text": "Here are the codes."}
    rng = random.Random(f"{model}{n}")
    codes = [{"unit": u, "move": rng.choice(sorted(MOVES)), "focus": rng.choice(sorted(FOCUS))}
             for u in range(1, n + 1)]
    if MOCK_FAIL == "wrong_units":
        codes = codes[:-1]
    if MOCK_FAIL == "invalid":
        codes[0]["move"] = "DIR"
    return {**base, "text": json.dumps(codes)}


def load_frozen(ctx, pinned=None):
    """The pinned (or else the latest) frozen segmentation, checked against the sign-off."""
    if pinned:
        path = Path(pinned)
    else:
        files = sorted(SEG.PATHS[ctx]["out"].glob(f"{SEG.PATHS[ctx]['prefix']}FROZEN_segmentation_*.json"))
        if not files:
            sys.exit(f"No frozen segmentation in {SEG.PATHS[ctx]['out']}. Run segment_letters.py --freeze first.")
        path = files[-1]
    payload = path.read_text(encoding="utf-8")
    frozen = json.loads(payload)
    letters, _, signoff_hash = SEG.load_signed_letters(ctx)
    if frozen.get("redaction_signoff_sha256") != signoff_hash:
        sys.exit("The frozen segmentation was made under a different sign-off.")
    if set(frozen["letters"]) != set(letters):
        sys.exit("The frozen segmentation does not cover exactly the signed-off letters.")
    for lid, v in frozen["letters"].items():
        if v["letter_sha256"] != letters[lid][1]:
            sys.exit(f"{lid} changed since segmentation was frozen.")
        ok, _ = SEG.check_coverage(letters[lid][0], v["units"])
        if not ok:
            sys.exit(f"{lid}: frozen units no longer cover the letter.")
    return path, sha256(payload), frozen, signoff_hash


def plan_cells(lids, runs, seed, fill_from, current):
    if not fill_from:
        plan = [(j, lid, r) for j in JUDGES for lid in lids for r in range(1, runs + 1)]
        random.Random(seed).shuffle(plan)
        return plan, {}, 1, runs, lids
    recs = [json.loads(l) for l in open(fill_from, encoding="utf-8")]
    for key in ("frozen_segmentation_sha256", "prompt_template_sha256", "manual_sha256",
                "script_sha256"):
        if {r[key] for r in recs} != {current[key]}:
            sys.exit(f"Refusing to refill: {key} differs from the original run.")
    if {r["judge"] for r in recs} != set(JUDGES):
        sys.exit("Refusing to refill: the judge list changed since the original run.")
    # The refill keeps the original run's plan: its run count, and every letter
    # in the frozen segmentation (including letters never attempted).
    planned = recs[0]["planned_runs"]
    grid = {(j, lid, r) for j in JUDGES for lid in lids for r in range(1, planned + 1)}
    kept = {(r["judge"], r["letter_id"], r["run"]) for r in recs if r["status"] != "error"}
    attempts = {}
    for r in recs:
        c = (r["judge"], r["letter_id"], r["run"])
        attempts[c] = max(attempts.get(c, 0), r["attempt"])
    return sorted(grid - kept), attempts, max(r["call_index"] for r in recs) + 1, planned, lids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--real", action="store_true")
    ap.add_argument("--fill-missing", metavar="FILE")
    ap.add_argument("--frozen", metavar="FILE", help="pin the frozen segmentation file")
    ap.add_argument("--runs", type=int, default=K_RUNS)
    ap.add_argument("--smoke-letters", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    if args.real and not args.frozen:
        sys.exit("A real run must pin the frozen segmentation with --frozen.")
    ctx = "real" if args.real else "synthetic"
    prov = provenance(__file__)
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    import subprocess
    manual_dirty = subprocess.run(["git", "status", "--porcelain", "--", str(MANUAL_FILE.relative_to(ROOT.parent))],
                                  cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    manual_blob = subprocess.run(["git", "rev-parse", f"HEAD:{MANUAL_FILE.relative_to(ROOT.parent)}"],
                                 cwd=ROOT.parent, capture_output=True, text=True).stdout.strip() or None
    if args.real and (prov["git_dirty"] is not False or manual_dirty):
        sys.exit("Refusing a real run from uncommitted code, prompts or manual. Commit first.")
    if args.fill_missing and not args.real and not Path(args.fill_missing).name.startswith("SYNTHETIC_"):
        sys.exit("--fill-missing on a synthetic run must name a SYNTHETIC_ file.")

    frozen_path, frozen_hash, frozen, signoff_hash = load_frozen(ctx, args.frozen)
    template = TEMPLATE_FILE.read_text(encoding="utf-8").strip()
    manual = manual_sections()
    current = {"frozen_segmentation_sha256": frozen_hash,
               "prompt_template_sha256": sha256(template), "manual_sha256": sha256(manual),
               "script_sha256": prov["script_sha256"]}
    lids = sorted(frozen["letters"])
    runs = args.runs
    if args.smoke:
        lids, runs = lids[:args.smoke_letters], 1
    plan, prior, first_index, runs, lids = plan_cells(lids, runs, args.seed, args.fill_missing, current)
    if not plan:
        print("Nothing to run.")
        return

    OUT[ctx].mkdir(parents=True, exist_ok=True)
    kind = "smoke_" if args.smoke else ""
    if args.fill_missing:
        out_path = Path(args.fill_missing)
        run_id = json.loads(open(out_path, encoding="utf-8").readline())["run_id"]
    else:
        # Never overwrite a log: wait for a new second if the name is taken.
        while True:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"task1_email1_{stamp}"
            out_path = OUT[ctx] / f"{PREFIX[ctx]}{kind}{run_id}.jsonl"
            if not out_path.exists():
                break
            time.sleep(0.2)
    versions = sdk_versions()
    caller = call_mock if args.dry_run else call_anthropic

    counts, cost = {}, {}
    with open(out_path, "a" if args.fill_missing else "x", encoding="utf-8", newline="\n") as out:
        for i, (judge, lid, run) in enumerate(plan, first_index):
            units = frozen["letters"][lid]["units"]
            prompt = (template.replace("{CODING_MANUAL_SECTIONS_3_TO_6}", manual)
                              .replace("{NUMBERED_UNITS}", numbered(units)))
            ts = datetime.now(timezone.utc)
            t0 = time.time()
            rec = {"run_id": run_id, "call_index": i, "seed": args.seed,
                   "mode": ("dry_run" if args.dry_run else "smoke" if args.smoke else
                            "fill_missing" if args.fill_missing else "real"),
                   "fill_missing": bool(args.fill_missing),
                   "attempt": prior.get((judge, lid, run), 0) + 1,
                   "judge": judge, "model_requested": judge, "letter_id": lid, "run": run,
                   "planned_runs": runs, "n_units": len(units),
                   "frozen_segmentation": frozen_path.name, "frozen_segmentation_sha256": frozen_hash,
                   "letter_sha256": frozen["letters"][lid]["letter_sha256"],
                   "redaction_signoff_sha256": signoff_hash,
                   "timestamp_utc": ts.isoformat(), "date": ts.date().isoformat(),
                   "prompt_file": str(TEMPLATE_FILE.relative_to(ROOT.parent)),
                   "prompt_template_sha256": sha256(template), "manual_sha256": sha256(manual),
                   "manual_file_sha256": sha256(MANUAL_FILE.read_text(encoding="utf-8")),
                   "manual_git_blob": manual_blob, "manual_git_dirty": bool(manual_dirty),
                   "prompt_sha256": sha256(prompt),
                   "system_prompt": None, "temperature": NOT_SENT, "effort": NOT_SENT,
                   "max_tokens": ANTHROPIC_MAX_TOKENS, "sdk_max_retries": 0,
                   "sdk_versions": versions, **prov,
                   **{k: None for k in RESULT_KEYS if k not in WRITER_ONLY_KEYS}}
            try:
                rec.update(caller(judge, prompt))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["request_id"] = getattr(e, "request_id", None)
            rec["latency_s"] = round(time.time() - t0, 3)
            codes, problem = (None, None) if rec["error"] else parse_codes(rec["text"], len(units))
            rec["status"] = ("error" if rec["error"] else
                             rec["status"] if rec.get("status") in ("refusal", "truncated", "other_stop") else
                             problem or "ok")
            # Codes are kept only from a complete, valid reply.
            rec["codes"] = codes if rec["status"] == "ok" else None
            rec["model_match"] = None if rec["error"] else rec["model_returned"] == (
                f"{judge}-MOCK" if args.dry_run else judge)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            u = rec["usage"] or {}
            pi, po = PRICES[judge]
            cost[judge] = cost.get(judge, 0) + ((u.get("input_tokens") or 0) * pi
                                                + (u.get("output_tokens") or 0) * po) / 1e6
            flag = "  MODEL MISMATCH ({})".format(rec["model_returned"]) if rec["model_match"] is False else ""
            print(f"[{i - first_index + 1}/{len(plan)}] {judge:18s} {lid} r{run}  {rec['status']}{flag}")
            if not args.dry_run:
                time.sleep(args.sleep)

    print(f"\nstatus counts: {counts} -> {out_path}")
    if args.smoke:
        n_real = len(list(SEG.PATHS["real"]["letters"].glob(SEG.PATHS["real"]["pattern"]))) or 22
        per_letter = {j: c / len(lids) for j, c in cost.items()}
        total = sum(per_letter.values()) * n_real * K_RUNS
        print("measured cost per letter per run: " +
              ", ".join(f"{j} ${c:.4f}" for j, c in per_letter.items()))
        print(f"projected real run: {n_real} letters x {len(JUDGES)} judges x {K_RUNS} runs "
              f"= {n_real * len(JUDGES) * K_RUNS} calls, at least ${total:.2f} at list prices "
              f"(lower bound: synthetic letters are shorter than real ones)")


if __name__ == "__main__":
    main()
