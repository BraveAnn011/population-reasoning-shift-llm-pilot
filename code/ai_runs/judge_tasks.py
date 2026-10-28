#!/usr/bin/env python3
"""Task 4 (attribution) and Task 2 (set comparison), Email 1 pilot.

Task 4 (--task attribute): one redacted letter per call, as written. The judge
guesses human vs AI, the writer's native language and where they grew up,
with confidence and the quoted cues behind each guess
(prompts/judge/task4_attribute_email1.txt). Always a separate call from the
Task 1 coding.

Task 2 (--task compare): two unlabeled sets of letters per call. The judge
describes how they differ (Part A) and estimates eight figures per set
(Part B) (prompts/judge/task2_compare_email1.txt). The groups compared are in
COMPARISONS; which group is Set 1 alternates across runs, and letter order
inside each set is shuffled per call.

Judges: claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5, claude-fable-5-1.
k = 5 runs per input, each a fresh single-turn request, no system prompt, no
sampling or effort parameters sent. Call order is shuffled.

Real runs read only the signed-off redacted letters, and only from committed
code and prompts. Group labels (for Task 2 sets and for the log) come from
the redaction key and are never shown to a judge. All real outputs go to
data_private/.

Usage (from the repository root):
    python3 ai_runs/judge_tasks.py --task attribute --dry-run
    python3 ai_runs/judge_tasks.py --task attribute --smoke
    python3 ai_runs/judge_tasks.py --task attribute --real
    python3 ai_runs/judge_tasks.py --task attribute --real --fill-missing FILE
    (the same for --task compare)
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from generate_writers import (ANTHROPIC_MAX_TOKENS, NOT_SENT, RESULT_KEYS, ROOT,
                              call_anthropic, provenance, sdk_versions, sha256)
from judge_letters import JUDGES, K_RUNS, OUT, PREFIX, PRICES, WRITER_ONLY_KEYS
import segment_letters as SEG

TEMPLATES = {"attribute": ROOT / "prompts" / "judge" / "task4_attribute_email1.txt",
             "compare": ROOT / "prompts" / "judge" / "task2_compare_email1.txt"}
RUN_NAME = {"attribute": "task4_attribute_email1", "compare": "task2_compare_email1"}
KEY = {"synthetic": ROOT / "dryrun" / "SYNTHETIC_key.csv",
       "real": ROOT.parent / "data_private" / "pilot_email1" / "redaction" / "KEY_do_not_open" / "key.csv"}

# Task 2 pairs that can run with AI letters only. Human comparisons wait for
# more human letters (coding_manual.md §11.1).
COMPARISONS = [("AI-US-EN", "AI-JP-EN"), ("AI-US-EN", "AI-EN")]

ATTR_ENUMS = {"author": {"human", "ai"},
              "native_language": {"japanese", "english", "other", "none"},
              "grew_up": {"japan", "united_states", "other", "none"}}
COMPARE_FIELDS = ["apology_pct", "other_first_pct", "plan_pct", "closing_pct",
                  "self_pct", "other_pct", "off_pct", "mean_sentences"]


def extract_object(text, required):
    """(object, offset): the last JSON object in the reply that has all the
    required keys, fenced json blocks first, then a raw decode from each '{'.
    offset is where it (or its fence) starts in the reply."""
    text = text or ""
    candidates = [(m.group(1), m.start()) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S)]
    dec = json.JSONDecoder()
    for group in (candidates, [(text, 0)]):
        found = None
        for cand, base in group:
            for i, ch in enumerate(cand):
                if ch != "{":
                    continue
                try:
                    obj, _ = dec.raw_decode(cand[i:])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and required <= set(obj):
                    found = (obj, base if group is candidates else i)
        if found is not None:
            return found
    return None, None


def parse_attribute(text, letter):
    """Return (answer, problem, cues_found). cues_found: share of quotes that
    appear in the letter (whitespace-normalised)."""
    obj, _ = extract_object(text, set(ATTR_ENUMS) | {"cues"})
    if obj is None:
        return None, "no_json", None
    ans = {}
    for k, allowed in ATTR_ENUMS.items():
        v = obj.get(k)
        v = v.strip().lower() if isinstance(v, str) else None
        c = obj.get(f"{k}_confidence")
        if v not in allowed or type(c) is not int or not 1 <= c <= 5:
            return None, "invalid_fields", None
        ans[k], ans[f"{k}_confidence"] = v, c
    cues = obj.get("cues")
    if not isinstance(cues, list) or not all(isinstance(x, dict) and isinstance(x.get("quote"), str)
                                             for x in cues):
        return None, "invalid_fields", None
    ans["cues"] = [{"quote": x["quote"], "suggests": str(x.get("suggests", ""))} for x in cues]
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    found = [norm(x["quote"]) in norm(letter) for x in ans["cues"] if norm(x["quote"])]
    return ans, None, (sum(found) / len(found) if found else None)


def parse_compare(text):
    """Return (estimates, part_a, problem)."""
    obj, offset = extract_object(text, {"set_1", "set_2"})
    if obj is None:
        return None, None, "no_json"
    est = {}
    for s in ("set_1", "set_2"):
        if not isinstance(obj[s], dict):
            return None, None, "invalid_fields"
        est[s] = {}
        for f in COMPARE_FIELDS:
            v = obj[s].get(f)
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0 or (f.endswith("_pct") and v > 100):
                return None, None, "invalid_fields"
            est[s][f] = v
    # Part A is the prose before the JSON (or its fence), ending at a "Part B" heading.
    part_a = re.split(r"(?im)^[\s*#_]*part\s*b\b", text[:offset], maxsplit=1)[0]
    part_a = re.sub(r"^[\s*#_]*part\s*a\b[.:*_\s]*", "", part_a.strip(), flags=re.I).strip() or None
    if part_a is None:
        return None, None, "no_part_a"
    return est, part_a, None


MOCK_FAIL = os.environ.get("JUDGE_MOCK_FAIL")  # tests only: bad_json, invalid, error


def call_mock(model, prompt):
    base = {"raw_response": {"mock": True}, "model_returned": f"{model}-MOCK",
            "request_id": "req_SYNTHETIC", "response_id": "msg_SYNTHETIC",
            "stop_reason": "end_turn", "status": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}
    if MOCK_FAIL == "error":
        raise RuntimeError("SYNTHETIC mock API error")
    if MOCK_FAIL == "bad_json":
        return {**base, "text": "I can't tell."}
    rng = random.Random(f"{model}{len(prompt)}")
    if "<set_1>" in prompt:
        est = {s: {f: rng.randint(0, 100) if f.endswith("_pct") else rng.randint(4, 12)
                   for f in COMPARE_FIELDS} for s in ("set_1", "set_2")}
        if MOCK_FAIL == "invalid":
            est["set_1"]["apology_pct"] = 140
        return {**base, "text": "Set 1 asks more questions.\n\n" + json.dumps(est)}
    letter = prompt.split("<email>", 1)[1].split("</email>", 1)[0].strip()
    ans = {"author": rng.choice(["human", "ai"]), "author_confidence": rng.randint(1, 5),
           "native_language": rng.choice(sorted(ATTR_ENUMS["native_language"])),
           "native_language_confidence": rng.randint(1, 5),
           "grew_up": rng.choice(sorted(ATTR_ENUMS["grew_up"])), "grew_up_confidence": rng.randint(1, 5),
           "cues": [{"quote": letter.split()[0], "suggests": "mock"}]}
    if MOCK_FAIL == "invalid":
        ans["author_confidence"] = 9
    return {**base, "text": json.dumps(ans, ensure_ascii=False)}


def load_key(ctx, lids):
    path = KEY[ctx]
    if not path.exists():
        sys.exit(f"No redaction key at {path}.")
    payload = path.read_text(encoding="utf-8")
    group = {r["letter_id"]: r["group"] for r in csv.DictReader(payload.splitlines())}
    missing = sorted(set(lids) - set(group))
    if missing:
        sys.exit("Letters missing from the key: " + ", ".join(missing))
    return group, sha256(payload)


def build_inputs(task, letters, group, smoke_letters, smoke):
    """Each input: (input_id, info). Attribution: one letter. Compare: two sets."""
    if task == "attribute":
        lids = sorted(letters)[:smoke_letters] if smoke else sorted(letters)
        return {lid: {"letter_ids": [lid]} for lid in lids}
    inputs = {}
    for a, b in COMPARISONS:
        ga = sorted(l for l in letters if group[l] == a)
        gb = sorted(l for l in letters if group[l] == b)
        if not ga or not gb:
            sys.exit(f"No letters for comparison {a} vs {b}.")
        inputs[f"{a}__vs__{b}"] = {"groups": [a, b], "letter_ids": {a: ga, b: gb}}
        if smoke:
            break
    return inputs


def render(task, template, info, letters, run, seed, input_id, judge):
    """The prompt and the per-call layout (which group is Set 1, letter order)."""
    if task == "attribute":
        lid = info["letter_ids"][0]
        prompt = template.replace("{LETTER}", letters[lid][0])
        assert "{LETTER}" not in prompt and letters[lid][0] in prompt
        return prompt, {}
    a, b = info["groups"]
    # Set order alternates across runs. The starting side is random per
    # comparison and alternates across judges, so each group is Set 1 in half
    # of all calls (10 of 20 with 4 judges x 5 runs).
    flip = (random.Random(f"{seed}{input_id}").random() < 0.5) != (JUDGES.index(judge) % 2 == 1)
    first, second = (a, b) if (run % 2 == 1) != flip else (b, a)
    rng = random.Random(f"{seed}{input_id}{run}")
    # The log keeps letter ids only; groups are joined from the key at analysis.
    # Judges see neutral labels 1..n per set; set_<k>_ids[i] is label i + 1.
    layout = {}
    blocks = {}
    for name, g in (("set_1", first), ("set_2", second)):
        ids = list(info["letter_ids"][g])
        rng.shuffle(ids)
        layout[f"{name}_ids"] = ids
        blocks[name] = "\n".join(f'<email id="{n}">\n{letters[lid][0]}\n</email>' for n, lid in enumerate(ids, 1))
    prompt = template.replace("{SET_1_EMAILS}", blocks["set_1"]).replace("{SET_2_EMAILS}", blocks["set_2"])
    assert "{SET_1_EMAILS}" not in prompt and "{SET_2_EMAILS}" not in prompt
    assert all(letters[l][0] in prompt for k in ("set_1_ids", "set_2_ids") for l in layout[k])
    return prompt, layout


def plan_cells(inputs, runs, seed, fill_from, current):
    if not fill_from:
        plan = [(j, i, r) for j in JUDGES for i in inputs for r in range(1, runs + 1)]
        random.Random(seed).shuffle(plan)
        return plan, {}, 1, runs
    recs = [json.loads(l) for l in open(fill_from, encoding="utf-8")]
    for key in current:
        if {r[key] for r in recs} != {current[key]}:
            sys.exit(f"Refusing to refill: {key} differs from the original run.")
    if {r["judge"] for r in recs} != set(JUDGES):
        sys.exit("Refusing to refill: the judge list changed since the original run.")
    planned = recs[0]["planned_runs"]
    grid = {(j, i, r) for j in JUDGES for i in inputs for r in range(1, planned + 1)}
    kept = {(r["judge"], r["input_id"], r["run"]) for r in recs if r["status"] != "error"}
    attempts = {}
    for r in recs:
        c = (r["judge"], r["input_id"], r["run"])
        attempts[c] = max(attempts.get(c, 0), r["attempt"])
    return sorted(grid - kept), attempts, max(r["call_index"] for r in recs) + 1, planned


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TEMPLATES), required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--real", action="store_true")
    ap.add_argument("--fill-missing", metavar="FILE")
    ap.add_argument("--runs", type=int, default=K_RUNS)
    ap.add_argument("--smoke-letters", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    task = args.task
    ctx = "real" if args.real else "synthetic"
    prov = provenance(__file__)
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    if args.real and prov["git_dirty"] is not False:
        sys.exit("Refusing a real run from uncommitted code or prompts. Commit first.")
    if args.fill_missing and not args.real and not Path(args.fill_missing).name.startswith("SYNTHETIC_"):
        sys.exit("--fill-missing on a synthetic run must name a SYNTHETIC_ file.")

    letters, _, signoff_hash = SEG.load_signed_letters(ctx)
    group, key_hash = load_key(ctx, letters)
    template = TEMPLATES[task].read_text(encoding="utf-8").strip()
    inputs = build_inputs(task, letters, group, args.smoke_letters, args.smoke)
    current = {"prompt_template_sha256": sha256(template), "redaction_signoff_sha256": signoff_hash,
               "key_sha256": key_hash, "script_sha256": prov["script_sha256"], "seed": args.seed}
    runs = 1 if args.smoke else args.runs
    plan, prior, first_index, runs = plan_cells(inputs, runs, args.seed, args.fill_missing, current)
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
            run_id = f"{RUN_NAME[task]}_{stamp}"
            out_path = OUT[ctx] / f"{PREFIX[ctx]}{kind}{run_id}.jsonl"
            if not out_path.exists():
                break
            time.sleep(0.2)
    versions = sdk_versions()
    caller = call_mock if args.dry_run else call_anthropic

    counts, cost = {}, {}
    with open(out_path, "a" if args.fill_missing else "x", encoding="utf-8", newline="\n") as out:
        for i, (judge, input_id, run) in enumerate(plan, first_index):
            info = inputs[input_id]
            prompt, layout = render(task, template, info, letters, run, args.seed, input_id, judge)
            lids = info["letter_ids"] if task == "attribute" else \
                [l for g in info["groups"] for l in info["letter_ids"][g]]
            ts = datetime.now(timezone.utc)
            t0 = time.time()
            rec = {"run_id": run_id, "task": task, "call_index": i, "seed": args.seed,
                   "mode": ("dry_run" if args.dry_run else "smoke" if args.smoke else
                            "fill_missing" if args.fill_missing else "real"),
                   "fill_missing": bool(args.fill_missing),
                   "attempt": prior.get((judge, input_id, run), 0) + 1,
                   "judge": judge, "model_requested": judge, "input_id": input_id, "run": run,
                   "planned_runs": runs,
                   "letter_ids": lids, **layout,
                   "letter_sha256": {l: letters[l][1] for l in lids},
                   "redaction_signoff_sha256": signoff_hash, "key_sha256": key_hash,
                   "timestamp_utc": ts.isoformat(), "date": ts.date().isoformat(),
                   "prompt_file": str(TEMPLATES[task].relative_to(ROOT.parent)),
                   "prompt_template_sha256": sha256(template), "prompt_sha256": sha256(prompt),
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
            answer, part_a, problem, cues_found = None, None, None, None
            if not rec["error"]:
                try:
                    if task == "attribute":
                        answer, problem, cues_found = parse_attribute(rec["text"], letters[lids[0]][0])
                    else:
                        answer, part_a, problem = parse_compare(rec["text"])
                except Exception as e:  # never lose the record of a paid call
                    answer, part_a, problem, cues_found = None, None, "parse_error", None
                    rec["parse_exception"] = f"{type(e).__name__}: {e}"
            rec["status"] = ("error" if rec["error"] else
                             rec["status"] if rec.get("status") in ("refusal", "truncated", "other_stop") else
                             problem or "ok")
            # Answers are kept only from a complete, valid reply.
            ok = rec["status"] == "ok"
            rec["answer"] = answer if ok else None
            rec["part_a"] = part_a if ok else None
            rec["cue_quotes_found"] = cues_found if ok else None
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
            print(f"[{i - first_index + 1}/{len(plan)}] {judge:18s} {input_id} r{run}  {rec['status']}{flag}")
            if not args.dry_run:
                time.sleep(args.sleep)

    print(f"\nstatus counts: {counts} -> {out_path}")
    if not args.dry_run:
        print("cost at list prices: " + ", ".join(f"{j} ${c:.4f}" for j, c in sorted(cost.items()))
              + f"  (total ${sum(cost.values()):.2f})")
    if args.smoke:
        n_real = len(list(SEG.PATHS["real"]["letters"].glob(SEG.PATHS["real"]["pattern"]))) or 22
        n_inputs = n_real if task == "attribute" else len(COMPARISONS)
        per_input = sum(cost.values()) / len(inputs)
        print(f"projected real run: {n_inputs} inputs x {len(JUDGES)} judges x {K_RUNS} runs "
              f"= {n_inputs * len(JUDGES) * K_RUNS} calls, about ${per_input * n_inputs * K_RUNS:.2f} "
              f"at list prices (synthetic letters are shorter than real ones; compare sets are smaller)")


if __name__ == "__main__":
    main()
