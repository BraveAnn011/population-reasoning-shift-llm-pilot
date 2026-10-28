#!/usr/bin/env python3
"""Stage A by AI: split redacted letters into move units for human checking.

Pilot procedure (a recorded deviation from coding_manual.md §3; see §3.1):
one segmenter model splits each redacted letter once; the researcher corrects
the result in a CHECK file; --freeze re-checks the corrected units against the
signed-off letters and writes a versioned frozen segmentation that every judge
uses.

Two contexts share every code path:
  synthetic  ai_runs/dryrun/SYNTHETIC_letters/SYNTHETIC_R*.txt -> ai_runs/dryrun/
  real       data_private/pilot_email1/redaction/letters/R*.txt -> data_private/...

Real letters are sent only with --real, only after the researcher signs off
redaction interactively (--sign-off, typed confirmation, a redaction-log row
for every letter), only if each letter's hash still matches the sign-off, only
if a leak scan is clean, and only from committed code and prompts.
Agents must never run --sign-off for real letters.

Usage (from the repository root):
    python3 ai_runs/segment_letters.py --dry-run --sign-off TEST      # synthetic sign-off
    python3 ai_runs/segment_letters.py --dry-run                      # synthetic, mock model
    python3 ai_runs/segment_letters.py --smoke                        # synthetic, real API
    python3 ai_runs/segment_letters.py --dry-run --freeze FILE [FILE ...]

    python3 ai_runs/segment_letters.py --sign-off "Your Name"         # researcher only
    python3 ai_runs/segment_letters.py --real [--only R03,R11]
    python3 ai_runs/segment_letters.py --freeze FILE [FILE ...]
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from generate_writers import (ANTHROPIC_MAX_TOKENS, NOT_SENT, ROOT, call_anthropic,
                              provenance, sdk_versions, sha256)

PROMPT_FILE = ROOT / "prompts" / "judge" / "task0_segment.txt"
SEGMENTER = "claude-opus-5-5"
NEWLINE_MARK = "⏎"  # ⏎ shows a line break inside a unit in the CHECK file
OK_STATUSES = ("ok", "corrected")

PATHS = {
    "synthetic": {
        "letters": ROOT / "dryrun" / "SYNTHETIC_letters",
        "pattern": "SYNTHETIC_R*.txt",
        "signoff": ROOT / "dryrun" / "SYNTHETIC_REDACTION_DONE.json",
        "redaction_log": ROOT / "dryrun" / "SYNTHETIC_redaction_log.csv",
        "out": ROOT / "dryrun",
        "prefix": "SYNTHETIC_",
    },
    "real": {
        "letters": ROOT.parent / "data_private" / "pilot_email1" / "redaction" / "letters",
        "pattern": "R[0-9][0-9].txt",
        "signoff": ROOT.parent / "data_private" / "pilot_email1" / "redaction" / "REDACTION_DONE.json",
        "redaction_log": ROOT.parent / "data_private" / "pilot_email1" / "redaction" / "redaction_log.csv",
        "out": ROOT.parent / "data_private" / "pilot_email1" / "segmentation",
        "prefix": "",
    },
}

LEAK_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "handle": r"(?<![\w.])@\w{2,}",
    "url": r"https?://|www\.",
    "phone": r"(?<![\d-])\+?\d[\d\s().-]{8,}\d(?![\d-])",
    "postal_code_jp": r"〒\s?\d{3}-?\d{4}|(?<!\d)\d{3}-\d{4}(?!\d)",
    "line_id": r"line\s?id",
    "device_signature": r"sent from (my|outlook|yahoo|mail)|get outlook for|iphoneから送信|から送信$",
}
DATE_LIKE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$")


# ---------------------------------------------------------------------------
# Letters, leak scan, sign-off
# ---------------------------------------------------------------------------

def read_letter(path):
    """Return (text sent to the model, SHA-256 of the file bytes). Line endings
    are normalised to \\n and a BOM is removed; the hash is of the raw file."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").strip()
    if NEWLINE_MARK in text:
        sys.exit(f"{path.name} contains the reserved character {NEWLINE_MARK}; replace it first.")
    return text, hashlib.sha256(raw).hexdigest()


def letter_files(ctx):
    p = PATHS[ctx]
    files = sorted(p["letters"].glob(p["pattern"]))
    if not files:
        sys.exit(f"No letters in {p['letters']}.")
    return {f.stem: f for f in files}


def scan_leaks(letters):
    hits = []
    for lid, (text, _) in letters.items():
        for name, pat in LEAK_PATTERNS.items():
            for m in re.finditer(pat, text, re.I | re.M):
                if name == "phone" and DATE_LIKE.match(m.group().strip()):
                    continue
                hits.append(f"{lid}: {name}")
                break
    return hits


def redaction_log_ids(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return {row["letter_id"].strip() for row in csv.DictReader(f) if row.get("letter_id")}


def sign_off(ctx, name):
    p = PATHS[ctx]
    if ctx == "real":
        if not sys.stdin.isatty():
            sys.exit("Sign-off must be typed by the researcher in an interactive terminal.")
    letters = {lid: read_letter(f) for lid, f in letter_files(ctx).items()}
    hits = scan_leaks(letters)
    if hits:
        sys.exit("Not signed off. Possible identifiers remain:\n  " + "\n  ".join(hits))
    logged = redaction_log_ids(p["redaction_log"])
    if logged is None:
        sys.exit(f"No redaction log at {p['redaction_log']}.")
    missing = sorted(set(letters) - logged)
    if missing:
        sys.exit("Every letter needs at least one row in the redaction log "
                 "(use placeholder 'none', rule 'reviewed_no_change' if nothing was changed). Missing: "
                 + ", ".join(missing))
    if ctx == "real":
        phrase = f"I REDACTED {len(letters)} LETTERS"
        typed = input(f'Type exactly "{phrase}" to sign off: ').strip()
        if typed != phrase:
            sys.exit("Not signed off.")
    record = {"context": ctx, "signed_by": name,
              "signed_at_utc": datetime.now(timezone.utc).isoformat(),
              "redaction_log_sha256": sha256(p["redaction_log"].read_text(encoding="utf-8")),
              "letters": {lid: h for lid, (_, h) in letters.items()}}
    p["signoff"].write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"Signed off {len(letters)} letters -> {p['signoff']}")


def load_signed_letters(ctx, only=None):
    p = PATHS[ctx]
    if not p["signoff"].exists():
        sys.exit(f"Redaction is not signed off ({p['signoff'].name} missing).")
    signoff_text = p["signoff"].read_text(encoding="utf-8")
    signoff = json.loads(signoff_text)
    if signoff.get("context") != ctx:
        sys.exit("The sign-off file is for a different context.")
    if not p["redaction_log"].exists() or \
            sha256(p["redaction_log"].read_text(encoding="utf-8")) != signoff.get("redaction_log_sha256"):
        sys.exit("The redaction log changed after sign-off. Re-check and sign off again.")
    files = letter_files(ctx)
    if set(files) != set(signoff["letters"]):
        sys.exit("Letter files do not match the sign-off list. Re-check and sign off again.")
    if only is not None:
        unknown = sorted(only - set(files))
        if unknown:
            sys.exit("Unknown letter ids in --only: " + ", ".join(unknown))
        if not only:
            sys.exit("--only selected no letters.")
    letters = {lid: read_letter(f) for lid, f in files.items() if only is None or lid in only}
    for lid, (_, h) in letters.items():
        if h != signoff["letters"][lid]:
            sys.exit(f"{lid} changed after sign-off. Re-check and sign off again.")
    hits = scan_leaks(letters)
    if hits:
        sys.exit("Refusing: possible identifiers remain:\n  " + "\n  ".join(hits))
    return letters, signoff, sha256(signoff_text)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def check_coverage(letter, units):
    """Every unit is found in order, and the units cover all non-whitespace text."""
    pos, gaps = 0, []
    for n, u in enumerate(units, 1):
        if not isinstance(u, str) or not u.strip():
            return False, f"unit {n} is empty or not text"
        i = letter.find(u.strip(), pos)
        if i < 0:
            return False, f"unit {n} not found in order (after character {pos})"
        gaps.append(letter[pos:i])
        pos = i + len(u.strip())
    gaps.append(letter[pos:])
    rest = re.sub(r"\s", "", "".join(gaps))
    if rest:
        return False, f"{len(rest)} characters not covered by any unit"
    return True, None


def parse_units(text):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return None
    try:
        units = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(units, list):
        return None
    return [u.strip() if isinstance(u, str) else u for u in units]


MOCK_FAIL = os.environ.get("SEGMENT_MOCK_FAIL")  # tests only: bad_json, refusal, error, coverage


def call_mock(model, prompt):
    base = {"raw_response": {"mock": True}, "model_returned": f"{model}-MOCK",
            "request_id": "req_SYNTHETIC", "response_id": "msg_SYNTHETIC",
            "stop_reason": "end_turn", "status": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}
    if MOCK_FAIL == "error":
        raise RuntimeError("SYNTHETIC mock API error")
    if MOCK_FAIL == "bad_json":
        return {**base, "text": "Sure! Here are the units: not json"}
    if MOCK_FAIL == "refusal":
        return {**base, "stop_reason": "refusal", "status": "refusal", "text": ""}
    if MOCK_FAIL == "coverage":
        return {**base, "text": json.dumps(["Something not in the letter."])}
    letter = prompt.split("<email>\n", 1)[1].rsplit("\n</email>", 1)[0]
    units = [u for u in re.split(r"(?<=[.!?。！？])\s+|\n+", letter) if u.strip()]
    return {"raw_response": {"mock": True}, "model_returned": f"{model}-MOCK",
            "request_id": "req_SYNTHETIC", "response_id": "msg_SYNTHETIC",
            "stop_reason": "end_turn", "status": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "text": json.dumps(units, ensure_ascii=False)}


def status_of(rec, units, covered):
    if rec.get("error"):
        return "error"
    if rec.get("status") in ("refusal", "truncated", "other_stop"):
        return rec["status"]
    if units is None:
        return "no_json"
    return "ok" if covered else "coverage_fail"


def write_check(chk, lid, letter_hash, status, note, units, letter):
    chk.write(f"===== {lid} | sha256={letter_hash} | {status}" + (f" | {note}" if note else "") + " =====\n")
    for n, u in enumerate(units if status == "ok" else [letter], 1):
        chk.write(f"[{n}] {u.strip().replace(chr(10), NEWLINE_MARK)}\n")
    chk.write("\n")


def run_segmentation(ctx, args):
    p = PATHS[ctx]
    template = PROMPT_FILE.read_text(encoding="utf-8").strip()
    prov = provenance(__file__)
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    if ctx == "real" and prov["git_dirty"] is not False:
        sys.exit("Refusing a real run from uncommitted code or prompts. Commit first.")
    only = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    letters, _, signoff_hash = load_signed_letters(ctx, only)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p["out"].mkdir(parents=True, exist_ok=True)
    kind = "segment_" if ctx == "real" else ("segment_" if args.dry_run else "smoke_segment_")
    log_path = p["out"] / f"{p['prefix']}{kind}{stamp}.jsonl"
    check_path = p["out"] / f"{p['prefix']}{kind}{stamp}_CHECK.txt"
    versions = sdk_versions()
    caller = call_mock if args.dry_run else call_anthropic

    counts = {}
    with open(log_path, "w", encoding="utf-8", newline="\n") as log, \
         open(check_path, "w", encoding="utf-8", newline="\n") as chk:
        chk.write("Check each letter's units, one unit per line.\n"
                  f"To fix: split or join lines and renumber. Do not change any text or {NEWLINE_MARK}.\n"
                  f"A line break inside one unit is written {NEWLINE_MARK}; when you join lines that were\n"
                  f"separated by a line break in the letter, put {NEWLINE_MARK} where the break was.\n"
                  "Any letter you change, or any letter not marked ok, must have its status word in the\n"
                  "header changed to: corrected. Then run --freeze on this file (and any re-run files).\n\n")
        for i, (lid, (letter, file_hash)) in enumerate(letters.items(), 1):
            prompt = template.replace("{LETTER}", letter)
            ts = datetime.now(timezone.utc)
            t0 = time.time()
            rec = {"letter_id": lid, "letter_file_sha256": file_hash, "letter_sha256": sha256(letter),
                   "redaction_signoff_sha256": signoff_hash,
                   "timestamp_utc": ts.isoformat(), "date": ts.date().isoformat(),
                   "model_requested": SEGMENTER,
                   "prompt_file": str(PROMPT_FILE.relative_to(ROOT.parent)),
                   "prompt_template_sha256": sha256(template), "prompt_sha256": sha256(prompt),
                   "system_prompt": None, "temperature": NOT_SENT, "effort": NOT_SENT,
                   "max_tokens": ANTHROPIC_MAX_TOKENS, "sdk_max_retries": 0,
                   "sdk_versions": versions, **prov,
                   "mode": "dry_run" if args.dry_run else "smoke" if ctx == "synthetic" else "real",
                   "model_returned": None, "request_id": None, "response_id": None,
                   "stop_reason": None, "usage": None, "text": None, "raw_response": None,
                   "error": None}
            try:
                rec.update(caller(SEGMENTER, prompt))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["request_id"] = getattr(e, "request_id", None)
            rec["latency_s"] = round(time.time() - t0, 3)
            units = parse_units(rec["text"]) if not rec["error"] else None
            covered, why = check_coverage(letter, units) if units is not None else (False, None)
            rec["units"] = units
            rec["status"] = status_of(rec, units, covered)
            rec["coverage_note"] = None if rec["status"] == "ok" else (why or rec["status"])
            rec["model_match"] = rec["model_returned"] == (
                f"{SEGMENTER}-MOCK" if args.dry_run else SEGMENTER)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            write_check(chk, lid, file_hash, rec["status"], rec["coverage_note"], units or [], letter)
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            flag = "" if rec["model_match"] else f"  MODEL MISMATCH ({rec['model_returned']})"
            print(f"[{i}/{len(letters)}] {lid}  {rec['status']}  units={len(units) if covered else '-'}{flag}")
            if not args.dry_run:
                time.sleep(args.sleep)

    print(f"\nstatus counts: {counts}\nlog:   {log_path}\ncheck: {check_path}")


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

HEADER = re.compile(r"^===== (\S+) \| sha256=(\w+) \| (\w+)[^\n]*=====\n(.*?)(?=^===== |\Z)", re.S | re.M)


def freeze(ctx, check_paths):
    p = PATHS[ctx]
    if ctx == "real" and provenance(__file__)["git_dirty"] is not False:
        sys.exit("Refusing a real freeze from uncommitted code. Commit first.")
    letters, signoff, signoff_hash = load_signed_letters(ctx)
    candidates = {}
    for cp in check_paths:
        cp = Path(cp)
        text = cp.read_text(encoding="utf-8")
        # The log shares the run's name up to its timestamp, even if the
        # researcher saved the corrected CHECK file under a new name.
        m = re.match(r"(.*?_\d{8}T\d{6}Z)", cp.name)
        log_path = cp.with_name(m.group(1) + ".jsonl") if m else None
        log = {}
        if log_path is None or not log_path.exists():
            if ctx == "real":
                sys.exit(f"No segmentation log found for {cp.name}. Keep the run's name up to its timestamp.")
            print(f"warning: no segmentation log found for {cp.name}; AI units recorded as unknown")
            log_path = None
        log_name = log_path.name if log_path else None
        log_hash = sha256(log_path.read_text(encoding="utf-8")) if log_path else None
        if log_path:
            for line in open(log_path, encoding="utf-8"):
                r = json.loads(line)
                log[r["letter_id"]] = r
        for lid, h, status, body in HEADER.findall(text):
            units = [m.group(1).replace(NEWLINE_MARK, "\n").strip()
                     for m in re.finditer(r"^\[\d+\] (.*)$", body, re.M)]
            candidates.setdefault(lid, []).append(
                {"check_file": cp.name, "check_file_sha256": sha256(text), "header_sha256": h,
                 "status": status, "units": units, "log": log.get(lid),
                 "log_name": log_name, "log_sha256": log_hash})

    frozen, problems = {}, []
    unknown = sorted(set(candidates) - set(letters))
    if unknown:
        problems.append("letters not in the sign-off: " + ", ".join(unknown))
    for lid, (letter, file_hash) in letters.items():
        usable = [c for c in candidates.get(lid, []) if c["status"] in OK_STATUSES]
        if not usable:
            problems.append(f"{lid}: no block marked ok or corrected")
            continue
        if len(usable) > 1:
            problems.append(f"{lid}: {len(usable)} blocks marked ok/corrected; keep only one")
            continue
        c = usable[0]
        if c["header_sha256"] != file_hash:
            problems.append(f"{lid}: letter file changed since segmentation")
            continue
        if len(c["units"]) == 1 and c["status"] != "corrected":
            problems.append(f"{lid}: single unit; segment it or mark it corrected")
            continue
        ok, why = check_coverage(letter, c["units"])
        if not ok:
            problems.append(f"{lid}: {why}")
            continue
        log = c["log"] or {}
        if ctx == "real" and not log:
            problems.append(f"{lid}: not found in segmentation log {c['log_name']}")
            continue
        if log and (log.get("letter_file_sha256") != file_hash
                    or log.get("redaction_signoff_sha256") != signoff_hash):
            problems.append(f"{lid}: segmentation log does not match the current letter or sign-off")
            continue
        ai_units = log.get("units") if log.get("status") == "ok" else ([] if log else None)
        frozen[lid] = {
            "letter_sha256": file_hash, "units": c["units"],
            "status_in_check": c["status"], "check_file": c["check_file"],
            "check_file_sha256": c["check_file_sha256"],
            "ai_units": ai_units, "ai_status": log.get("status"),
            "units_changed_by_researcher": None if ai_units is None else ai_units != c["units"],
            "segmentation_log": c["log_name"], "segmentation_log_sha256": c["log_sha256"],
            "segmentation_request_id": log.get("request_id"),
            "segmentation_model_returned": log.get("model_returned"),
            "segmentation_model_match": log.get("model_match"),
            "segmentation_prompt_sha256": log.get("prompt_sha256"),
        }
    if problems:
        sys.exit("Not frozen. Fix these:\n  " + "\n  ".join(problems))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = {"context": ctx, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
           "redaction_signoff_sha256": signoff_hash, "freeze_provenance": provenance(__file__),
           "n_letters": len(frozen),
           "n_changed_by_researcher": sum(v["units_changed_by_researcher"] is True for v in frozen.values()),
           "n_change_unknown": sum(v["units_changed_by_researcher"] is None for v in frozen.values()),
           "letters": frozen}
    payload = json.dumps(out, ensure_ascii=False, indent=1)
    p["out"].mkdir(parents=True, exist_ok=True)
    path = p["out"] / f"{p['prefix']}FROZEN_segmentation_{stamp}.json"
    if path.exists():
        sys.exit(f"{path} already exists; not overwriting.")
    path.write_text(payload, encoding="utf-8")
    print(f"Frozen {len(frozen)} letters ({out['n_changed_by_researcher']} changed by researcher) -> {path}\n"
          f"sha256 {sha256(payload)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ctx_group = ap.add_mutually_exclusive_group()
    ctx_group.add_argument("--dry-run", action="store_true", help="synthetic letters, mock model")
    ctx_group.add_argument("--smoke", action="store_true", help="synthetic letters, real API")
    ctx_group.add_argument("--real", action="store_true", help="signed-off redacted letters, real API")
    ap.add_argument("--sign-off", metavar="NAME", help="record that redaction is finished")
    ap.add_argument("--freeze", metavar="CHECK_FILE", nargs="+", help="re-check corrections and freeze")
    ap.add_argument("--only", help="comma-separated letter ids to (re-)run")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    ctx = "synthetic" if (args.dry_run or args.smoke) else "real"
    if args.sign_off:
        return sign_off(ctx, args.sign_off)
    if args.freeze:
        return freeze(ctx, args.freeze)
    if not (args.dry_run or args.smoke or args.real):
        ap.error("choose --dry-run, --smoke or --real")
    run_segmentation(ctx, args)


if __name__ == "__main__":
    main()
