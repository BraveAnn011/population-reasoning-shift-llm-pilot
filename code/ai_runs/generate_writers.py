#!/usr/bin/env python3
"""Generate AI writer samples for the Email 1 (catch-up) pilot.

Design (Study Instruments v1, 2026-10-26, Email 1 only):
  * Prompts are read verbatim from ai_runs/prompts/email1/*.txt; one file per
    instrument (AI-US-EN, AI-JP-EN, AI-EN, AI-JA). The prompt hash is the
    SHA-256 of the exact string sent (the file text with outer whitespace
    stripped).
  * Every sample is a fresh, single-turn request: one user message, no system
    prompt, no conversation history, a new client per call.
  * No sampling or effort parameters are sent. Each provider's defaults apply
    and are logged as "provider_default (not sent)".
  * Every call is logged to JSONL, including failures, with the same keys on
    every line: request/response timestamps, model id requested and returned,
    prompt hash, request id, stop reason, status, usage, SDK versions, the
    full raw response, and the reply text.
  * Automatic refusal fallbacks are NOT enabled: a fallback would silently
    swap the writer model. SDK auto-retries are off (max_retries=0) so every
    attempt is visible in the log.

Status of a record:
  ok         normal stop, non-empty text
  refusal    the model or provider refused
  blocked    the provider blocked the prompt or the output
  truncated  stopped on a token limit
  other_stop any other stop reason (e.g. pause_turn); kept, not replaced
  empty      normal stop but no text
  error      the API call failed (network, 4xx/5xx) or the reply could not be read

Replacement rule (fixed before any real run): only cells whose attempts are all
"error" are re-run with --fill-missing. Refusal, blocked, truncated and empty
are outcomes and are kept as data, never replaced.

Usage (from the repository root):
    python3 ai_runs/generate_writers.py --dry-run         # mock, no API calls
    python3 ai_runs/generate_writers.py --list-models     # pin ids first
    python3 ai_runs/generate_writers.py --smoke           # real APIs, 1 session each -> dryrun/SYNTHETIC_smoke_*
    python3 ai_runs/generate_writers.py --sessions 5      # real run -> raw/
    python3 ai_runs/generate_writers.py --fill-missing ai_runs/raw/<file>.jsonl

Keys are read from environment variables only: ANTHROPIC_API_KEY,
OPENAI_API_KEY, GEMINI_API_KEY. Keys are never written to the log.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts" / "email1"
RAW_DIR = ROOT / "raw"
DRYRUN_DIR = ROOT / "dryrun"

INSTRUMENTS = ["AI-US-EN", "AI-JP-EN", "AI-EN", "AI-JA"]
LABELS = {"AI-US-EN": "Email 1:", "AI-JP-EN": "Email 1:", "AI-EN": "Email 1:", "AI-JA": "メール3:"}

# Writer models, pinned from --list-models on 2026-10-27. The Email 1 pilot
# uses one writer: the most widely used Claude model. It is also a judge
# (judges: claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5,
# claude-fable-5-1); its judgments of its own letters are the self-recognition
# check. GPT and Gemini writers join the full study.
WRITERS = {
    "sonnet55": {"provider": "anthropic", "model": "claude-sonnet-5-5"},
}

# Anthropic requires max_tokens. Set high enough never to cut a 5-10 sentence
# email (thinking tokens count against it on Fable). Other providers: not sent.
ANTHROPIC_MAX_TOKENS = 16000

NOT_SENT = "provider_default (not sent)"

# Every record carries exactly these keys, in this order.
RESULT_KEYS = ["model_returned", "request_id", "response_id", "stop_reason",
               "stop_details", "status", "usage", "content_block_types",
               "refusal_text", "block_info", "text", "label_ok", "body",
               "error", "raw_response"]


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompts():
    prompts = {}
    for name in INSTRUMENTS:
        path = PROMPT_DIR / f"{name}.txt"
        text = path.read_text(encoding="utf-8").strip()
        prompts[name] = {"text": text, "sha256": sha256(text),
                         "file": str(path.relative_to(ROOT.parent))}
    return prompts


def provenance(script=None):
    """SHA-256 of the running script and the git commit it ran from (with a
    dirty flag for ai_runs/ code and prompts)."""
    import subprocess
    script = Path(script or __file__)
    script_sha = sha256(script.read_text(encoding="utf-8"))
    def git(*a):
        return safe(lambda: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True,
                                           text=True, check=True).stdout.strip())
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "ai_runs/*.py", "ai_runs/prompts")
    return {"script": script.name, "script_sha256": script_sha, "git_commit": commit,
            "git_dirty": bool(dirty) if dirty is not None else None}


def sdk_versions():
    v = {"python": platform.python_version()}
    for mod, key in (("anthropic", "anthropic"), ("openai", "openai"), ("google.genai", "google-genai")):
        try:
            m = __import__(mod, fromlist=["__version__"])
            v[key] = getattr(m, "__version__", "unknown")
        except ImportError:
            v[key] = None
    return v


def safe(fn, default=None):
    """Read one field without letting a parsing error lose the record."""
    try:
        return fn()
    except Exception:
        return default


def dump(obj):
    return safe(lambda: obj.model_dump(mode="json"), default=safe(lambda: str(obj)))


# ---------------------------------------------------------------------------
# Provider calls. Each returns a partial result dict. The raw response is
# captured first, before any parsing.
# ---------------------------------------------------------------------------

def call_anthropic(model, prompt):
    import anthropic
    client = anthropic.Anthropic(max_retries=0)
    msg = client.messages.create(
        model=model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    r = {"raw_response": dump(msg)}
    blocks = safe(lambda: list(msg.content), [])
    r["model_returned"] = safe(lambda: msg.model)
    r["request_id"] = safe(lambda: msg._request_id)
    r["response_id"] = safe(lambda: msg.id)
    r["stop_reason"] = safe(lambda: msg.stop_reason)
    r["stop_details"] = safe(lambda: dump(msg.stop_details) if msg.stop_details else None)
    r["usage"] = safe(lambda: dump(msg.usage))
    r["content_block_types"] = [safe(lambda b=b: b.type) for b in blocks]
    r["text"] = "".join(safe(lambda b=b: b.text, "") for b in blocks
                        if safe(lambda b=b: b.type) == "text")
    sr = r["stop_reason"]
    r["status"] = ("refusal" if sr == "refusal" else
                   "truncated" if sr == "max_tokens" else
                   None if sr in ("end_turn", "stop_sequence") else "other_stop")
    return r


def call_openai(model, prompt):
    from openai import OpenAI
    client = OpenAI(max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    r = {"raw_response": dump(resp)}
    choice = safe(lambda: resp.choices[0])
    r["model_returned"] = safe(lambda: resp.model)
    r["request_id"] = safe(lambda: resp._request_id)
    r["response_id"] = safe(lambda: resp.id)
    r["stop_reason"] = safe(lambda: choice.finish_reason)
    r["usage"] = safe(lambda: dump(resp.usage) if resp.usage else None)
    r["refusal_text"] = safe(lambda: choice.message.refusal)
    r["text"] = safe(lambda: choice.message.content, None) or ""
    sr = r["stop_reason"]
    r["status"] = ("refusal" if r["refusal_text"] else
                   "blocked" if sr == "content_filter" else
                   "truncated" if sr == "length" else
                   "error" if choice is None else None)
    return r


def call_google(model, prompt):
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(model=model, contents=prompt)
    r = {"raw_response": dump(resp)}
    cand = safe(lambda: resp.candidates[0] if resp.candidates else None)
    r["model_returned"] = safe(lambda: resp.model_version)
    r["response_id"] = safe(lambda: resp.response_id)
    r["stop_reason"] = safe(lambda: str(cand.finish_reason.name if hasattr(cand.finish_reason, "name")
                                        else cand.finish_reason))
    r["usage"] = safe(lambda: dump(resp.usage_metadata) if resp.usage_metadata else None)
    r["block_info"] = {
        "prompt_block_reason": safe(lambda: str(resp.prompt_feedback.block_reason)
                                    if resp.prompt_feedback and resp.prompt_feedback.block_reason else None),
        "safety_ratings": safe(lambda: [dump(s) for s in cand.safety_ratings] if cand.safety_ratings else None),
    }
    # Build text from parts, skipping thought summaries, instead of resp.text.
    parts = safe(lambda: list(cand.content.parts or []), []) if cand else []
    r["content_block_types"] = ["thought" if safe(lambda p=p: p.thought) else "text" for p in parts]
    r["text"] = "".join(safe(lambda p=p: p.text, "") or "" for p in parts if not safe(lambda p=p: p.thought))
    sr = (r["stop_reason"] or "").upper()
    r["status"] = ("blocked" if cand is None or r["block_info"]["prompt_block_reason"] else
                   "blocked" if sr in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII") else
                   "truncated" if sr == "MAX_TOKENS" else None)
    return r


def call_mock(model, prompt):
    time.sleep(0.01)
    label = "メール3:" if "メール3" in prompt else "Email 1:"
    return {"model_returned": f"{model}-MOCK", "request_id": "req_SYNTHETIC",
            "response_id": "msg_SYNTHETIC", "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "content_block_types": ["text"],
            "text": f"{label}\nSYNTHETIC mock email. Hi [name], it has been a while.",
            "raw_response": {"mock": True}}


CALLERS = {"anthropic": call_anthropic, "openai": call_openai, "google": call_google}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GEMINI_API_KEY"}


def check_label(text, instrument):
    """label_ok: the reply starts with the required label (after whitespace or
    markdown bold) and has no other text before it. body: text after the label."""
    label = LABELS[instrument]
    m = re.match(r"\s*\**\s*" + re.escape(label.rstrip(":")) + r"\s*[:：]\s*\**", text or "")
    if not m:
        return False, None
    return True, text[m.end():].strip()


def run_call(spec, prompt_text, instrument, dry_run):
    t0 = time.time()
    ts_req = datetime.now(timezone.utc)
    result = {k: None for k in RESULT_KEYS}
    try:
        caller = call_mock if dry_run else CALLERS[spec["provider"]]
        result.update(caller(spec["model"], prompt_text))
    except Exception as e:  # failed calls are logged, never dropped
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["request_id"] = result["request_id"] or safe(lambda: e.request_id)
    if result["status"] is None:
        result["status"] = "ok" if (result["text"] or "").strip() else "empty"
    result["label_ok"], result["body"] = check_label(result["text"], instrument)
    ts_resp = datetime.now(timezone.utc)
    return ts_req, ts_resp, round(time.time() - t0, 3), result


def list_models():
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        print("=== anthropic ===")
        for m in anthropic.Anthropic().models.list():
            print(" ", m.id)
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        print("=== openai ===")
        for m in sorted(OpenAI().models.list(), key=lambda m: m.id):
            print(" ", m.id)
    if os.environ.get("GEMINI_API_KEY"):
        from google import genai
        print("=== google ===")
        for m in genai.Client(api_key=os.environ["GEMINI_API_KEY"]).models.list():
            print(" ", m.name)
    print("\nPin the chosen ids in WRITERS at the top of this script, and save this output in the run notes.")


def build_plan(writers, sessions, seed, fill_from, prompts):
    """Return (plan, attempt_of, next_call_index). Plan items are (writer, instrument, session).
    With fill_from, the full planned grid is rebuilt so cells never attempted
    (an interrupted run) are filled too; cells with any non-error record are kept."""
    if not fill_from:
        plan = [(w, i, s) for w in writers for i in INSTRUMENTS for s in range(1, sessions + 1)]
        random.Random(seed).shuffle(plan)
        return plan, {}, 1
    recs = [json.loads(l) for l in open(fill_from, encoding="utf-8")]
    for r in recs:  # refuse to mix prompts or models within one run
        if r["prompt_sha256"] != prompts[r["instrument"]]["sha256"]:
            sys.exit(f"Prompt {r['instrument']} changed since this run; refusing to fill.")
        if r["writer"] in WRITERS and r["model_requested"] != WRITERS[r["writer"]]["model"]:
            sys.exit(f"Model for '{r['writer']}' changed since this run; refusing to fill.")
    planned = max(r.get("planned_sessions", 0) for r in recs) or max(r["session"] for r in recs)
    grid = {(w, i, s) for w in {r["writer"] for r in recs} if w in writers
            for i in INSTRUMENTS for s in range(1, planned + 1)}
    kept, attempts = set(), {}
    for r in recs:
        c = (r["writer"], r["instrument"], r["session"])
        attempts[c] = max(attempts.get(c, 0), r.get("attempt", 1))
        if r["status"] != "error":
            kept.add(c)
    plan = sorted(grid - kept)
    return plan, attempts, max(r["call_index"] for r in recs) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=5, help="sessions per model per prompt")
    ap.add_argument("--writers", default=",".join(WRITERS), help="comma-separated writer keys")
    ap.add_argument("--dry-run", action="store_true",
                    help="mock calls -> dryrun/SYNTHETIC_* (with --fill-missing: mock calls only)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="real API calls, 1 session per model per prompt -> dryrun/SYNTHETIC_smoke_*")
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--fill-missing", metavar="FILE",
                      help="re-run only cells whose attempts are all errors; appends to FILE")
    ap.add_argument("--seed", type=int, default=20261027, help="seed for the call order")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls")
    args = ap.parse_args()

    if args.list_models:
        list_models()
        return

    sessions = 1 if args.smoke else args.sessions
    writers = [w.strip() for w in args.writers.split(",")]
    for w in writers:
        if w not in WRITERS:
            sys.exit(f"Unknown writer: {w}")
        spec = WRITERS[w]
        if not args.dry_run:
            if spec["model"].startswith("PIN_"):
                sys.exit(f"Pin a model id for '{w}' in WRITERS first (see --list-models).")
            if not os.environ.get(KEY_ENV[spec["provider"]]):
                sys.exit(f"{KEY_ENV[spec['provider']]} is not set.")

    if args.fill_missing and args.dry_run:
        fp = Path(args.fill_missing).resolve()
        if fp.parent != DRYRUN_DIR or not fp.name.startswith("SYNTHETIC_"):
            sys.exit("--dry-run with --fill-missing is allowed only on ai_runs/dryrun/SYNTHETIC_* files.")

    prompts = load_prompts()
    plan, prior_attempts, first_index = build_plan(writers, sessions, args.seed,
                                                   args.fill_missing, prompts)
    if not plan:
        print("Nothing to run.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.fill_missing:
        out_path = Path(args.fill_missing)
        run_id = json.loads(open(out_path, encoding="utf-8").readline())["run_id"]
        mode_open = "a"
    else:
        run_id = f"email1_{stamp}"
        if args.dry_run or args.smoke:
            DRYRUN_DIR.mkdir(exist_ok=True)
            prefix = "SYNTHETIC_smoke_" if args.smoke else "SYNTHETIC_"
            out_path = DRYRUN_DIR / f"{prefix}{run_id}.jsonl"
        else:
            RAW_DIR.mkdir(exist_ok=True)
            out_path = RAW_DIR / f"{run_id}.jsonl"
        mode_open = "w"

    versions = sdk_versions()
    prov = provenance()
    counts = {}
    with open(out_path, mode_open, encoding="utf-8") as out:
        for i, (w, inst, s) in enumerate(plan, first_index):
            spec, p = WRITERS[w], prompts[inst]
            ts_req, ts_resp, latency, result = run_call(spec, p["text"], inst, args.dry_run)
            attempt = prior_attempts.get((w, inst, s), 0) + 1
            record = {
                "run_id": run_id, "call_index": i, "seed": args.seed,
                "mode": "dry_run" if args.dry_run else "smoke" if args.smoke else
                        "fill_missing" if args.fill_missing else "real",
                "attempt": attempt, "retry_of": f"{w}/{inst}/s{s}" if attempt > 1 else None,
                "timestamp_utc": ts_req.isoformat(), "response_timestamp_utc": ts_resp.isoformat(),
                "latency_s": latency, "date": ts_req.date().isoformat(),
                "writer": w, "provider": spec["provider"], "model_requested": spec["model"],
                "instrument": inst, "session": s,
                "planned_sessions": sessions if not args.fill_missing else None,
                "prompt_file": p["file"], "prompt_sha256": p["sha256"], "prompt_text": p["text"],
                "system_prompt": None,
                "temperature": NOT_SENT, "top_p": NOT_SENT, "effort": NOT_SENT,
                "max_tokens": ANTHROPIC_MAX_TOKENS if spec["provider"] == "anthropic" else NOT_SENT,
                "sdk_max_retries": 0, "sdk_versions": versions, **prov,
                **result,
            }
            record["model_match"] = record["model_returned"] == spec["model"] or (
                args.dry_run and record["model_returned"] == f"{spec['model']}-MOCK")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            flag = "" if result["label_ok"] else "  label MISSING"
            flag += "" if record["model_match"] else f"  MODEL MISMATCH ({record['model_returned']})"
            print(f"[{i - first_index + 1}/{len(plan)}] {w:6s} {inst:9s} s{s} a{attempt}  {result['status']}{flag}")
            if not args.dry_run:
                time.sleep(args.sleep)

    not_ok = sum(v for k, v in counts.items() if k != "ok")
    print(f"\nstatus counts: {counts}  (not ok: {not_ok}) -> {out_path}")


if __name__ == "__main__":
    main()
