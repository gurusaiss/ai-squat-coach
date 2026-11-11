"""Turns deterministic Finding objects into consumer-readable explanations
and actionable feedback via an embedded LLM agent -- the agent writes
prose, it does not decide verdicts (see verdicts.py).

Provider order (per product decision: no paid-only API dependency):
  1. Grok (xAI), via GROK_API_KEY -- plain HTTPS call to xAI's
     OpenAI-compatible /v1/chat/completions endpoint, no SDK required.
  2. Gemini (Google), via GEMINI_API_KEY -- plain HTTPS call to
     generativelanguage.googleapis.com, no SDK required.
  3. Static, skill-authored feedback templates -- used if neither key is
     configured, both calls fail (quota/network/auth), or a response can't
     be parsed as the expected JSON.

Note on "free": xAI's API is a promotional credit grant for new accounts,
not a permanent free tier; Google's Gemini API has an ongoing (but
rate-limited and occasionally changing) free tier. Both are configured here
because that's what was asked for, but this is disclosed honestly in
BUILD_NOTES.md rather than presented as a guaranteed-free setup -- provider
pricing/limits change outside this app's control.

Whichever path answers, the response is parsed against the same expected
JSON shape and never allowed to change a verdict -- see SYSTEM_PROMPT.
"""
from __future__ import annotations

import json
import os
import re

import requests

from .skill_loader import criteria_by_id
from .verdicts import Finding, fmt_time

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GEMINI_API_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REQUEST_TIMEOUT_SEC = 60

SYSTEM_PROMPT = """You are a reporting layer for a barbell-squat video analysis tool.
You are given a list of findings that were ALREADY computed and verdicted by
deterministic code from a reference document (a squat-technique chapter).
Each finding already has: a verdict (meets / does_not_meet / cannot_assess),
the raw measurement(s) that produced it, and the exact supporting quote from
the reference document.

Your ONLY job is to write, for each finding, in order:
  - "explanation": one or two plain-language sentences a non-technical gym-goer
    could understand, stating what was measured and why it does or doesn't
    match the quoted standard. Reference the actual number(s) given. If the
    verdict is cannot_assess, explain why in terms of the given confidence
    note -- do not guess a verdict.
  - "feedback": one short, actionable sentence grounded in the quoted
    standard. For cannot_assess, say what would be needed to assess it
    (e.g. a clearer camera angle) instead of generic advice.

Rules:
  - NEVER invent a numeric threshold, rule, or standard that is not present
    in the finding's quote or measurement. If you want to explain *why* a
    standard matters, you may paraphrase the supplied quote, but do not add
    new exercise-science claims from general knowledge.
  - NEVER change a verdict. Explanations must agree with the given verdict.
  - Keep explanation+feedback grounded strictly in the data given to you for
    THAT finding; do not average across reps inside a single finding's text.
  - Output must be valid JSON only, no markdown fences, no commentary outside
    the JSON.

After the per-finding items, add one "overall_summary" (3-5 sentences) that
synthesizes patterns ACROSS the given findings only (e.g. "every rep fell
short of depth" or "back angle was consistent across all 5 reps") and one
"confidence_caveats" (a short paragraph listing what could not be assessed
and why, in plain language).

Return exactly this JSON shape:
{
  "findings": [
    {"key": "<criterion_id>|<rep_index>", "explanation": "...", "feedback": "..."}
  ],
  "overall_summary": "...",
  "confidence_caveats": "..."
}
"""


def _finding_payload(f: Finding) -> dict:
    return {
        "key": f"{f.criterion_id}|{f.rep_index}",
        "criterion_id": f.criterion_id,
        "rep_index": f.rep_index,
        "verdict": f.verdict,
        "variant": f.variant,
        "timestamp": fmt_time(f.timestamp_sec),
        "frame": f.frame,
        "measurement": f.measurement,
        "confidence_note": f.confidence_note,
        "document_ref_pages": f.ref_pages,
        "document_sections": f.sections,
        "document_quote": f.quote,
    }


def _call_grok(user_payload: str) -> str:
    api_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("GROK_API_KEY not set")
    model = os.environ.get("GROK_MODEL", "grok-4-fast")
    resp = requests.post(
        GROK_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        timeout=REQUEST_TIMEOUT_SEC,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Grok API HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_gemini(user_payload: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = GEMINI_API_URL_TMPL.format(model=model)
    resp = requests.post(
        url,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_payload}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        },
        timeout=REQUEST_TIMEOUT_SEC,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def _template_fallback(findings: list[Finding]) -> dict:
    crit = criteria_by_id()
    out = {"findings": [], "overall_summary": None, "confidence_caveats": None}
    for f in findings:
        c = crit[f.criterion_id]
        fb = c.get("feedback", {})
        if f.verdict == "cannot_assess":
            text = fb.get("cannot_assess_default") or f"Cannot assess: {f.confidence_note}"
        elif f.variant and fb.get(f"{f.verdict}_{f.variant}"):
            text = fb[f"{f.verdict}_{f.variant}"]
        elif fb.get(f.verdict):
            text = fb[f.verdict]
        else:
            text = f"{f.criterion_id}: {f.verdict}"
        try:
            text = text.format(value=list(f.measurement.values())[0])
        except Exception:
            pass
        out["findings"].append({"key": f"{f.criterion_id}|{f.rep_index}", "explanation": text, "feedback": text})
    out["overall_summary"] = "Template-based summary (LLM write-up unavailable this run): see individual findings below."
    out["confidence_caveats"] = "Some findings used template fallback text rather than the LLM agent; see BUILD_NOTES.md."
    return out


def generate_writeups(findings: list[Finding]) -> tuple[dict, str]:
    """Returns (writeups_dict, method) where method in {'grok','gemini','template_fallback'}."""
    payload = json.dumps([_finding_payload(f) for f in findings], indent=2)

    try:
        raw = _call_grok(payload)
        return _extract_json(raw), "grok"
    except Exception as e:
        print(f"[assess] Grok path failed, trying Gemini: {e}")

    try:
        raw = _call_gemini(payload)
        return _extract_json(raw), "gemini"
    except Exception as e:
        print(f"[assess] Gemini path failed, falling back to templates: {e}")

    return _template_fallback(findings), "template_fallback"
