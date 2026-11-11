# Build Notes

## Architecture

```
Browser (upload/progress/results UI, plain HTML+JS)
   |
   v
FastAPI (backend/app/main.py)
   - POST /api/upload        -> saves file, spawns background thread, returns job_id
   - GET  /api/jobs/{id}     -> poll status/progress
   - GET  /api/jobs/{id}/video -> annotated mp4
   - GET  /api/skill          -> the raw skill yaml (transparency/inspectability)
   |
   v
Pipeline (backend/app/pipeline/run.py), per video:
   1. pose.py        MediaPipe Pose -> per-frame landmarks (hip/knee/ankle/shoulder/ear/wrist/foot)
   2. barbell.py      heuristic bar x-position (wrist proxy + Hough-line edge refinement)
   3. reps.py         rep count + bottom-position frames from smoothed hip-y trajectory
   4. measurements.py geometric measurements per rep (depth, back angle, knee offset, bar deviation, ascent timing)
   5. verdicts.py     DETERMINISTIC rule engine: skill yaml thresholds -> meets/does_not_meet/cannot_assess + evidence
   6. assess.py       LLM agent (Grok -> Gemini -> templates) turns verdicts+evidence into consumer prose; cannot change the verdict
   7. annotate.py     renders annotated video (skeleton, bar marker + traced path, rep/bottom overlays)
   |
   v
skill/squat_skill.yaml  <- single source of truth for every threshold/citation, loaded at runtime
```

## Why this split (model/tool choices)

- **MediaPipe Pose (legacy Solutions API, CPU, bundled model)** for
  landmarks: free, no network dependency at inference time, fast enough to
  process a 20-40s clip in well under a minute on CPU, and gives the
  specific joints (hip/knee/ankle/shoulder) the reference document's own
  diagnostic angles are defined on (Figure 2-5). We pinned
  `mediapipe==0.10.21` deliberately — the newest `mediapipe` on PyPI
  (1.2.2) ships a completely different Tasks API that requires downloading
  a `.task` model file from Google Cloud Storage at first run, which this
  sandboxed environment's network policy blocks. The legacy Solutions API
  bundles its model in the wheel, so it works fully offline.
- **No trained barbell detector.** Within the time box, training or
  sourcing a reliable barbell/plate detector was out of scope. We used a
  disclosed heuristic instead (wrist-midpoint x-position, refined by a
  Hough-line search for a horizontal edge near the shoulder line) and
  flagged every bar-path finding with its tracking-quality ratio
  (`N/M frames edge-refined`) rather than presenting it as verified
  tracking. This is the single biggest accuracy risk in the system — see
  Limitations.
- **Every numeric threshold, including implementation calibrations, lives in
  `skill/squat_skill.yaml`, not in Python.** `verdicts.py` and
  `measurements.py` read band widths, checkpoint fractions, and tolerance
  values from the loaded skill rather than hard-coding them, so a reviewer
  (or the demo's "change a rule in the source document" step) can edit the
  skill file and see the app's behavior change without touching code — this
  was tightened after an initial pass left a few bands as Python literals;
  caught during a self-review before handing this off for testing (see
  AI_USAGE.md).
- **Deterministic verdicts, LLM for prose only.** The assignment explicitly
  warns that "generic squat advice from a model is not enough" and that
  conclusions must follow the supplied standard. Letting an LLM look at a
  video and freely assess it risks exactly that: plausible-sounding but
  ungrounded verdicts. Instead, plain Python computes every verdict from
  numeric thresholds taken from (or explicitly calibrated against) the
  skill file, and the LLM only receives the already-decided verdict, the
  measurement, and the exact supporting quote, with instructions it is
  not permitted to change a verdict or introduce a new threshold. This
  also means the verdict logic is unit-testable without ever calling an
  LLM (`backend/tests/test_pipeline_logic.py`), which is both cheaper and
  far more reliable to verify than a model's output would be.
- **LLM call path: Grok (xAI) first, Gemini (Google) fallback, both via
  plain HTTPS, no vendor SDK.** This was an explicit product decision: use
  providers with a genuine no-/low-cost path to a working key rather than a
  metered-only API. `assess.py` calls `https://api.x.ai/v1/chat/completions`
  (OpenAI-compatible schema) with `GROK_API_KEY`; if that's unset or the
  call fails for any reason (quota, auth, network, bad JSON), it calls
  `https://generativelanguage.googleapis.com/.../generateContent` with
  `GEMINI_API_KEY`. If both fail, the app falls back to the skill's own
  static feedback templates rather than hanging or returning nothing, and
  marks the response accordingly (`writeup_method`: `grok` / `gemini` /
  `template_fallback` — always visible in the UI, never silently hidden).
  Model IDs (`grok-4-fast`, `gemini-2.5-flash` by default) are read from
  `GROK_MODEL`/`GEMINI_MODEL` env vars rather than hard-coded, because both
  vendors rename/retire model IDs frequently and this is exactly the kind
  of "unsupported claim" the system elsewhere refuses to make silently.
  **Honesty note on "free":** xAI's API key comes with a one-time
  promotional credit for new accounts, not a standing free tier; Google's
  Gemini API does have an ongoing free tier, though its exact rate limits
  have changed more than once. Both are still the right choice relative to
  a paid-only API, but "free" here means "has a no-cost path to a working
  key today," not "guaranteed free forever" — worth knowing before relying
  on this for a real product. This sandbox's network policy blocks
  `api.x.ai` outbound (confirmed via the proxy's own status endpoint), so
  the Grok call path was implemented to xAI's documented spec and verified
  for connectivity/shape against Gemini (which IS reachable here), but
  could not be live-exercised end-to-end against xAI from inside this
  sandbox — flagged honestly rather than claimed as tested. It will work
  normally run from a machine without that restriction.
- **Plain HTML/CSS/JS frontend, no framework.** Nothing in the UI needed
  client-side routing or complex state; a build step would have cost more
  time than it saved in a 4-6 hour box. FastAPI serves it directly as
  static files.
- **In-memory job queue (Python thread + dict), not Celery/Redis.** The
  assignment explicitly asks for "the smallest implementation that proves
  the complete experience." A real product would need a durable queue so
  jobs survive a restart and scale across workers; that is called out
  under Limitations rather than built, since the core thing to prove here
  — does the upload-to-feedback loop work and is it grounded — doesn't
  need it.

## UX decisions

Beyond the required upload → analysis → annotation → feedback flow, the
following were added specifically to make the result legible to a
non-technical user rather than just correct:

- **Stepper + live stage labels** during processing (Upload → Pose
  tracking → Barbell estimate → Rep detection → Measurements → AI write-up
  → Rendering), driven off the same progress messages the backend already
  emits, so the user always knows what's happening and roughly how long is
  left — not just a bare percentage.
- **Scorecard** at the top of the results (rep count, criteria met, below
  standard, cannot-assess counts) so the headline result is visible before
  reading any individual finding.
- **Click-to-seek findings**: clicking any finding card jumps the annotated
  video to that exact frame and plays from there, so "why did it say that"
  is always one click from "see it happen," instead of the user having to
  manually scrub a multi-rep video.
- **Bar path trail**, not just a per-frame dot: the annotated video now
  draws the accumulated, translucent path of the estimated bar position
  across each repetition, which is what the assignment specifically asks
  for ("track the barbell and show its path") — the first version only
  drew a live marker.
- **"About this analysis" panel**, generated from the same `/api/skill`
  endpoint the pipeline reads from, surfacing the camera assumptions and
  the insufficient-evidence policy in plain language, plus an explicit
  "no mocked steps" statement and which provider answered this run — so a
  skeptical reviewer doesn't have to read source code to find these
  guarantees.
- **Provider badge** always visible, naming whichever of Grok / Gemini /
  template-fallback produced the wording for that run.

## Time spent (approximate)

| Phase | Time |
|---|---|
| Reading assignment + reference document, planning | 0.5h |
| Skill extraction (`squat_skill.yaml`) | 0.75h |
| Pose/barbell/rep/measurement pipeline | 1.5h |
| Verdict engine + LLM write-up layer + fallbacks | 1h |
| Annotated video renderer | 0.5h |
| FastAPI backend + frontend | 1h |
| Testing (unit tests; end-to-end on sample/difficult video pending a supplied file) | 0.75h |
| Docs (README/BUILD_NOTES/AI_USAGE) | 0.5h |
| Self-review pass (calibration-threshold fix, fresh-venv reproducibility check) | 0.5h |
| Provider swap to Grok+Gemini (no paid-only API), bar-path trail, UX overhaul (stepper, scorecard, click-to-seek, about panel), provider-fallback unit tests | 1.5h |
| **Total** | **~9h** |

The initial working build (first 6 rows, ~6.5h) was done in one pass against
the stated 4-6 hour box. The remaining ~2.5h was a second pass driven by
explicit reviewer feedback after that first delivery (swap the LLM provider
to a specific pair, strengthen the UX, re-verify every assignment
requirement) — reported separately rather than folded in, so the timebox
comparison for the original build stays honest.

(This was built by an AI agent working through the assignment end-to-end
rather than a human candidate working solo; see AI_USAGE.md. Effort is
reported honestly against the stated 4-6 hour box rather than rounded down.)

## Cost per video (approximate)

- Pose tracking + CV: local CPU compute, effectively free per video (a
  ~30s clip at 30fps processes in well under a minute on a modern CPU).
- LLM write-up call: one call per video (not per finding), ~1-3K input
  tokens (all findings + quotes) and ~1-2K output tokens. At that size,
  this comfortably fits inside Gemini's free-tier per-minute/per-day quota
  for `gemini-2.5-flash` for demo-level usage, and inside xAI's
  promotional credit for a very large number of videos before it would run
  out. Neither provider is used for anything latency- or volume-sensitive
  (one call per video, not per finding, not per frame), which is precisely
  why a free-tier-friendly provider was workable here.

## Key trade-offs

1. **Accuracy vs. scope on bar tracking** — chose a disclosed heuristic
   over no bar-path criterion at all, and over spending the whole time box
   building/training a detector. Every bar-path finding carries its
   tracking-quality ratio so a reviewer can see how much to trust it.
2. **Determinism vs. flexibility** — hard-coding the verdict thresholds in
   Python (reading from the skill yaml) rather than asking the LLM to
   reason freely over the raw numbers means less "intelligence" per
   finding but a verifiable, unit-testable, reproducible result — which
   the evaluation criteria ("do conclusions follow the supplied standards"
   ) explicitly reward over a more flexible but less trustworthy design.
3. **Single side-view only, no multi-angle fusion** — correctly scoped
   against the assignment (side-view video is the stated input), but means
   several document criteria (stance width, knee valgus, bar back
   placement) are honestly reported as not assessable rather than guessed
   at from a view that can't support them.
4. **In-process job queue** — proves the real async upload/processing flow
   requested, at the cost of restart-safety and horizontal scaling, which
   were out of scope for a demo timebox.

## Known limitations

- Barbell x-position is an approximation (see above); confidence is
  reported per finding, but it is the weakest link in the system.
- The head/eye-gaze criterion is explicitly advisory-grade: MediaPipe's
  face landmarks from a typical side-view recording distance are not
  reliable enough to literally determine gaze direction, so this is
  reported as a coarse head-tilt proxy and frequently `cannot_assess`.
- Thresholds not literally stated in the document (back-angle tolerance
  band, knee-travel band, bar-path deviation band) are implementation
  calibrations, clearly labeled as such in `skill/squat_skill.yaml` and
  never presented as the document's own numbers.
- Single in-process job store: restarting the server loses in-flight job
  status (the uploaded file and any finished result files on disk are not
  lost, but the job record is).
- No authentication/multi-tenancy; this is a single-user demo prototype.
- Processing is capped at the first 90 seconds of a video to keep demo
  turnaround predictable; longer videos are truncated with a warning
  surfaced in the UI, not silently dropped.
