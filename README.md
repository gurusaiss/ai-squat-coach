# AI Squat Coach

A consumer-facing web app that accepts a side-view barbell squat video,
tracks the movement, and grades each repetition against the standards
extracted from a reference coaching document — not generic AI advice.

See `BUILD_NOTES.md` for architecture/trade-offs/cost and `AI_USAGE.md` for
how AI was used to build this.

> **Submission status:** the application itself is complete and unit-tested
> end to end (see `backend/tests`). What's still outstanding is running it
> against the assignment's common sample video and a second difficult/
> unsuitable video to capture the required evidence artifacts — that step
> needs an actual video file, which hasn't been provided yet, and is
> explicitly being done as a separate, final pass rather than claimed here
> as already done.

**Why local launch, not a hosted link:** this was built inside a sandboxed
cloud dev session with no persistent public-facing host attached to it, so
a shareable hosted URL isn't available from here. The app is a completely
standard FastAPI + static-frontend service, so hosting it (Render, Fly.io,
a VPS, etc.) is a non-event once you have infrastructure to point it at —
nothing about the design assumes local-only. `run.sh` gets it running
locally in under a minute as the fallback the assignment explicitly allows
for when "hosting is impractical."

## What it does

1. **Document → skill.** `skill/squat_skill.yaml` is a structured,
   citable extraction of the reference PDF (squat-technique chapter). Each
   criterion carries its source reference pages/section and a verbatim
   quote. The app reads this file at runtime — edit it and the
   application's behavior changes without touching code.
2. **Upload → track.** MediaPipe Pose extracts body landmarks per frame; a
   lightweight heuristic approximates barbell x-position; repetitions and
   their bottom positions are detected from the hip trajectory.
3. **Measure → verdict (deterministic).** Plain Python computes depth,
   back angle, knee travel, bar path, and hip-drive/ascent-consistency
   measurements per rep, and applies the skill's thresholds to produce a
   `meets` / `does_not_meet` / `cannot_assess` verdict per criterion per
   rep. This part is deterministic and unit-tested (`backend/tests`) — the
   document drives the verdict, not a model.
4. **Explain (LLM agent).** An embedded agent — Grok first, Gemini as
   fallback — is given only the already-computed verdicts, measurements,
   and the exact document quote per finding, and writes the consumer-facing
   explanation/feedback text. It cannot change a verdict or invent a
   threshold (see `backend/app/pipeline/assess.py`'s system prompt). If
   neither provider is reachable, the app falls back to static,
   skill-authored text rather than guessing.
5. **Annotate → serve.** An annotated video (skeleton, bar-position
   estimate **with its path traced over the rep**, rep/bottom markers, live
   back-angle/depth readout) is rendered and served back alongside the
   structured findings, with click-to-seek between a finding and the exact
   video moment it came from.

## Quick start (local)

Requires Python 3.11 (MediaPipe does not yet support 3.13) and `ffmpeg` on
PATH.

```bash
cd squat-coach
chmod +x run.sh
./run.sh
```

That script creates the venv, installs `backend/requirements.txt`, and
starts the server at `http://localhost:8000`. Or do it manually:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

**LLM credentials (optional, no paid-only dependency):**

The app calls two free-tier-friendly providers directly over HTTPS (no SDK
required), in this order, and falls back automatically:

1. **Grok (xAI)** — set `GROK_API_KEY` (get one at https://console.x.ai).
2. **Gemini (Google)** — set `GEMINI_API_KEY` (get one free at
   https://aistudio.google.com/apikey). Used automatically if Grok's key is
   missing or its call fails.
3. **Static templates** — if neither key is set or both calls fail (quota,
   network, auth), the app falls back to skill-authored template text
   rather than guessing. The UI always shows which of the three answered
   (`writeup_method`: `grok` / `gemini` / `template_fallback`).

```bash
export GROK_API_KEY=xai-...       # optional
export GEMINI_API_KEY=AIza...     # optional
```

Model IDs default to `grok-4-fast` and `gemini-2.5-flash` and can be
overridden with `GROK_MODEL` / `GEMINI_MODEL` if a provider renames or
retires one — see `BUILD_NOTES.md` for why these aren't hard-coded, and a
note on each provider's actual (not assumed) free-tier terms.

Open `http://localhost:8000`.

## Repository layout

```
skill/squat_skill.yaml       # document-derived rules the agent/app actually uses
backend/app/main.py          # FastAPI app: upload, job status, results, video serving
backend/app/jobs.py          # in-memory background job tracking
backend/app/pipeline/        # the actual CV + assessment pipeline
  pose.py                    # MediaPipe landmark extraction
  barbell.py                 # heuristic bar x-position estimate
  reps.py                    # rep/bottom-position detection
  measurements.py            # per-rep geometric measurements
  verdicts.py                # deterministic rule engine (reads skill yaml)
  assess.py                  # LLM write-up layer (Grok -> Gemini -> templates)
  annotate.py                # annotated video renderer
  run.py                     # orchestrates the full pipeline
backend/tests/                # unit tests for the deterministic logic
frontend/                     # plain HTML/CSS/JS single-page app
docs/evidence/                # test run evidence (see BUILD_NOTES.md)
```

## Known limitations (see BUILD_NOTES.md for full list)

- Barbell position is approximated (wrist position + edge refinement), not
  a verified object tracker — bar-path findings are explicitly lower
  confidence and say so.
- Single in-process job queue; does not survive a server restart or run
  multiple workers.
- Eye-gaze/head-position criterion is explicitly advisory-grade only.
- Everything assumes a clean side-on camera view, as the reference document
  itself requires for the angles it defines.
- Grok's API was implemented against xAI's documented format but could not
  be live-tested from the sandbox this was built in (its network policy
  blocks `api.x.ai`); Gemini reachability was confirmed directly. Test the
  Grok path on your own machine, where no such restriction applies.
