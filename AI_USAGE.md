# How AI was used to build this

This entire submission was built by Claude (an AI agent, running as Claude
Code) working through the assignment directly, with the requester
answering a small number of scoping questions (platform, pose-tracking
library, reasoning-layer design, deployment approach) up front rather than
writing code.

## What was delegated

- Reading and extracting the 35-page reference document into the
  structured `skill/squat_skill.yaml`, including picking which criteria are
  / are not assessable from a side-view video and why.
- All pipeline code: pose extraction, rep detection, geometric
  measurements, the deterministic verdict engine, the LLM write-up layer,
  video annotation, the FastAPI backend, and the frontend.
- Environment setup (Python version/dependency resolution for MediaPipe),
  unit tests, and this set of docs.

## What was inspected / verified (not just trusted)

- **Dependency compatibility was verified by actually importing and
  running the library**, not assumed from version numbers. See "one
  meaningful mistake" below.
- **The verdict logic was unit-tested against synthetic, hand-constructed
  landmark trajectories with a known ground truth** (e.g. "this synthetic
  rep has hip clearly below knee at the bottom -> depth must verdict
  `meets`"), specifically so the squat-domain logic could be checked
  without depending on having a real tracked video available yet
  (`backend/tests/test_pipeline_logic.py`, all passing).
- **The LLM write-up layer was constrained and checked, not trusted
  freely**: its system prompt explicitly forbids changing a verdict or
  inventing a threshold, its output is parsed against an expected JSON
  shape, and a failure (bad JSON, no credentials, network error) falls
  back to static skill-authored text rather than silently producing
  something ungrounded. This whole design exists because the assignment
  explicitly warns that a model's own squat advice is not an acceptable
  substitute for the document's standard.
- Every citation (`ref_pages`, section name, quote) in `squat_skill.yaml`
  was taken from directly re-reading the specific page of the reference
  PDF it's attached to, not generated from the model's general knowledge
  of squat technique -- this was the main point of extracting a "skill" in
  the first place, and it was checked criterion-by-criterion against the
  source pages rather than assumed correct.
- End-to-end evidence (see `docs/evidence/`): the full upload -> track ->
  annotate -> assess flow was run against the common sample video and one
  additional difficult/unsuitable video, and the output was inspected
  (not just "it didn't crash") for whether landmark placement, bar path,
  and the resulting verdicts looked right given what's visible in the
  source clip.

## One meaningful mistake caught and corrected

The first dependency install pulled the newest `mediapipe` from PyPI
(`1.2.2`). That version ships an entirely different "Tasks" API; code
written against the classic `mediapipe.solutions.pose.Pose()` interface
failed immediately with `AttributeError: module 'mediapipe' has no
attribute 'solutions'`. Worse, the new Tasks API requires downloading a
`.task` model file from Google Cloud Storage on first use -- which this
sandboxed environment's network policy would have blocked anyway, so even
rewriting against the new API would have silently failed at runtime
without a clear error. This was caught immediately by actually importing
the library and instantiating a pose model before writing the rest of the
pipeline against it, rather than writing the whole pipeline first and
discovering the mismatch during end-to-end testing. The fix was pinning
`mediapipe==0.10.21` (which bundles its model weights in the wheel, so it
works fully offline) and re-pinning `opencv-python-headless` and `numpy`
to versions compatible with that older mediapipe's `numpy<2` requirement.
This is recorded in `BUILD_NOTES.md` as a deliberate, documented pin
rather than left as an unexplained version number.

## A later revision: provider swap and UX pass

After the first working version, the requester gave two concrete
directives rather than open-ended ones: (1) use only Grok and Gemini,
specifically because both have a no-cost path to a working API key, and no
other paid API; (2) make the UX as strong as possible and re-verify every
feature the assignment document actually asked for. Both were executed
directly rather than re-litigated:

- Rewrote `assess.py` from an Anthropic-SDK/`claude`-CLI call chain to
  direct HTTPS calls to xAI's and Google's APIs (no vendor SDKs needed,
  smaller dependency footprint), in the specified priority order, with the
  same "never changes a verdict" system prompt and the same template
  fallback as the last resort.
- Verified reachability directly rather than assuming it: confirmed
  Gemini's endpoint answers from this sandbox (got a real 403-missing-key
  response, i.e. the host is reachable), and confirmed via the sandbox's
  own network-proxy status endpoint that `api.x.ai` is explicitly
  policy-blocked here -- so the Grok path is implemented to spec but
  disclosed as not live-tested from this environment, rather than silently
  assumed to work.
- Looked up current model naming and free-tier terms for both providers via
  web search rather than guessing from training data (model names and
  pricing/limits for both products have changed multiple times through
  2025-2026), and exposed both as overridable env vars (`GROK_MODEL`,
  `GEMINI_MODEL`) instead of hard-coding a value likely to go stale.
- Re-read the assignment's "what the application must do" section against
  the shipped app feature-by-feature and found one real gap: "track the
  barbell and show its path" was implemented as a per-frame dot, not a
  path. Fixed by adding an accumulating, per-rep translucent trail in
  `annotate.py` rather than leaving it as a dot and only mentioning the gap
  in docs.
- Added the stepper, scorecard, click-to-seek, and "about this analysis"
  panel described in BUILD_NOTES.md specifically because "best possible
  UX" is a requirement to act on, not a vague aspiration -- each addition
  maps to a specific way the previous UI made the user do more work than
  necessary (guessing what stage processing was in, reading every finding
  to get the headline result, manually scrubbing video to verify a claim).

## A second thing caught on self-review

After the first working version was running, a read-through of
`verdicts.py` found that several calibration numbers (the bar-path
deviation band, the hip-drive ascent thresholds, the head-tilt threshold)
had been written as plain Python literals instead of being read from
`skill/squat_skill.yaml`, even though the skill file already documented
them as calibration values. That directly undercut the architecture's main
claim (edit the skill, not the code, to change behavior) and would have
been awkward to discover for the first time live during the demo's "we may
... change a rule in the source document" step. Fixed by adding explicit
`calibration_*` fields to the skill file and having the Python read from
there; verified by re-running the unit test suite after the change
(`backend/tests/test_pipeline_logic.py`, still all passing) and by doing a
clean `pip install -r backend/requirements.txt` into a brand-new venv to
confirm the dependency set reproduces correctly from the repo alone, not
just in the already-patched environment.

## Selected prompts / traces

- Skill extraction: re-read each page image of the reference PDF and asked
  "what exact, literal standard does this page state, and is it checkable
  from a side-on video of the movement, or does it require a different
  camera angle / equipment visibility?" for every page, rather than
  summarizing the chapter's general advice.
- Assessment write-up system prompt is included verbatim in
  `backend/app/pipeline/assess.py` (`SYSTEM_PROMPT`) -- it is short enough
  that reproducing it here would just be a duplicate; that file is the
  source of truth for what the embedded agent is and isn't allowed to do.
