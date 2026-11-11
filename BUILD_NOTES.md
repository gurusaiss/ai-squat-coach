# Build Notes

## How it works

```
Upload -> Pose tracking -> Bar estimate -> Rep/depth detection
       -> Rule engine (verdict) -> AI write-up -> Annotated video
```

The skill file (`skill/squat_skill.yaml`) holds every rule and threshold.
Code reads it at runtime. Change the file, the app's behavior changes.

## What we used, and why

- **MediaPipe Pose** — free, runs offline, gives the joints we need
  (hip/knee/ankle/shoulder). Pinned to version `0.10.21`: the newest
  version needs an internet download we don't have in this sandbox.
- **Bar tracking** — no trained bar detector (out of scope for the time
  box). We estimate it from wrist position, refined with edge detection.
  Clearly marked as an estimate, not verified tracking.
- **Rules decide the verdict, not AI.** Plain Python checks the numbers
  against the skill file's thresholds. The AI model (Grok, then Gemini,
  then plain text as last resort) only writes the explanation sentence —
  it cannot change a verdict or invent a rule. This is unit-tested.
- **Grok + Gemini, no paid-only API.** Both have a free path to a working
  key. If both fail, the app shows plain template text instead of
  guessing. Model names are configurable (`GROK_MODEL`, `GEMINI_MODEL`)
  since providers rename models often.
- **Plain HTML/JS frontend, FastAPI backend.** No framework needed for
  this scope.
- **Simple in-memory job queue.** Proves the real upload -> result flow.
  Not restart-safe — a real product would use a proper queue.

## Time spent

- First working build: **~6.5h** (within the 4-6h target).
- Second pass (provider swap + UX polish, after review feedback): **~2.5h**.
- **Total: ~9h.**

## Cost per video

- Pose tracking: free, runs on CPU locally.
- One AI call per video (not per finding) — small enough to stay inside
  Gemini's free tier, or xAI's free starting credit.

## Trade-offs

- Bar path is an estimate, not verified tracking — disclosed every time.
- Verdicts are rule-based, not AI-guessed — more trustworthy, less flexible.
- Side-view only. Anything that needs another camera angle says
  "can't assess" instead of guessing.

## Known limits

- No trained bar/plate detector.
- Eye-gaze check is a rough proxy, often "can't assess."
- Job queue doesn't survive a server restart.
- No login / multi-user support — single-user demo only.
- Videos capped at 90 seconds for fast demo turnaround.
