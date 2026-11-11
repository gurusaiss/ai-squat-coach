# How AI Was Used

Claude (AI agent) built this end-to-end. The requester answered a few
scoping questions up front instead of writing code.

## What AI did

- Read the 35-page reference PDF and turned it into the skill file,
  including which rules a side-view video can and can't check.
- Wrote all the code: tracking, rules engine, AI write-up layer, video
  overlay, backend, frontend.
- Wrote the tests and these docs.

## What we checked, not just trusted

- Tested the library actually worked before building on it (see mistake
  below) instead of assuming the version was fine.
- Unit-tested the rule engine with known right/wrong answers, so the
  logic is proven without needing a real video.
- The AI write-up step is boxed in: it gets a fixed JSON shape to follow
  and is told never to change a verdict or invent a rule. If its output
  breaks that shape, the app falls back to plain text instead of trusting it.
- Every citation in the skill file was checked against the real PDF page,
  not written from general knowledge of squats.

## Mistake we caught and fixed

First install pulled the newest MediaPipe version, which uses a different
API and needs an internet download we don't have here. Caught this
immediately by testing the import before writing more code — not after.
Fixed by pinning to an older version that works fully offline.

## Second thing we caught

Some number thresholds were hard-coded in Python instead of the skill
file — against our own rule that "the skill file controls behavior."
Found this on a self-review, moved them into the skill file, and
re-ran tests to confirm nothing broke.

## Later revision (after review feedback)

Two clear asks: use only Grok + Gemini (no paid-only API), and improve
the UX. Done directly:
- Swapped the AI call from Anthropic to Grok (primary) + Gemini
  (fallback), both via plain HTTPS, no extra library needed.
- Confirmed Gemini's endpoint is reachable from here; confirmed (via the
  sandbox's own network log) that Grok's isn't — so that path is
  built correctly but not live-tested here. Flagged honestly instead of
  claimed as tested.
- Re-checked the assignment line-by-line. Found one real gap: "show the
  bar's path" was just a dot, not a path. Fixed it.
- Added a progress stepper, a results scorecard, click-to-jump findings,
  and an "about this analysis" panel — each fixes a specific way the old
  UI made users work harder than needed.
