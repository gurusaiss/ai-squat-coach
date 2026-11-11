"""Verifies the Grok -> Gemini -> template fallback chain in assess.py
without making real network calls or needing real API keys: requests.post
is monkeypatched to simulate each provider's response/failure shape.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.verdicts import Finding
from app.pipeline import assess


def _finding():
    return Finding(
        criterion_id="depth", rep_index=1, verdict="meets", variant=None,
        timestamp_sec=1.23, frame=37, measurement={"depth_margin_norm": 0.12},
        confidence_note="", ref_pages=[2, 18], sections=["Figure 2-1 caption"],
        quote="The hip joint ... must drop below parallel with the ground.",
    )


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


EXPECTED_JSON = {
    "findings": [{"key": "depth|1", "explanation": "ok", "feedback": "keep it up"}],
    "overall_summary": "fine",
    "confidence_caveats": "none",
}


def test_grok_used_when_key_present_and_call_succeeds(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "fake-grok-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_post(url, **kwargs):
        assert url == assess.GROK_API_URL
        return FakeResponse(200, {"choices": [{"message": {"content": json.dumps(EXPECTED_JSON)}}]})

    with patch("app.pipeline.assess.requests.post", side_effect=fake_post):
        result, method = assess.generate_writeups([_finding()])
    assert method == "grok"
    assert result["overall_summary"] == "fine"


def test_falls_back_to_gemini_when_grok_fails(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "fake-grok-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    def fake_post(url, **kwargs):
        if url == assess.GROK_API_URL:
            return FakeResponse(429, text="rate limited")
        assert "generativelanguage.googleapis.com" in url
        return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(EXPECTED_JSON)}]}}]})

    with patch("app.pipeline.assess.requests.post", side_effect=fake_post):
        result, method = assess.generate_writeups([_finding()])
    assert method == "gemini"
    assert result["overall_summary"] == "fine"


def test_falls_back_to_template_when_both_fail(monkeypatch):
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result, method = assess.generate_writeups([_finding()])
    assert method == "template_fallback"
    assert result["findings"][0]["key"] == "depth|1"
    assert result["findings"][0]["explanation"]  # non-empty template text


def test_falls_back_to_template_on_bad_json(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "fake-grok-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_post(url, **kwargs):
        return FakeResponse(200, {"choices": [{"message": {"content": "not json at all"}}]})

    with patch("app.pipeline.assess.requests.post", side_effect=fake_post):
        result, method = assess.generate_writeups([_finding()])
    assert method == "template_fallback"


if __name__ == "__main__":
    class FakeMonkeypatch:
        def __init__(self):
            self._saved = {}

        def setenv(self, k, v):
            import os
            self._saved.setdefault(k, os.environ.get(k))
            os.environ[k] = v

        def delenv(self, k, raising=False):
            import os
            self._saved.setdefault(k, os.environ.get(k))
            os.environ.pop(k, None)

        def restore(self):
            import os
            for k, v in self._saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    tests = [
        test_grok_used_when_key_present_and_call_succeeds,
        test_falls_back_to_gemini_when_grok_fails,
        test_falls_back_to_template_when_both_fail,
        test_falls_back_to_template_on_bad_json,
    ]
    for t in tests:
        mp = FakeMonkeypatch()
        try:
            t(mp)
            print(f"PASS {t.__name__}")
        finally:
            mp.restore()
    print("All provider fallback tests passed.")
