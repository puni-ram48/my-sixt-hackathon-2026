"""LLM client for the hybrid pipeline: the project's OpenAI-compatible endpoint, plus
a disk cache and usage logging.

- Same configuration as the baseline (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from .env).
- Every reply is cached under data/cache/llm/<sha256 of model+messages>.json, so experiments that
  send an identical prompt (e.g. config C and D share the extraction step) pay once, and the
  evaluation can be replayed with no API calls. The cache keeps the original token usage and
  latency, so reported cost and latency are those of a real, uncached run.
- Raw replies are therefore always saved (the error report asked for this).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from solution.data import DATA
from solution.extract import RETRY_STATUS, llm_config

log = logging.getLogger("hybrid.llm")
CACHE = DATA / "cache" / "llm"


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


def _post(base: str, key: str, payload: dict, retries: int = 6) -> tuple[dict, float]:
    request = urllib.request.Request(
        base + "/chat/completions", json.dumps(payload).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    last: Exception | None = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in RETRY_STATUS:
                break
        except Exception as e:  # network error, timeout
            last = e
        time.sleep(min(5 * 2 ** attempt, 120))
    raise RuntimeError(f"LLM call failed: {last}")


def chat(messages: list[dict], model: str | None = None, use_cache: bool = True) -> LLMResult:
    base, key, default_model = llm_config()
    model = model or default_model
    payload = {"model": model, "temperature": 0, "messages": messages}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{digest}.json"
    if use_cache and path.exists():
        try:
            c = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(c["text"], model, c["prompt_tokens"], c["completion_tokens"], c["latency_s"], True)
        except (ValueError, KeyError):
            pass  # corrupt cache entry: call again
    reply, latency = _post(base, key, payload)
    try:
        text = reply["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected reply shape: {str(reply)[:200]}")
    usage = reply.get("usage") or {}
    res = LLMResult(text, model, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
                    round(latency, 2), False)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text, "prompt_tokens": res.prompt_tokens,
                                "completion_tokens": res.completion_tokens, "latency_s": res.latency_s,
                                "model": model}), encoding="utf-8")
    return res


def verifier_model() -> str | None:
    """Optional different model for the verifier (LLM_VERIFY_MODEL). None = same as extractor."""
    return os.environ.get("LLM_VERIFY_MODEL") or None


def parse_json_object(reply: str) -> dict:
    """First complete {...} object in a reply; {} if there is none.

    Decodes from the first "{" and stops at the end of that object, so a reply that contains several
    objects (or text/code fences around one) still yields the first object.
    """
    decoder = json.JSONDecoder()
    s = reply or ""
    start = s.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(s, start)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        start = s.find("{", start + 1)
    return {}
