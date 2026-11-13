"""LLM extraction: one call per document, returns value and confidence per field."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .data import MAX_PROMPT_CHARS, cuad_definitions, doc_text, source_path

PROMPT = """You extract fields from one document. Reply with ONE JSON object and nothing else:
{{"<field name>": {{"value": <value or null>, "confidence": <number from 0 to 1>}}}}

Rules:
- Copy values exactly as printed in the document. Do not guess, infer or calculate.
- If a field is not in the document, or you cannot read it, use null.
- A wrong value is much worse than null. Be honest about uncertainty in "confidence".
- Use the field description where one is given; it defines the expected format.

Fields:
{fields}
"""


def describe(schema: dict, definitions: dict[str, str] | None = None) -> str:
    """One line per field: name, definition, and answer-shape hints.

    The definition comes from the schema when it has one (ExtractBench), otherwise
    from the dataset's own definitions (CUAD, read from the raw question text).
    """
    definitions = definitions or {}
    lines = []
    for name, spec in schema.get("properties", {}).items():
        desc = spec.get("description") or definitions.get(name) or spec.get("title") or ""
        line = f"- {name}"
        if desc:
            line += f": {str(desc)[:600]}"
        if spec.get("x-normalizer") == "span":
            line += (" (answer: if it is a clause, the complete sentence(s) that state it; "
                     "if it is a date, name or title, the value exactly as written; copied exactly either way)")
        lines.append(line)
    return "\n".join(lines)


def build_messages(rec: dict) -> list[dict]:
    definitions = cuad_definitions() if rec["source_dataset"] == "cuad" else {}
    prompt = PROMPT.format(fields=describe(rec["schema"], definitions))
    text = doc_text(rec)
    if text:
        body = f"{prompt}\n--- DOCUMENT ---\n{text[:MAX_PROMPT_CHARS]}"
        return [{"role": "user", "content": [{"type": "text", "text": body}]}]
    # No text layer (a scan): attach the file itself for a vision-capable model.
    path = source_path(rec)
    if not path.exists():
        raise FileNotFoundError(f"no text and no file for {rec['doc_id']}: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    if mime.startswith("image/"):
        part = {"type": "image_url", "image_url": {"url": url}}
    else:
        part = {"type": "file", "file": {"filename": path.name, "file_data": url}}
    return [{"role": "user", "content": [{"type": "text", "text": prompt}, part]}]


def llm_config() -> tuple[str, str, str]:
    base, key, model = (os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
    if not (base and key and model):
        raise RuntimeError("Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (in .env, see .env.example).")
    return base.rstrip("/"), key, model


RETRY_STATUS = {429, 500, 502, 503, 504}  # rate limit and transient server errors


def chat(messages: list[dict], retries: int = 6) -> str:
    base, key, model = llm_config()
    body = json.dumps({"model": model, "temperature": 0, "messages": messages}).encode()
    request = urllib.request.Request(
        base + "/chat/completions", body,
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in RETRY_STATUS:
                break  # e.g. 401 bad key: retrying will not help
        except Exception as e:  # network error, timeout, bad reply shape
            last = e
        # Back off longer each time: 5, 10, 20, 40, 80 seconds.
        time.sleep(min(5 * 2 ** attempt, 120))
    raise RuntimeError(f"LLM call failed after {attempt + 1} tries: {last}")


def parse_json(reply: str) -> dict:
    s = reply.strip()
    if s.startswith("```"):
        s = s.strip("`").removeprefix("json").strip()
    try:
        return json.loads(s[s.index("{"): s.rindex("}") + 1])
    except ValueError:
        return {}


def extract_fields(rec: dict, chat_fn=None) -> dict[str, dict[str, Any]]:
    """chat_fn(messages) -> reply text; defaults to chat(). The evaluator injects a cached client."""
    answer = parse_json((chat_fn or chat)(build_messages(rec)))
    fields = {}
    for name in rec["schema"].get("properties", {}):
        item = answer.get(name)
        if not isinstance(item, dict):
            item = {"value": item}
        try:
            confidence = min(max(float(item.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.0
        fields[name] = {"value": item.get("value"), "confidence": confidence}
    return fields
