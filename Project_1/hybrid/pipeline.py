"""Hybrid pipeline: one record in -> one prediction line + a full trace out.

    preprocess (clauses) -> retrieve (BM25 / embeddings / classifier, RRF) -> extract with evidence
    -> [verify] -> decide

Configurations used in the evaluation:
    A  bm25 retrieval                         + extraction + existing validation + confidence
    B  embedding retrieval                    + extraction + existing validation + confidence
    C  bm25 + embedding + classifier (RRF)    + extraction + clause-type gate
    D  C + independent verification
    E  agentic RAG: the LLM searches (BM25) and reads clauses itself, then answers  (hybrid/agent.py)
    F  E + independent verification
    G  baseline whole-document extraction + hybrid evidence gates (clause type, completeness, field policy)
    H  G + independent verification
Records that are not contracts (ExtractBench forms, scans) go to the forms pipeline (hybrid/forms.py, config
forms_final: two independent vision reads + text-layer checks + learned field policy).
HYBRID_NONCONTRACT=review sends every form field to a human instead; =baseline runs the original pipeline.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, replace

from solution.data import doc_text
from solution.verify import Grounder

from .decide import Policy, decide, signals_for
from .extract import extract
from .fields import field_specs
from .retrieve import DocIndex
from .verify import verify

log = logging.getLogger("hybrid.pipeline")


@dataclass
class Config:
    name: str
    methods: tuple[str, ...] = ("bm25", "embed", "classifier")
    k: int = 6
    verify: bool = True
    policy: Policy = field(default_factory=Policy)
    fields_per_call: int = 14
    agent: bool = False       # E/F: agentic search-and-read instead of one-shot retrieval
    use_models: bool = True   # load embeddings + classifier (False = BM25 only, no CPU-heavy model)
    baseline_extract: bool = False  # G/H: the original whole-document prompt (solution/extract.py)
    field_policy: str | None = None  # name of a fitted field policy (hybrid/field_policy.py) to apply
    verifier_model: str | None = None  # None = LLM_VERIFY_MODEL or the extractor model
    weights: dict | None = None  # weighted RRF; None = equal weights


CONFIGS = {
    "A_bm25": Config("A_bm25", methods=("bm25",), verify=False, use_models=False, policy=Policy(use_classifier=False)),
    "B_embed": Config("B_embed", methods=("embed",), verify=False, policy=Policy(use_classifier=False)),
    # Classifier weight 5 and k=8 chosen on calibration contracts (hybrid/eval_retrieval.py, dev-train 160-199):
    # weighted RRF hit@8 86.8% vs 77.8% for equal weights and 57.4% for BM25.
    "C_hybrid": Config("C_hybrid", verify=False, k=8, weights={"classifier": 5.0}),
    "D_hybrid_verify": Config("D_hybrid_verify", verify=True, k=8, weights={"classifier": 5.0}),
    "E_agent": Config("E_agent", methods=("bm25",), verify=False, agent=True, use_models=False,
                      policy=Policy(use_classifier=False)),
    "F_agent_verify": Config("F_agent_verify", methods=("bm25",), verify=True, agent=True, use_models=False,
                             policy=Policy(use_classifier=False)),
    "G_baseline_gated": Config("G_baseline_gated", verify=False, baseline_extract=True,
                               field_policy="G_baseline_gated", policy=Policy(require_in_candidates=False)),
    "H_baseline_gated_verify": Config("H_baseline_gated_verify", verify=True, baseline_extract=True,
                                      field_policy="G_baseline_gated", policy=Policy(require_in_candidates=False)),
    # Same as H with a verifier from another model family (more independent than a second Claude pass).
    "H_verify_gpt": Config("H_verify_gpt", verify=True, baseline_extract=True, verifier_model="gpt-5.5",
                           field_policy="G_baseline_gated", policy=Policy(require_in_candidates=False)),
    # Submitted configuration: G with threshold 0.95 (the only threshold with a positive net saving on both the
    # calibration contracts and the held-out dev-val contracts; verifiers measured and found to add nothing).
    "FINAL": Config("FINAL", verify=False, baseline_extract=True, field_policy="G_baseline_gated",
                    policy=Policy(require_in_candidates=False, threshold=0.95)),
    "H_verify_gemini": Config("H_verify_gemini", verify=True, baseline_extract=True,
                              verifier_model="vertex_ai/gemini-3.1-pro",
                              field_policy="G_baseline_gated", policy=Policy(require_in_candidates=False)),
}
DEFAULT = "FINAL"


def _usage(calls) -> dict:
    return {"calls": len(calls), "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "llm_latency_s": round(sum(c.latency_s for c in calls), 2)}


def process(rec: dict, cfg: Config, pool=None) -> tuple[dict, dict]:
    """Never raises. Returns (prediction line, trace)."""
    t0 = time.perf_counter()
    specs = field_specs(rec)
    text = doc_text(rec) if rec.get("source_dataset") == "cuad" else None
    if not text:
        return _baseline(rec, t0)

    idx = DocIndex(rec["doc_id"], text, use_models=cfg.use_models)
    cands = {f.name: idx.retrieve(f, methods=cfg.methods, k=cfg.k, weights=cfg.weights) for f in specs}
    agent_log = None
    if cfg.agent:
        from .agent import run_agent, seen_candidates

        run = run_agent(idx, specs)
        extractions, calls, agent_log = run.extractions, run.calls, run.log
        verify_cands = cands                                  # field-specific BM25 clauses for contradiction checks
        seen = seen_candidates(idx, run)
        cands = {f.name: seen for f in specs}                 # evidence must come from clauses the agent read
    elif cfg.baseline_extract:
        extractions, calls = _baseline_extractions(rec, specs)
        verify_cands = cands
    else:
        extractions, calls = extract(idx, specs, cands, cfg.fields_per_call, pool)
        verify_cands = cands
    verdicts, vcalls = ({}, [])
    if cfg.verify:
        verdicts, vcalls = verify(idx, {f.name: f for f in specs}, extractions, verify_cands, pool=pool,
                                  model=cfg.verifier_model)
    policy = cfg.policy
    if cfg.field_policy:
        from .field_policy import load

        table = load(cfg.field_policy)
        if table is None:
            log.warning("field policy %s not fitted: every field goes to review", cfg.field_policy)
        policy = replace(policy, field_policy=table or {})

    grounder = Grounder(text)
    fields, trace_fields = {}, {}
    for f in specs:
        ex = extractions.get(f.name)
        if ex is None:
            from .extract import Extraction
            ex = Extraction(f.name, error="missing")
        sig = signals_for(idx, grounder, f, ex, cands[f.name])
        d = decide(f, ex, sig, verdicts.get(f.name), policy)
        fields[f.name] = d.output()
        trace_fields[f.name] = {
            "decision": d.action, "reasons": d.reasons, "signals": d.signals,
            "retrieval": [c.trace() for c in cands[f.name][:8]],
            "extraction": ex.trace(),
            "verification": verdicts[f.name].trace() if f.name in verdicts else None,
        }
        log.debug("%s | %s | %s | %s", rec["doc_id"], f.name, d.action, d.reasons[0])
    usage = {"extract": _usage(calls), "verify": _usage(vcalls),
             "wall_s": round(time.perf_counter() - t0, 2), "n_clauses": len(idx.clauses)}
    return ({"doc_id": rec["doc_id"], "fields": fields},
            {"doc_id": rec["doc_id"], "config": cfg.name, "usage": usage, "fields": trace_fields,
             **({"agent": agent_log} if agent_log is not None else {})})


def _baseline_extractions(rec: dict, specs) -> tuple[dict, list]:
    """The original whole-document prompt, through the cached client. Same replies as the baseline run."""
    from solution.extract import extract_fields

    from .extract import Extraction
    from .llm import chat

    calls = []

    def chat_fn(messages):
        res = chat(messages)
        calls.append(res)
        return res.text

    try:
        raw = extract_fields(rec, chat_fn=chat_fn)
    except Exception as e:
        log.warning("%s: baseline extraction failed: %s", rec["doc_id"], e)
        return {f.name: Extraction(f.name, error=str(e)) for f in specs}, calls
    out = {}
    for f in specs:
        item = raw.get(f.name) or {}
        v = item.get("value")
        if isinstance(v, str) and not v.strip():
            v = None
        if v is not None and not isinstance(v, str):
            v = str(v)
        out[f.name] = Extraction(f.name, value=v, confidence=float(item.get("confidence", 0.0) or 0.0),
                                 supported=v is not None, explanation="whole-document extraction")
    return out, calls


def _baseline(rec: dict, t0: float) -> tuple[dict, dict]:
    """Records that are not contracts."""
    usage = {"wall_s": round(time.perf_counter() - t0, 2)}
    mode = os.environ.get("HYBRID_NONCONTRACT", "forms_final")
    from .forms import FORMS_CONFIGS, process as forms_process

    if mode in FORMS_CONFIGS:
        return forms_process(rec, FORMS_CONFIGS[mode])
    if mode.lower() == "baseline":
        from solution.run import process as baseline_process

        return baseline_process(rec), {"doc_id": rec["doc_id"], "config": "baseline_fallback", "usage": usage,
                                       "fields": {}}
    props = (rec.get("schema") or {}).get("properties", {})
    return ({"doc_id": rec["doc_id"], "fields": {n: {"value": None, "confidence": 0.0} for n in props}},
            {"doc_id": rec["doc_id"], "config": "out_of_scope_review", "usage": usage, "fields": {},
             "reason": "not a contract: every field sent to human review"})
