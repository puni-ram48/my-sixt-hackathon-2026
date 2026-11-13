"""Offline tests for the hybrid pipeline (no API calls, no embedding model).

Run from Project_1:  .venv\\Scripts\\python.exe -m unittest tests.test_hybrid -v
Made-up texts only.
"""
import unittest

from hybrid.classify import clause_labels
from hybrid.clauses import split_clauses
from hybrid.decide import Policy, decide
from hybrid.extract import Extraction, _to_extraction
from hybrid.fields import FieldSpec
from hybrid.llm import parse_json_object
from hybrid.retrieve import DocIndex, find_span
from hybrid.verify import Verification, _parse

TEXT = ("1. Governing Law\n\nThis Agreement is governed by the laws of the State of Alpha.\n\n"
        "2. Limitation of Liability\n\nIn no event shall either party's total liability exceed the fees paid "
        "in the twelve months before the claim.\n\n"
        "3. Indemnification\n\nThe Supplier shall indemnify the Buyer against all third-party claims and liability.")
F = FieldSpec("Cap On Liability", "A cap on the amount of liability a party can incur.", "span", "string")


def ok_extraction(**kw):
    base = dict(field=F.name, value="In no event shall either party's total liability exceed the fees paid in the "
                "twelve months before the claim.", evidence=[], explanation="", confidence=0.95, supported=True)
    base.update(kw)
    return Extraction(**base)


GOOD = {"grounding": 1.0, "sentence_aligned": True, "in_candidates": True, "clause_type_p": 0.9}


class PreprocessTests(unittest.TestCase):
    def test_headings_attach_and_offsets_trace_back(self):
        clauses = split_clauses("d", TEXT)
        self.assertEqual([c.heading for c in clauses],
                         ["1. Governing Law", "2. Limitation of Liability", "3. Indemnification"])
        for c in clauses:
            self.assertEqual(TEXT[c.start:c.end].strip(), c.text)

    def test_find_span_ignores_whitespace(self):
        s = find_span(TEXT, "governed  by the\nlaws")
        self.assertIsNotNone(s)
        self.assertIn("governed", TEXT[s[0]:s[1]])
        self.assertIsNone(find_span(TEXT, "laws of the State of Omega"))

    def test_clause_labels_from_gold_spans(self):
        clauses = split_clauses("d", TEXT)
        s = TEXT.index("In no event")
        labels = clause_labels(clauses, {F.name: {"spans": [{"start": s, "end": s + 60}]}})
        self.assertEqual(labels, [set(), {F.name}, set()])


class RetrievalTests(unittest.TestCase):
    def test_bm25_only_retrieval_is_explained(self):
        idx = DocIndex("d", TEXT, use_models=False)
        cands = idx.retrieve(F, methods=("bm25",), k=2)
        self.assertEqual(cands[0].clause.heading, "2. Limitation of Liability")
        self.assertTrue(cands[0].reasons[0].startswith("bm25 rank 1"))

    def test_locate_maps_quote_to_clause(self):
        idx = DocIndex("d", TEXT, use_models=False)
        self.assertEqual(idx.locate("The Supplier shall indemnify the Buyer"), [2])

    def test_locate_finds_text_in_a_heading(self):
        # A title line becomes a heading; it must still map to its clause (e.g. a document name).
        idx = DocIndex("d", TEXT, use_models=False)
        self.assertEqual(idx.locate("Limitation of Liability"), [1])


class DecisionTests(unittest.TestCase):
    P = Policy(threshold=0.9)

    def test_all_good_books(self):
        d = decide(F, ok_extraction(), GOOD, Verification("SUPPORTED"), self.P)
        self.assertEqual(d.action, "BOOK")
        self.assertGreaterEqual(d.confidence, 0.9)

    def test_null_abstains(self):
        self.assertEqual(decide(F, ok_extraction(value=None, supported=False), {}, None, self.P).action, "ABSTAIN")

    def test_fabricated_value_rejected(self):
        d = decide(F, ok_extraction(), {**GOOD, "grounding": 0.3}, None, self.P)
        self.assertEqual((d.action, d.value), ("REJECT", None))

    def test_wrong_clause_rejected_even_with_high_confidence(self):
        d = decide(F, ok_extraction(confidence=0.99), {**GOOD, "clause_type_p": 0.001}, None, self.P)
        self.assertEqual(d.action, "REJECT")

    def test_verifier_unsupported_rejects_and_ambiguous_reviews(self):
        self.assertEqual(decide(F, ok_extraction(), GOOD, Verification("UNSUPPORTED"), self.P).action, "REJECT")
        d = decide(F, ok_extraction(), GOOD, Verification("AMBIGUOUS"), self.P)
        self.assertEqual(d.action, "REVIEW")
        self.assertLess(d.confidence, 0.9)
        self.assertIsNotNone(d.value)

    def test_contradiction_goes_to_review(self):
        v = Verification("SUPPORTED", contradiction=True)
        self.assertEqual(decide(F, ok_extraction(), GOOD, v, self.P).action, "REVIEW")

    def test_low_confidence_reviews(self):
        self.assertEqual(decide(F, ok_extraction(confidence=0.6), GOOD, None, self.P).action, "REVIEW")

    def test_other_strong_clauses_mean_incomplete(self):
        sig = {**GOOD, "uncovered_same_type": [["C7", 0.8], ["C9", 0.3]]}
        d = decide(F, ok_extraction(), sig, Verification("SUPPORTED"), self.P)
        self.assertEqual(d.action, "REVIEW")
        self.assertIn("C7", d.reasons[0])
        weak = {**GOOD, "uncovered_same_type": [["C9", 0.3]]}
        self.assertEqual(decide(F, ok_extraction(), weak, None, self.P).action, "BOOK")

    def test_field_policy_blocks_unreliable_and_unknown_fields(self):
        pol = Policy(field_policy={F.name: {"n": 9, "correct": 7, "bookable": False}})
        self.assertEqual(decide(F, ok_extraction(), GOOD, None, pol).action, "REVIEW")
        self.assertEqual(decide(F, ok_extraction(), GOOD, None, Policy(field_policy={})).action, "REVIEW")
        ok = Policy(field_policy={F.name: {"n": 20, "correct": 20, "bookable": True}})
        self.assertEqual(decide(F, ok_extraction(), GOOD, None, ok).action, "BOOK")

    def test_classifier_ignored_when_disabled(self):
        d = decide(F, ok_extraction(), {**GOOD, "clause_type_p": 0.001}, None, Policy(use_classifier=False))
        self.assertEqual(d.action, "BOOK")


class OutOfScopeTests(unittest.TestCase):
    def test_non_contract_records_go_to_review_without_llm(self):
        import os
        from unittest import mock
        from hybrid.pipeline import CONFIGS, process
        rec = {"doc_id": "extractbench:x", "source_dataset": "extractbench", "pages": ["raw/none.pdf"],
               "schema": {"properties": {"total": {"type": "string"}, "date": {"type": "string"}}}}
        with mock.patch.dict(os.environ, {"HYBRID_NONCONTRACT": "review"}):
            line, trace = process(rec, CONFIGS["G_baseline_gated"])
        self.assertEqual(line["fields"], {"total": {"value": None, "confidence": 0.0},
                                          "date": {"value": None, "confidence": 0.0}})
        self.assertEqual(trace["config"], "out_of_scope_review")


class MalformedReplyTests(unittest.TestCase):
    def test_parse_json_object(self):
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_object("no json here"), {})
        self.assertEqual(parse_json_object("[1, 2]"), {})
        # A reply that simulates several turns: take the first object only.
        self.assertEqual(parse_json_object('{"actions": [1]}\n```\n{"answers": {}}'), {"actions": [1]})

    def test_bad_extraction_items_are_safe(self):
        ex = _to_extraction("x", {"value": "v", "confidence": "high", "evidence": "oops", "supported": True}, set())
        self.assertEqual((ex.confidence, ex.evidence), (0.0, []))
        self.assertIsNone(_to_extraction("x", {"value": "  "}, set()).value)
        self.assertIsNone(_to_extraction("x", ["list"], set()).value)

    def test_bad_verdict_defaults_to_ambiguous(self):
        self.assertEqual(_parse({"decision": "maybe"}).decision, "AMBIGUOUS")
        self.assertEqual(_parse(None).decision, "AMBIGUOUS")


if __name__ == "__main__":
    unittest.main()
