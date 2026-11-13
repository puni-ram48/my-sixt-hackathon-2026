"""Tests for the grounding gate, plus controls on the real dev data.

Run from Project_1:  .venv\\Scripts\\python.exe -m unittest tests.test_verify -v
"""
import unittest

from solution.data import DATA, doc_text, read_jsonl
from solution.run import process
from solution.verify import Grounder, gate

DOC = ("This Agreement is to be construed according to the laws of the State of Illinois. "
       "The Distributor shall pay $1,250.00 within 30 days. Agreement Date: 7th day of September, 1999.")


class GrounderTests(unittest.TestCase):
    def setUp(self):
        self.g = Grounder(DOC)

    def test_verbatim_is_found(self):
        self.assertEqual(self.g.score("the State of Illinois"), 1.0)

    def test_case_spacing_and_punctuation_ignored(self):
        self.assertEqual(self.g.score("  THE state   of illinois  "), 1.0)

    def test_thousands_separator_matches_plain_number(self):
        self.assertEqual(self.g.score(1250), 1.0)
        self.assertEqual(self.g.score("1,250.00"), 1.0)

    def test_fabricated_value_scores_low(self):
        self.assertLess(self.g.score("Ontario, Canada and Mexico"), 0.5)

    def test_multi_answer_needs_every_part(self):
        self.assertEqual(self.g.score("Illinois | 30 days"), 1.0)
        self.assertLess(self.g.score("Illinois | Texas"), 0.9)

    def test_booleans_and_objects_are_not_checked(self):
        self.assertIsNone(self.g.score(True))
        self.assertIsNone(self.g.score("false"))  # boolean written as text, e.g. ExtractBench
        self.assertIsNone(self.g.score({"a": 1}))

    def test_empty_string_is_not_grounded(self):
        self.assertEqual(self.g.score(""), 0.0)


class SentenceAlignmentTests(unittest.TestCase):
    """Made-up texts only: the rule must come from the method, not from gold answers."""

    TEXT = ("Alpha clause one applies. Beta clause two is here; it runs for 30 days.\n\n"
            "Gamma stands alone.")

    def setUp(self):
        self.g = Grounder(self.TEXT)

    def test_whole_sentence_is_aligned(self):
        self.assertTrue(self.g.aligned("Beta clause two is here; it runs for 30 days."))

    def test_several_whole_sentences_are_aligned(self):
        self.assertTrue(self.g.aligned("Alpha clause one applies. Beta clause two is here; it runs for 30 days."))

    def test_fragment_is_not_aligned(self):
        self.assertFalse(self.g.aligned("clause two is here"))
        self.assertFalse(self.g.aligned("Beta clause two"))

    def test_paragraph_is_split_into_sentences(self):
        self.assertTrue(self.g.aligned("Gamma stands alone."))

    def test_non_string_is_not_checked(self):
        self.assertIsNone(self.g.aligned(42))

    def test_clause_after_list_marker_is_aligned(self):
        text = "Intro sentence here.\n\n         (A)      Gamma stands alone and it ends here.\n\n(B) Delta follows."
        g = Grounder(text)
        self.assertTrue(g.aligned("Gamma stands alone and it ends here."))
        self.assertTrue(g.aligned("Delta follows."))
        self.assertFalse(g.aligned("stands alone and it"))  # still a fragment, mid-sentence

    def test_section_number_before_clause_is_aligned(self):
        text = "Previous text ends here.\n\n         5.6      No Action to Invalidate. During the Term nothing happens."
        g = Grounder(text)
        self.assertTrue(g.aligned("During the Term nothing happens."))

    def test_absent_value_is_not_aligned(self):
        self.assertFalse(self.g.aligned("Delta applies."))


class GateTests(unittest.TestCase):
    def test_ungrounded_value_becomes_null(self):
        self.assertEqual(gate("made up", 0.99, 0.2), (None, 0.0))

    def test_grounded_value_is_kept(self):
        self.assertEqual(gate("Illinois", 0.9, 1.0), ("Illinois", 0.9))

    def test_unchecked_value_is_kept(self):
        self.assertEqual(gate(True, 0.8, None), (True, 0.8))


class RealDataControls(unittest.TestCase):
    """Positive control: true answers must pass. Negative control: wrong answers must fail."""

    @classmethod
    def setUpClass(cls):
        records = read_jsonl(DATA / "unified/dev.jsonl")
        cls.cuad = [r for r in records if r["source_dataset"] == "cuad"][:40]

    def test_gold_spans_are_grounded(self):
        checked = passed = 0
        for rec in self.cuad:
            grounder = Grounder(doc_text(rec))
            for value in rec["gold"].values():
                if value is None:
                    continue
                checked += 1
                passed += grounder.score(value) >= 0.9
        self.assertGreater(checked, 100)
        self.assertGreaterEqual(passed / checked, 0.98, f"{passed}/{checked} gold spans grounded")

    def test_wrong_answers_from_other_contracts_fail(self):
        checked = rejected = 0
        for rec, other in zip(self.cuad, self.cuad[1:] + self.cuad[:1]):
            grounder = Grounder(doc_text(rec))  # text of this contract...
            for value in other["gold"].values():  # ...checked against another contract's answers
                if not isinstance(value, str) or len(value) < 40:
                    continue
                checked += 1
                rejected += grounder.score(value) < 0.9
        # Some clauses are boilerplate that also appears in other contracts, so allow a margin.
        self.assertGreater(checked, 50)
        self.assertGreaterEqual(rejected / checked, 0.5)

    def test_pipeline_with_true_answers_books_them(self):
        """Plumbing check: if the model returns the gold answers, they are booked."""
        rec = self.cuad[0]

        def oracle(_rec):
            return {k: {"value": v, "confidence": 0.99} for k, v in rec["gold"].items()}

        pred = process(rec, answer=oracle)
        for name, value in rec["gold"].items():
            self.assertEqual(pred["fields"][name]["value"], value)

    def test_pipeline_nulls_hallucinated_answers(self):
        rec = self.cuad[0]

        def hallucinate(_rec):
            return {k: {"value": "Ontario, Canada and Mexico", "confidence": 0.99} for k in rec["gold"]}

        pred = process(rec, answer=hallucinate)
        self.assertTrue(all(f["value"] is None for f in pred["fields"].values()))

    def test_failed_extraction_sends_everything_to_review(self):
        rec = self.cuad[0]

        def broken(_rec):
            raise RuntimeError("endpoint down")

        pred = process(rec, answer=broken)
        self.assertEqual(set(pred["fields"]), set(rec["schema"]["properties"]))
        self.assertTrue(all(f["value"] is None for f in pred["fields"].values()))


if __name__ == "__main__":
    unittest.main()
