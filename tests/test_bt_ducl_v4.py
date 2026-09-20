import unittest

import numpy as np
from transformers import AutoTokenizer

from experiments.bt_ducl.semantic import encode_semantic_example, semantic_spans


BASE = "models/llama32-1b"


class SemanticMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(BASE)

    def _row(self, output):
        return {
            "instruction": "Generate a tree.",
            "input": "Move the item.",
            "output": output,
            "meta": {},
        }

    def test_spans_cover_generic_tags_closing_tags_and_values(self):
        xml = '<root><CustomAlias robot="alpha"><Sequence/></CustomAlias></root>'
        values = [xml[a:b] for a, b in semantic_spans(xml)]
        self.assertEqual(values.count("root"), 2)
        self.assertEqual(values.count("CustomAlias"), 2)
        self.assertIn("Sequence", values)
        self.assertIn("alpha", values)

    def test_mask_includes_alias_control_attributes_and_eos(self):
        xml = '<BehaviorTree ID="MainTree"><AliasMove destination="dock"/></BehaviorTree>'
        encoded = encode_semantic_example(self._row(xml), self.tokenizer, 256)
        selected = [token_id for token_id, label in zip(encoded.input_ids, encoded.labels)
                    if label != -100]
        selected_text = self.tokenizer.decode(selected, skip_special_tokens=False)
        self.assertIn("BehaviorTree", selected_text)
        self.assertIn("AliasMove", selected_text)
        self.assertIn("MainTree", selected_text)
        self.assertIn("dock", selected_text)
        self.assertEqual(selected[-1], self.tokenizer.eos_token_id)
        self.assertGreater(encoded.semantic_tokens, 0)
        self.assertLess(encoded.semantic_tokens, encoded.completion_tokens)

    def test_malformed_fragment_still_masks_recognizable_semantics(self):
        encoded = encode_semantic_example(
            self._row('<MoveTo robot="alpha"><Fallback'), self.tokenizer, 256)
        self.assertGreater(encoded.semantic_tokens, 2)


if __name__ == "__main__":
    unittest.main()
