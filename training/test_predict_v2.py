import types, unittest
from training.predict_v2 import encoder_capacity, _encode_words


class Cap(unittest.TestCase):
    def _model(self, limit, pad):
        cfg = types.SimpleNamespace(max_position_embeddings=limit, pad_token_id=pad)
        return types.SimpleNamespace(encoder=types.SimpleNamespace(config=cfg))

    def test_phobert(self):
        # 258 vị trí, padding_idx=1 -> chuỗi dài nhất an toàn là 256
        self.assertEqual(encoder_capacity(self._model(258, 1)), 256)

    def test_bert_style(self):
        self.assertEqual(encoder_capacity(self._model(512, 0)), 511)

    def test_encode_respects_max_len(self):
        tok = types.SimpleNamespace(encode=lambda w, add_special_tokens: [7, 8, 9])
        ids, first, last, limit = _encode_words(["a"] * 100, tok, 20, 0, 2, 3)
        self.assertLessEqual(len(ids), 20)
        self.assertEqual(limit, len(first))
        self.assertTrue(all(i < len(ids) for i in first + last))


if __name__ == "__main__":
    unittest.main()
