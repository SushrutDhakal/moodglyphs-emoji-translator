import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import mmr, normalize_emotion_vector, top_k_emotions
from core.emotion_model import EMOTION_LABELS


class TestMMR(unittest.TestCase):
    def test_mmr_no_duplicates(self):
        doc_emb = np.random.randn(128)
        cand_embs = np.random.randn(10, 128)
        cand_scores = np.random.rand(10)
        
        selected = mmr(doc_emb, cand_embs, cand_scores, k=5)
        
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected), 5)
    
    def test_mmr_selects_best_first(self):
        doc_emb = np.random.randn(128)
        cand_embs = np.random.randn(10, 128)
        cand_scores = np.arange(10).astype(float)
        
        selected = mmr(doc_emb, cand_embs, cand_scores, k=3)
        
        self.assertEqual(selected[0], 9)
    
    def test_mmr_empty_candidates(self):
        doc_emb = np.random.randn(128)
        cand_embs = np.empty((0, 128))
        cand_scores = np.array([])
        
        selected = mmr(doc_emb, cand_embs, cand_scores, k=3)
        
        self.assertEqual(len(selected), 0)


class TestEmotionUtils(unittest.TestCase):
    def test_normalize_emotion_vector(self):
        vec = np.array([0.5, 0.001, 0.8, 0.0])
        
        normalized = normalize_emotion_vector(vec, min_threshold=0.01)
        
        self.assertEqual(normalized[1], 0.0)
        self.assertEqual(normalized[3], 0.0)
        self.assertGreater(normalized[0], 0)
        self.assertGreater(normalized[2], 0)
    
    def test_top_k_emotions(self):
        vec = np.array([0.1, 0.9, 0.5, 0.2, 0.7])
        labels = ['a', 'b', 'c', 'd', 'e']
        
        top = top_k_emotions(vec, labels, k=3)
        
        self.assertEqual(len(top), 3)
        self.assertEqual(top[0][0], 'b')
        self.assertEqual(top[1][0], 'e')
        self.assertEqual(top[2][0], 'c')


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        try:
            from core import Translator
            self.translator = Translator.load()
        except Exception:
            self.skipTest("Model not available")
    
    def test_translate_returns_emojis(self):
        text = "I miss the old days"
        emoji_seq, emotion_vec = self.translator.translate(text)
        
        self.assertIsInstance(emoji_seq, str)
        self.assertGreater(len(emoji_seq), 0)
        self.assertEqual(len(emotion_vec), len(EMOTION_LABELS))
    
    def test_reverse_returns_text(self):
        emoji_seq = "🕰️🌅"
        results = self.translator.reverse(emoji_seq, k=1)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        text, emotions = results[0]
        self.assertIsInstance(text, str)
        self.assertIsInstance(emotions, dict)


if __name__ == '__main__':
    unittest.main()
