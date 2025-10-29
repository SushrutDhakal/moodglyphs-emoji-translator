import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.profile import UserProfile


class TestUserProfile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_profile(self):
        profile = UserProfile("testuser", self.temp_dir)
        
        self.assertEqual(profile.username, "testuser")
        self.assertEqual(profile.total_interactions, 0)
        self.assertEqual(len(profile.emoji_usage), 0)
    
    def test_update_emoji_usage(self):
        profile = UserProfile("testuser", self.temp_dir)
        
        profile.update_emoji_usage("🔥💯")
        
        self.assertEqual(profile.emoji_usage["🔥"], 1)
        self.assertEqual(profile.emoji_usage["💯"], 1)
        self.assertEqual(profile.total_interactions, 1)
        
        profile.update_emoji_usage("🔥")
        self.assertEqual(profile.emoji_usage["🔥"], 2)
        self.assertEqual(profile.total_interactions, 2)
    
    def test_save_and_load(self):
        profile1 = UserProfile("testuser", self.temp_dir)
        profile1.update_emoji_usage("🎉🎊")
        profile1.save()
        
        profile2 = UserProfile("testuser", self.temp_dir)
        
        self.assertEqual(profile2.total_interactions, 1)
        self.assertEqual(profile2.emoji_usage["🎉"], 1)
        self.assertEqual(profile2.emoji_usage["🎊"], 1)
    
    def test_preference_boost(self):
        profile = UserProfile("testuser", self.temp_dir)
        
        for _ in range(5):
            profile.update_emoji_usage("🔥")
        
        for _ in range(10):
            profile.update_emoji_usage("💯")
        
        boost_fire = profile.get_preference_boost("🔥")
        self.assertGreater(boost_fire, 0.5)
        self.assertLess(boost_fire, 0.7)
        
        boost_hundred = profile.get_preference_boost("💯")
        self.assertEqual(boost_hundred, 1.0)
        
        boost_unused = profile.get_preference_boost("👻")
        self.assertEqual(boost_unused, 0.0)
    
    def test_update_embedding_shift(self):
        profile = UserProfile("testuser", self.temp_dir)
        
        selected_embs = np.random.randn(3, 128)
        text_emb = np.random.randn(128)
        
        profile.update_embedding_shift(selected_embs, text_emb)
        
        self.assertIsNotNone(profile.embedding_shift)
        self.assertEqual(profile.embedding_shift.shape, (128,))
    
    def test_delete_profile(self):
        profile = UserProfile("testuser", self.temp_dir)
        profile.save()
        
        profile_path = Path(self.temp_dir) / "testuser.json"
        self.assertTrue(profile_path.exists())
        
        deleted = UserProfile.delete_profile("testuser", self.temp_dir)
        self.assertTrue(deleted)
        self.assertFalse(profile_path.exists())
        
        deleted = UserProfile.delete_profile("nonexistent", self.temp_dir)
        self.assertFalse(deleted)
    
    def test_get_favorite_emojis(self):
        profile = UserProfile("testuser", self.temp_dir)
        
        profile.update_emoji_usage("🔥" * 10)
        profile.update_emoji_usage("💯" * 5)
        profile.update_emoji_usage("🎉" * 3)
        
        favorites = profile.get_favorite_emojis(k=2)
        
        self.assertEqual(len(favorites), 2)
        self.assertEqual(favorites[0][0], "🔥")
        self.assertEqual(favorites[1][0], "💯")


if __name__ == '__main__':
    unittest.main()
