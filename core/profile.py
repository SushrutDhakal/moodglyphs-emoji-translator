import json
import numpy as np
from pathlib import Path
from collections import defaultdict


class UserProfile:
    def __init__(self, username, profiles_dir="profiles"):
        self.username = username
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        
        self.emoji_usage = defaultdict(float)
        self.embedding_shift = None
        self.total_interactions = 0
        
        self._load()
    
    def _get_profile_path(self):
        return self.profiles_dir / f"{self.username}.json"
    
    def _load(self):
        path = self._get_profile_path()
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.emoji_usage = defaultdict(float, data.get('emoji_usage', {}))
            self.total_interactions = data.get('total_interactions', 0)
            
            if 'embedding_shift' in data and data['embedding_shift']:
                self.embedding_shift = np.array(data['embedding_shift'])
    
    def save(self):
        data = {
            'username': self.username,
            'emoji_usage': dict(self.emoji_usage),
            'total_interactions': self.total_interactions,
            'embedding_shift': self.embedding_shift.tolist() if self.embedding_shift is not None else None
        }
        
        path = self._get_profile_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def update_emoji_usage(self, emoji_sequence):
        for emoji in emoji_sequence:
            self.emoji_usage[emoji] += 1
        self.total_interactions += 1
        self.save()
    
    def update_embedding_shift(self, selected_emoji_embeddings, text_embedding, alpha=0.3):
        if len(selected_emoji_embeddings) == 0:
            return
        
        avg_selected = np.mean(selected_emoji_embeddings, axis=0)
        new_shift = avg_selected - text_embedding
        
        if self.embedding_shift is None:
            self.embedding_shift = new_shift
        else:
            self.embedding_shift = alpha * self.embedding_shift + (1 - alpha) * new_shift
        
        self.save()
    
    def get_preference_boost(self, emoji):
        if self.total_interactions == 0:
            return 0.0
        
        usage = self.emoji_usage.get(emoji, 0)
        return min(usage / self.total_interactions * 2.0, 1.0)
    
    def get_favorite_emojis(self, k=10):
        sorted_emojis = sorted(
            self.emoji_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emojis[:k]
    
    @classmethod
    def delete_profile(cls, username, profiles_dir="profiles"):
        path = Path(profiles_dir) / f"{username}.json"
        if path.exists():
            path.unlink()
            return True
        return False
