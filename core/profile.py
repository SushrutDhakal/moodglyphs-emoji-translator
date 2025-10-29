import json
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict
import time


class BayesianPreferences:
    """Bayesian user preference modeling with Dirichlet priors"""
    def __init__(self, prior_strength=1.0):
        self.prior_strength = prior_strength
        self.emoji_counts = defaultdict(int)
        self.emotion_emoji_counts = defaultdict(lambda: defaultdict(int))
        self.total_count = 0
    
    def update(self, emoji, emotion_label=None):
        self.emoji_counts[emoji] += 1
        self.total_count += 1
        
        if emotion_label:
            self.emotion_emoji_counts[emotion_label][emoji] += 1
    
    def get_probability(self, emoji, emotion_label=None):
        if emotion_label:
            count = self.emotion_emoji_counts[emotion_label][emoji]
            total = sum(self.emotion_emoji_counts[emotion_label].values())
            return (count + self.prior_strength) / (total + self.prior_strength * 100)
        else:
            count = self.emoji_counts[emoji]
            return (count + self.prior_strength) / (self.total_count + self.prior_strength * 100)
    
    def to_dict(self):
        return {
            'emoji_counts': dict(self.emoji_counts),
            'emotion_emoji_counts': {
                emotion: dict(emojis)
                for emotion, emojis in self.emotion_emoji_counts.items()
            },
            'total_count': self.total_count
        }
    
    def from_dict(self, data):
        self.emoji_counts = defaultdict(int, data.get('emoji_counts', {}))
        self.emotion_emoji_counts = defaultdict(
            lambda: defaultdict(int),
            {
                emotion: defaultdict(int, emojis)
                for emotion, emojis in data.get('emotion_emoji_counts', {}).items()
            }
        )
        self.total_count = data.get('total_count', 0)


class ExponentialMovingAverage:
    """Track user embedding preferences with EMA"""
    def __init__(self, alpha=0.1, dimension=384):
        self.alpha = alpha
        self.embedding = None
        self.dimension = dimension
        self.update_count = 0
    
    def update(self, new_embedding):
        if self.embedding is None:
            self.embedding = new_embedding.copy()
        else:
            self.embedding = self.alpha * new_embedding + (1 - self.alpha) * self.embedding
        
        self.update_count += 1
        
        if self.update_count > 100:
            self.alpha = max(0.01, self.alpha * 0.95)
    
    def get_personalized_query(self, query_embedding, blend=0.2):
        if self.embedding is None or np.linalg.norm(self.embedding) < 1e-6:
            return query_embedding
        
        personalized = (1 - blend) * query_embedding + blend * self.embedding
        return personalized / (np.linalg.norm(personalized) + 1e-8)
    
    def to_dict(self):
        return {
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'alpha': self.alpha,
            'update_count': self.update_count
        }
    
    def from_dict(self, data):
        if data.get('embedding'):
            self.embedding = np.array(data['embedding'])
        self.alpha = data.get('alpha', 0.1)
        self.update_count = data.get('update_count', 0)


class UserProfile:
    def __init__(self, username, profiles_dir="profiles"):
        self.username = username
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        
        self.emoji_usage = defaultdict(float)
        self.embedding_shift = None
        self.total_interactions = 0
        
        self.bayesian = BayesianPreferences()
        self.ema = ExponentialMovingAverage()
        
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
            
            if 'bayesian' in data:
                self.bayesian.from_dict(data['bayesian'])
            
            if 'ema' in data:
                self.ema.from_dict(data['ema'])
    
    def save(self):
        data = {
            'username': self.username,
            'emoji_usage': dict(self.emoji_usage),
            'total_interactions': self.total_interactions,
            'embedding_shift': self.embedding_shift.tolist() if self.embedding_shift is not None else None,
            'bayesian': self.bayesian.to_dict(),
            'ema': self.ema.to_dict()
        }
        
        path = self._get_profile_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def update_emoji_usage(self, emoji_sequence, emotion_label=None, text_embedding=None):
        for emoji in emoji_sequence:
            self.emoji_usage[emoji] += 1
            self.bayesian.update(emoji, emotion_label)
        
        if text_embedding is not None:
            self.ema.update(text_embedding)
        
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
    
    def get_preference_boost(self, emoji, emotion_label=None):
        if self.total_interactions == 0:
            return 0.0
        
        basic_boost = self.emoji_usage.get(emoji, 0) / self.total_interactions * 2.0
        bayesian_prob = self.bayesian.get_probability(emoji, emotion_label)
        
        return min(basic_boost * 0.6 + bayesian_prob * 0.4, 1.0)
    
    def get_personalized_embedding(self, query_embedding, blend=0.2):
        """Get personalized query embedding using EMA"""
        return self.ema.get_personalized_query(query_embedding, blend)
    
    def get_favorite_emojis(self, k=10):
        sorted_emojis = sorted(
            self.emoji_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emojis[:k]
    
    def get_stats(self):
        """Get user profile statistics"""
        return {
            'username': self.username,
            'total_interactions': self.total_interactions,
            'unique_emojis': len(self.emoji_usage),
            'favorite_emojis': self.get_favorite_emojis(5),
            'bayesian_total': self.bayesian.total_count,
            'ema_updates': self.ema.update_count,
            'has_embedding_shift': self.embedding_shift is not None
        }
    
    @classmethod
    def delete_profile(cls, username, profiles_dir="profiles"):
        path = Path(profiles_dir) / f"{username}.json"
        if path.exists():
            path.unlink()
            return True
        return False
