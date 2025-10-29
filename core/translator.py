import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import hashlib
import time
from collections import OrderedDict, deque
import psutil
import os

from .emotion_model import EmotionModel, NUM_EMOTIONS, EMOTION_LABELS
from .emoji_bank import EmojiBank
from .profile import UserProfile
from .utils import mmr, top_k_emotions


class PerformanceMonitor:
    """Comprehensive performance tracking"""
    def __init__(self):
        self.stage_timings = {}
        self.memory_snapshots = []
        self.request_latencies = deque(maxlen=1000)
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.baseline_memory = self.get_current_memory()
    
    def get_current_memory(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def record_stage(self, stage_name, duration_ms):
        """Record timing for a pipeline stage"""
        if stage_name not in self.stage_timings:
            self.stage_timings[stage_name] = []
        self.stage_timings[stage_name].append(duration_ms)
    
    def record_request(self, latency_ms):
        """Record completed request latency"""
        self.request_latencies.append(latency_ms)
        
        current_memory = self.get_current_memory()
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def get_stage_stats(self, stage_name):
        """Get statistics for a specific stage"""
        if stage_name not in self.stage_timings:
            return None
        
        timings = self.stage_timings[stage_name]
        return {
            'mean_ms': np.mean(timings),
            'median_ms': np.median(timings),
            'p95_ms': np.percentile(timings, 95),
            'p99_ms': np.percentile(timings, 99),
            'count': len(timings)
        }
    
    def get_latency_percentiles(self):
        """Get request latency percentiles"""
        if not self.request_latencies:
            return {}
        
        latencies = list(self.request_latencies)
        return {
            'p50_ms': np.percentile(latencies, 50),
            'p90_ms': np.percentile(latencies, 90),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'mean_ms': np.mean(latencies),
            'total_requests': len(latencies)
        }
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        return {
            'current_mb': self.get_current_memory(),
            'baseline_mb': self.baseline_memory,
            'peak_mb': self.peak_memory,
            'delta_mb': self.get_current_memory() - self.baseline_memory
        }
    
    def get_throughput(self, window_seconds=60):
        """Calculate requests per second"""
        if not self.request_latencies:
            return 0.0
        
        recent_count = len(self.request_latencies)
        uptime = time.time() - self.start_time
        
        if uptime < window_seconds:
            return recent_count / uptime if uptime > 0 else 0
        
        return recent_count / window_seconds
    
    def get_comprehensive_stats(self):
        """Get all performance statistics"""
        return {
            'latency': self.get_latency_percentiles(),
            'memory': self.get_memory_stats(),
            'throughput_rps': self.get_throughput(),
            'uptime_seconds': time.time() - self.start_time,
            'stage_breakdown': {
                stage: self.get_stage_stats(stage)
                for stage in self.stage_timings.keys()
            }
        }


class TranslationCache:
    """LRU cache with TTL for translations"""
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = OrderedDict()
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def get(self, text):
        key = hashlib.md5(text.encode()).hexdigest()
        
        if key in self.cache:
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                self.misses += 1
                return None
            
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, text, value):
        key = hashlib.md5(text.encode()).hexdigest()
        
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            oldest_key = next(iter(self.timestamps))
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()


class Translator:
    def __init__(self, encoder, tokenizer, emotion_model, emoji_bank, profiles_dir="profiles", use_cache=True, enable_monitoring=True):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.emotion_model = emotion_model
        self.emoji_bank = emoji_bank
        self.profiles_dir = profiles_dir
        
        self.encoder.eval()
        self.emotion_model.eval()
        
        self.dataset_texts = []
        self.dataset_text_embs = None
        self.dataset_emotions = []
        
        self.cache = TranslationCache() if use_cache else None
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        self.stats = {'total_requests': 0, 'cache_hits': 0, 'avg_latency_ms': 0}
    
    @classmethod
    def load(cls, model_name="sentence-transformers/all-MiniLM-L6-v2", 
             emotion_weights="models/emotion_model.pt",
             emoji_bank_dir="emoji_bank",
             profiles_dir="profiles"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        
        emotion_model = EmotionModel(model_name, n_emotions=NUM_EMOTIONS)
        
        weights_path = Path(emotion_weights)
        if weights_path.exists():
            emotion_model.load_state_dict(
                torch.load(emotion_weights, map_location='cpu')
            )
        else:
            print(f"Warning: No weights found at {emotion_weights}")
        
        emoji_bank = EmojiBank.load(
            f"{emoji_bank_dir}/emoji_bank.json",
            f"{emoji_bank_dir}/emoji_bank.npy"
        )
        
        return cls(encoder, tokenizer, emotion_model, emoji_bank, profiles_dir)
    
    def _encode_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        
        with torch.no_grad():
            out = self.encoder(**inputs, return_dict=True)
        
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            emb = out.last_hidden_state[:, 0]
        
        return emb.cpu().numpy().squeeze()
    
    def _predict_emotion(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        
        with torch.no_grad():
            logits = self.emotion_model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            probs = torch.sigmoid(logits)
        
        return probs.cpu().numpy().squeeze()
    
    def translate(self, text, username="default", k=3, lambda_param=0.7, 
                  use_personalization=True):
        start_time = time.perf_counter()
        self.stats['total_requests'] += 1
        
        if self.cache:
            cached = self.cache.get(text + str(k) + str(lambda_param))
            if cached:
                self.stats['cache_hits'] += 1
                return cached
        
        stage_start = time.perf_counter()
        text_emb = self._encode_text(text)
        if self.monitor:
            self.monitor.record_stage('text_encoding', (time.perf_counter() - stage_start) * 1000)
        
        stage_start = time.perf_counter()
        emotion_vec = self._predict_emotion(text)
        if self.monitor:
            self.monitor.record_stage('emotion_prediction', (time.perf_counter() - stage_start) * 1000)
        
        stage_start = time.perf_counter()
        top_n = min(100, len(self.emoji_bank))
        indices, scores = self.emoji_bank.fast_search(text_emb, k=top_n)
        if self.monitor:
            self.monitor.record_stage('emoji_search', (time.perf_counter() - stage_start) * 1000)
        
        cand_scores = np.zeros(len(self.emoji_bank))
        cand_scores[indices] = scores
        
        if use_personalization:
            stage_start = time.perf_counter()
            profile = UserProfile(username, self.profiles_dir)
            
            for idx in indices:
                emoji = self.emoji_bank.meta[idx]['emoji']
                boost = profile.get_preference_boost(emoji)
                cand_scores[idx] += boost * 0.2
            
            if profile.embedding_shift is not None:
                adjusted_emb = text_emb + profile.embedding_shift * 0.3
                adjusted_indices, adjusted_scores = self.emoji_bank.fast_search(adjusted_emb, k=top_n)
                for idx, score in zip(adjusted_indices, adjusted_scores):
                    cand_scores[idx] = 0.7 * cand_scores[idx] + 0.3 * score
            
            if self.monitor:
                self.monitor.record_stage('personalization', (time.perf_counter() - stage_start) * 1000)
        
        stage_start = time.perf_counter()
        selected_ids = mmr(text_emb, self.emoji_bank.embs, cand_scores, k=k, lambda_param=lambda_param)
        
        selected_ids_sorted = sorted(
            selected_ids,
            key=lambda i: cand_scores[i],
            reverse=True
        )
        
        emoji_seq = "".join(
            self.emoji_bank.get_emoji_by_index(i)
            for i in selected_ids_sorted
        )
        if self.monitor:
            self.monitor.record_stage('mmr_selection', (time.perf_counter() - stage_start) * 1000)
        
        result = (emoji_seq, emotion_vec)
        
        if self.cache:
            self.cache.set(text + str(k) + str(lambda_param), result)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.stats['avg_latency_ms'] = (
            (self.stats['avg_latency_ms'] * (self.stats['total_requests'] - 1) + latency_ms) /
            self.stats['total_requests']
        )
        
        if self.monitor:
            self.monitor.record_request(latency_ms)
        
        return result
    
    def translate_batch(self, texts, k=3, lambda_param=0.7):
        """Translate multiple texts efficiently"""
        results = []
        
        encodings = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        
        with torch.no_grad():
            encoder_out = self.encoder(**encodings, return_dict=True)
            if hasattr(encoder_out, 'pooler_output') and encoder_out.pooler_output is not None:
                embeddings = encoder_out.pooler_output
            else:
                embeddings = encoder_out.last_hidden_state[:, 0]
            
            emotion_logits = self.emotion_model(
                encodings['input_ids'],
                encodings['attention_mask']
            )
            emotion_probs = torch.sigmoid(emotion_logits)
        
        embeddings_np = embeddings.cpu().numpy()
        emotions_np = emotion_probs.cpu().numpy()
        
        for i, text in enumerate(texts):
            text_emb = embeddings_np[i]
            emotion_vec = emotions_np[i]
            
            indices, scores = self.emoji_bank.fast_search(text_emb, k=min(100, len(self.emoji_bank)))
            
            cand_scores = np.zeros(len(self.emoji_bank))
            cand_scores[indices] = scores
            
            selected_ids = mmr(text_emb, self.emoji_bank.embs, cand_scores, k=k, lambda_param=lambda_param)
            selected_ids_sorted = sorted(selected_ids, key=lambda idx: cand_scores[idx], reverse=True)
            
            emoji_seq = "".join(self.emoji_bank.get_emoji_by_index(idx) for idx in selected_ids_sorted)
            results.append((emoji_seq, emotion_vec))
        
        return results
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        stats = self.stats.copy()
        
        if self.cache:
            stats['cache_hit_rate'] = self.cache.hits / (self.cache.hits + self.cache.misses) if (self.cache.hits + self.cache.misses) > 0 else 0
            stats['cache_size'] = len(self.cache.cache)
            stats['cache_hits'] = self.cache.hits
            stats['cache_misses'] = self.cache.misses
        
        stats['using_faiss'] = self.emoji_bank._use_faiss
        stats['emoji_bank_size'] = len(self.emoji_bank)
        stats['model_size_mb'] = self.emotion_model.get_model_size_mb()
        
        if self.monitor:
            comprehensive = self.monitor.get_comprehensive_stats()
            stats.update(comprehensive)
        
        return stats
    
    def reverse(self, emoji_seq, k=3):
        emojis = list(emoji_seq)
        
        indices = []
        descriptions = []
        
        for emoji in emojis:
            idx = self.emoji_bank.find_emoji_index(emoji)
            indices.append(idx)
            descriptions.append(self.emoji_bank.meta[idx]['desc'])
        
        if not indices:
            return [("No valid emojis found.", {})]
        
        emoji_embs = self.emoji_bank.embs[indices]
        avg_emb = np.mean(emoji_embs, axis=0)
        
        if self.dataset_text_embs is not None and len(self.dataset_texts) > 0:
            sims = cosine_similarity(
                avg_emb.reshape(1, -1),
                self.dataset_text_embs
            ).flatten()
            
            top_indices = np.argsort(sims)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                text = self.dataset_texts[idx]
                emotions = self.dataset_emotions[idx]
                results.append((text, emotions))
            
            return results
        else:
            combined_desc = ", ".join(descriptions)
            
            emotion_vec = self._predict_emotion(combined_desc)
            top_emotions = top_k_emotions(emotion_vec, EMOTION_LABELS, k=3)
            
            emotion_str = ", ".join([f"{name} ({score:.2f})" for name, score in top_emotions])
            
            interpretation = (
                f"Emotional tone: {emotion_str}. "
                f"Themes: {combined_desc}"
            )
            
            return [(interpretation, dict(top_emotions))]
    
    def load_dataset_for_reverse(self, texts, emotions_list=None):
        self.dataset_texts = texts
        self.dataset_emotions = emotions_list if emotions_list else [{}] * len(texts)
        
        embeddings = []
        for text in texts:
            emb = self._encode_text(text)
            embeddings.append(emb)
        
        self.dataset_text_embs = np.vstack(embeddings)
    
    def interactive_session(self, username="default"):
        profile = UserProfile(username, self.profiles_dir)
        print(f"\n=== MoodGlyphs Interactive Session ===")
        print(f"User: {username}")
        print(f"Total interactions: {profile.total_interactions}")
        
        if profile.emoji_usage:
            print(f"Favorite emojis: {' '.join([e for e, _ in profile.get_favorite_emojis(5)])}")
        
        print("\nCommands: [a]ccept [e]dit [s]how emotions [q]uit")
        print("=" * 40)
        
        while True:
            try:
                text = input("\nEnter text (or 'quit' to exit): ").strip()
                
                if text.lower() in ['quit', 'q', 'exit']:
                    break
                
                if not text:
                    continue
                
                emoji_seq, emotion_vec = self.translate(text, username=username)
                
                print(f"\nGenerated: {emoji_seq}")
                
                top_emotions = top_k_emotions(emotion_vec, EMOTION_LABELS, k=5)
                
                while True:
                    action = input("\n[a]ccept [e]dit [s]how emotions [r]etry: ").strip().lower()
                    
                    if action == 'a':
                        profile.update_emoji_usage(emoji_seq)
                        
                        selected_indices = [
                            self.emoji_bank.find_emoji_index(e)
                            for e in emoji_seq
                        ]
                        selected_embs = self.emoji_bank.embs[selected_indices]
                        text_emb = self._encode_text(text)
                        profile.update_embedding_shift(selected_embs, text_emb)
                        
                        print("Saved to profile!")
                        break
                    
                    elif action == 'e':
                        new_seq = input("Enter your emoji sequence: ").strip()
                        if new_seq:
                            profile.update_emoji_usage(new_seq)
                            
                            try:
                                selected_indices = [
                                    self.emoji_bank.find_emoji_index(e)
                                    for e in new_seq
                                ]
                                selected_embs = self.emoji_bank.embs[selected_indices]
                                text_emb = self._encode_text(text)
                                profile.update_embedding_shift(selected_embs, text_emb)
                                print("Custom sequence saved!")
                            except:
                                print("Some emojis not in bank, but sequence saved.")
                        break
                    
                    elif action == 's':
                        print("\nDetected emotions:")
                        for emotion, score in top_emotions:
                            bar = "█" * int(score * 20)
                            print(f"  {emotion:15s} {bar} {score:.3f}")
                    
                    elif action == 'r':
                        emoji_seq, emotion_vec = self.translate(
                            text,
                            username=username,
                            lambda_param=np.random.uniform(0.5, 0.9)
                        )
                        print(f"\nGenerated: {emoji_seq}")
                    
                    else:
                        print("Invalid command. Use [a]ccept, [e]dit, [s]how, or [r]etry")
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"\nGoodbye! Total interactions: {profile.total_interactions}")
