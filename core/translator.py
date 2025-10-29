import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from .emotion_model import EmotionModel, NUM_EMOTIONS, EMOTION_LABELS
from .emoji_bank import EmojiBank
from .profile import UserProfile
from .utils import mmr, top_k_emotions


class Translator:
    def __init__(self, encoder, tokenizer, emotion_model, emoji_bank, profiles_dir="profiles"):
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
        text_emb = self._encode_text(text)
        emotion_vec = self._predict_emotion(text)
        
        cand_embs = self.emoji_bank.embs
        
        cand_scores = cosine_similarity(
            text_emb.reshape(1, -1),
            cand_embs
        ).flatten()
        
        if use_personalization:
            profile = UserProfile(username, self.profiles_dir)
            
            for i, meta in enumerate(self.emoji_bank.meta):
                emoji = meta['emoji']
                boost = profile.get_preference_boost(emoji)
                cand_scores[i] += boost * 0.2
            
            if profile.embedding_shift is not None:
                adjusted_emb = text_emb + profile.embedding_shift * 0.3
                adjusted_scores = cosine_similarity(
                    adjusted_emb.reshape(1, -1),
                    cand_embs
                ).flatten()
                cand_scores = 0.7 * cand_scores + 0.3 * adjusted_scores
        
        selected_ids = mmr(text_emb, cand_embs, cand_scores, k=k, lambda_param=lambda_param)
        
        selected_ids_sorted = sorted(
            selected_ids,
            key=lambda i: cand_scores[i],
            reverse=True
        )
        
        emoji_seq = "".join(
            self.emoji_bank.get_emoji_by_index(i)
            for i in selected_ids_sorted
        )
        
        return emoji_seq, emotion_vec
    
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
