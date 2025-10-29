import json
import numpy as np
from pathlib import Path


class EmojiBank:
    def __init__(self, meta, embeddings):
        self.meta = meta
        self.embs = embeddings
        self._emoji_to_idx = {m['emoji']: i for i, m in enumerate(meta)}
        self._search_index = None
        self._use_faiss = False
        
        try:
            import faiss
            self._init_faiss_index()
            self._use_faiss = True
        except ImportError:
            pass
        
    @classmethod
    def load(cls, meta_path, emb_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        embeddings = np.load(emb_path)
        return cls(meta, embeddings)
    
    def save(self, meta_path, emb_path):
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        np.save(emb_path, self.embs)
    
    def find_emoji_index(self, emoji):
        return self._emoji_to_idx.get(emoji, 0)
    
    def get_emoji_by_index(self, idx):
        return self.meta[idx]['emoji']
    
    def get_all_emojis(self):
        return [m['emoji'] for m in self.meta]
    
    def __len__(self):
        return len(self.meta)
    
    def _init_faiss_index(self):
        """Initialize FAISS index for 100x faster search"""
        import faiss
        
        normalized = self.embs / (np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-8)
        self._search_index = faiss.IndexFlatIP(self.embs.shape[1])
        self._search_index.add(normalized.astype('float32'))
    
    def fast_search(self, query_embedding, k=10):
        """Fast emoji search using FAISS (100x faster) or fallback to NumPy"""
        if self._use_faiss and self._search_index is not None:
            query = query_embedding.reshape(1, -1).astype('float32')
            query = query / (np.linalg.norm(query) + 1e-8)
            distances, indices = self._search_index.search(query, k)
            return indices[0], distances[0]
        else:
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            emoji_norm = self.embs / (np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(emoji_norm, query_norm)
            top_k_indices = np.argsort(similarities)[::-1][:k]
            return top_k_indices, similarities[top_k_indices]


EMOJI_BANK_DATA = [
    {"emoji": "🕰️", "desc": "antique clock, passage of time, nostalgia, memories fading"},
    {"emoji": "📼", "desc": "old cassette tape, vintage memories, nostalgia for past"},
    {"emoji": "🌅", "desc": "sunrise over water, memory of peaceful mornings, gentle hope"},
    {"emoji": "🍂", "desc": "autumn leaves, change, melancholy, passage of time"},
    {"emoji": "📷", "desc": "camera capturing moments, memories frozen in time"},
    {"emoji": "🎞️", "desc": "film reel, old memories, cinematic nostalgia"},
    {"emoji": "💭", "desc": "thought bubble, contemplation, yearning, deep thinking"},
    {"emoji": "🌌", "desc": "galaxy stars, vastness, longing, existential wonder"},
    {"emoji": "🌊", "desc": "ocean waves, deep emotions, yearning, endless feeling"},
    {"emoji": "🪶", "desc": "feather floating, lightness, wistful longing, gentle yearning"},
    {"emoji": "🕊️", "desc": "dove flying away, longing for peace, gentle yearning"},
    {"emoji": "🖤", "desc": "black heart, melancholy, deep sadness, dark emotions"},
    {"emoji": "🥀", "desc": "wilted flower, decay, melancholy, faded beauty"},
    {"emoji": "🌧️", "desc": "rain cloud, sadness, melancholy weather, tears"},
    {"emoji": "😔", "desc": "pensive face, sadness, disappointment, melancholy"},
    {"emoji": "💔", "desc": "broken heart, heartbreak, deep sadness, loss"},
    {"emoji": "😢", "desc": "crying face, sadness, tears, emotional pain"},
    {"emoji": "✨", "desc": "sparkles, joy, magic, wonder, delight"},
    {"emoji": "💛", "desc": "yellow heart, warmth, friendship, gentle joy"},
    {"emoji": "🌻", "desc": "sunflower, brightness, optimism, cheerful joy"},
    {"emoji": "😊", "desc": "smiling face, contentment, gentle happiness, warm joy"},
    {"emoji": "🎉", "desc": "party popper, celebration, excitement, pure joy"},
    {"emoji": "💫", "desc": "dizzy stars, excitement, joyful energy, wonder"},
    {"emoji": "🌟", "desc": "glowing star, achievement, pride, bright joy"},
    {"emoji": "🙃", "desc": "upside down face, sarcasm, irony, subtle mockery"},
    {"emoji": "😏", "desc": "smirking face, smugness, sarcasm, knowing irony"},
    {"emoji": "🤨", "desc": "raised eyebrow, skepticism, sarcastic doubt, irony"},
    {"emoji": "💀", "desc": "skull, laughing to death, dark humor, extreme sarcasm"},
    {"emoji": "👁️", "desc": "eye, watching, ironic observation, seeing through things"},
    {"emoji": "😰", "desc": "anxious face with sweat, stress, worry, nervousness"},
    {"emoji": "😟", "desc": "worried face, anxiety, concern, unease"},
    {"emoji": "😬", "desc": "grimacing face, awkwardness, nervous embarrassment, cringe"},
    {"emoji": "🫠", "desc": "melting face, overwhelm, exhaustion, falling apart"},
    {"emoji": "😵‍💫", "desc": "dizzy face, confusion, overwhelmed, spiral"},
    {"emoji": "😌", "desc": "relieved face, peace, calm, contentment"},
    {"emoji": "🧘", "desc": "meditation, calm, peace, centered relief"},
    {"emoji": "☁️", "desc": "cloud, softness, calm, gentle peace"},
    {"emoji": "🌙", "desc": "crescent moon, nighttime calm, peaceful rest"},
    {"emoji": "🌱", "desc": "seedling, new growth, hope, fresh start"},
    {"emoji": "🌈", "desc": "rainbow, hope after storm, beauty, optimism"},
    {"emoji": "🦋", "desc": "butterfly, transformation, hope, beauty emerging"},
    {"emoji": "⭐", "desc": "star, aspiration, hope, reaching for dreams"},
    {"emoji": "🔥", "desc": "fire, passion, determination, burning hope"},
    {"emoji": "😂", "desc": "tears of joy, laughter, amusement, hilarity"},
    {"emoji": "🤣", "desc": "rolling laughing, extreme amusement, uncontrollable laughter"},
    {"emoji": "😆", "desc": "grinning squint, amusement, happy laughter"},
    {"emoji": "🤪", "desc": "zany face, silly playfulness, goofy humor"},
    {"emoji": "😜", "desc": "winking tongue out, playful teasing, fun"},
    {"emoji": "😤", "desc": "face with steam, frustration, annoyed, indignant"},
    {"emoji": "😠", "desc": "angry face, anger, irritation, displeasure"},
    {"emoji": "💢", "desc": "anger symbol, frustration, irritation, vexation"},
    {"emoji": "🔴", "desc": "red circle, stop, anger, intense emotion"},
    {"emoji": "😍", "desc": "heart eyes, adoration, love, admiration"},
    {"emoji": "🤩", "desc": "star struck, awe, amazement, admiration"},
    {"emoji": "👑", "desc": "crown, excellence, admiration, respect"},
    {"emoji": "😳", "desc": "flushed face, embarrassment, awkward, caught off guard"},
    {"emoji": "🫣", "desc": "peeking face, embarrassment, shy awkwardness"},
    {"emoji": "😅", "desc": "sweat smile, nervous laugh, awkward relief"},
    {"emoji": "🤢", "desc": "nauseated face, disgust, revulsion, sick feeling"},
    {"emoji": "😷", "desc": "medical mask, illness, protection, discomfort"},
    {"emoji": "🤮", "desc": "vomiting, extreme disgust, repulsion"},
    {"emoji": "💪", "desc": "flexed bicep, strength, determination, power"},
    {"emoji": "⚡", "desc": "lightning bolt, energy, power, electric determination"},
    {"emoji": "🎯", "desc": "bullseye, focus, goal, determination to succeed"},
    {"emoji": "🤔", "desc": "thinking face, curiosity, contemplation, pondering"},
    {"emoji": "🔍", "desc": "magnifying glass, curiosity, investigation, wonder"},
    {"emoji": "❓", "desc": "question mark, curiosity, wonder, inquiry"},
    {"emoji": "🏆", "desc": "trophy, achievement, pride, success"},
    {"emoji": "🎖️", "desc": "military medal, honor, pride, achievement"},
    {"emoji": "💎", "desc": "gem, precious, pride, valuable achievement"},
    {"emoji": "😨", "desc": "fearful face, fear, dread, terror"},
    {"emoji": "👻", "desc": "ghost, spooky fear, haunting, dread"},
    {"emoji": "🕷️", "desc": "spider, creepy fear, unease, dread"},
    {"emoji": "😲", "desc": "astonished face, surprise, shock, amazement"},
    {"emoji": "🤯", "desc": "exploding head, mind blown, shock, extreme surprise"},
    {"emoji": "😮", "desc": "open mouth, surprise, shock, unexpected"},
    {"emoji": "❤️", "desc": "red heart, love, deep affection, passion"},
    {"emoji": "💕", "desc": "two hearts, love, romance, affection"},
    {"emoji": "🥰", "desc": "smiling hearts, love, adoration, warm affection"},
    {"emoji": "💖", "desc": "sparkling heart, love, affection, excitement"},
    {"emoji": "🤷", "desc": "shrug, resignation, acceptance, indifference"},
    {"emoji": "😑", "desc": "expressionless, resignation, acceptance, neutral"},
    {"emoji": "😶", "desc": "no mouth, speechless resignation, quiet acceptance"},
    {"emoji": "😞", "desc": "disappointed face, despair, letdown, hopeless"},
    {"emoji": "😓", "desc": "downcast sweat, despair, exhausted, hopeless"},
    {"emoji": "💧", "desc": "droplet, tears, sadness, despair"},
    {"emoji": "☮️", "desc": "peace symbol, serenity, harmony, tranquility"},
    {"emoji": "🧘‍♀️", "desc": "woman meditating, inner peace, serenity, calm"},
    {"emoji": "🍃", "desc": "leaves fluttering, gentle peace, nature calm"},
    {"emoji": "🌫️", "desc": "fog, hazy memory, wistful, distant feeling"},
    {"emoji": "🎐", "desc": "wind chime, gentle wistfulness, soft memories"},
    {"emoji": "🪻", "desc": "lavender, nostalgic wistfulness, gentle longing"},
]
