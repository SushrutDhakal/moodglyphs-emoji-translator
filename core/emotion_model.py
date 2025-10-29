import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

NUM_EMOTIONS = len(EMOTION_LABELS)


class EmotionModel(nn.Module):
    def __init__(self, model_name, n_emotions=NUM_EMOTIONS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, n_emotions)
        )
        
    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
            
        return self.head(pooled)
    
    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
        return probs


def create_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
