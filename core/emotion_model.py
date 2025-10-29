import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pathlib import Path


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
    
    def quantize(self, output_path=None):
        """8-bit dynamic quantization for 75% size reduction"""
        self.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            self,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        if output_path:
            torch.save(quantized_model.state_dict(), output_path)
        
        return quantized_model
    
    def export_to_onnx(self, tokenizer, output_path, opset_version=14):
        """Export to ONNX for 2-3x inference speedup"""
        self.eval()
        
        dummy_text = "example text for export"
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        torch.onnx.export(
            self,
            (input_ids, attention_mask),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        return output_path
    
    def get_model_size_mb(self):
        """Calculate model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2
    
    def get_attention_weights(self, input_ids, attention_mask):
        """Extract attention weights for visualization"""
        with torch.no_grad():
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
        return out.attentions


class ContrastiveEmotionModel(nn.Module):
    """Enhanced model with contrastive learning for text-emoji alignment"""
    def __init__(self, model_name, n_emotions=NUM_EMOTIONS, emoji_dim=384):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, n_emotions)
        )
        
        self.emoji_projection = nn.Sequential(
            nn.Linear(hidden, emoji_dim),
            nn.ReLU(),
            nn.LayerNorm(emoji_dim)
        )
    
    def forward(self, input_ids, attention_mask, return_embedding=False):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        
        emotion_logits = self.emotion_head(pooled)
        
        if return_embedding:
            emoji_embedding = self.emoji_projection(pooled)
            return emotion_logits, emoji_embedding
        
        return emotion_logits
    
    def contrastive_loss(self, text_embeddings, emoji_embeddings, temperature=0.07):
        """InfoNCE loss for text-emoji alignment"""
        text_emb_norm = F.normalize(text_embeddings, p=2, dim=1)
        emoji_emb_norm = F.normalize(emoji_embeddings, p=2, dim=1)
        
        similarity_matrix = torch.matmul(text_emb_norm, emoji_emb_norm.T) / temperature
        
        batch_size = text_embeddings.shape[0]
        labels = torch.arange(batch_size, device=text_embeddings.device)
        
        loss_text_to_emoji = F.cross_entropy(similarity_matrix, labels)
        loss_emoji_to_text = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_text_to_emoji + loss_emoji_to_text) / 2


def create_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.7):
    """Knowledge distillation loss for model compression"""
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    
    distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels)
    
    return alpha * distill_loss + (1 - alpha) * hard_loss
