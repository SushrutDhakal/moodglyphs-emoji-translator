import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.emotion_model import EmotionModel, create_tokenizer, EMOTION_LABELS, NUM_EMOTIONS


class EmotionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        
        if 'emotions' in sample:
            emotion_vector = np.array(sample['emotions'], dtype=np.float32)
        else:
            labels = sample['labels']
            emotion_vector = np.zeros(NUM_EMOTIONS, dtype=np.float32)
            for i, emotion_name in enumerate(EMOTION_LABELS):
                if emotion_name in labels:
                    emotion_vector[i] = labels[emotion_name]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.from_numpy(emotion_vector)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    mae = np.mean(np.abs(all_preds - all_labels), axis=0)
    mse = np.mean((all_preds - all_labels) ** 2, axis=0)
    
    avg_loss = total_loss / len(dataloader)
    avg_mae = np.mean(mae)
    avg_mse = np.mean(mse)
    
    return avg_loss, avg_mae, avg_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--model-name', type=str, 
                       default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--output', type=str, default='models/emotion_model.pt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup-steps', type=int, default=100)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = create_tokenizer(args.model_name)
    model = EmotionModel(args.model_name, n_emotions=NUM_EMOTIONS)
    model.to(device)
    
    train_dataset = EmotionDataset(args.train, tokenizer)
    val_dataset = EmotionDataset(args.val, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_mae, val_mse = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model to {output_path}")
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
