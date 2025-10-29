import csv
import json
import numpy as np
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from core.emoji_bank import EmojiBank

csv_path = script_dir.parent / "full_emoji.csv"
output_dir = script_dir / "emoji_bank"
output_dir.mkdir(exist_ok=True)

print("Loading model...")
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def embed(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    
    with torch.no_grad():
        out = model(**inputs, return_dict=True)
    
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        emb = out.pooler_output
    else:
        emb = out.last_hidden_state[:, 0]
    
    return emb.squeeze().cpu().numpy()

emoji_data = []
print(f"Reading CSV from {csv_path}...")

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        unicode_str = row['unicode']
        name = row['name']
        
        code_points = unicode_str.replace('U+', '').split()
        emoji_char = ''.join(chr(int(cp, 16)) for cp in code_points)
        
        description = f"{name}. This emoji represents {name.replace('-', ' ')}."
        
        emoji_data.append({
            'emoji': emoji_char,
            'desc': description
        })

print(f"Loaded {len(emoji_data)} emojis")
print("Generating embeddings...")

embeddings = []
for i, item in enumerate(emoji_data):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(emoji_data)}")
    
    emb = embed(item['desc'])
    embeddings.append(emb)

embeddings_array = np.vstack(embeddings)

emoji_bank = EmojiBank(emoji_data, embeddings_array)

emoji_bank.save(
    output_dir / "emoji_bank.json",
    output_dir / "emoji_bank.npy"
)

print(f"\nEmoji bank saved to {output_dir}")
print(f"  Total emojis: {len(emoji_bank)}")
print(f"  Embedding dimension: {embeddings_array.shape[1]}")

