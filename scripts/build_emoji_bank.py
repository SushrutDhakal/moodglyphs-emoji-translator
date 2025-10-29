import sys
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.emoji_bank import EmojiBank, EMOJI_BANK_DATA


def build_emoji_embeddings(model_name, output_dir):
    print(f"Loading model: {model_name}...")
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
    
    print(f"Encoding {len(EMOJI_BANK_DATA)} emojis...")
    
    embeddings = []
    for i, item in enumerate(EMOJI_BANK_DATA):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(EMOJI_BANK_DATA)}")
        
        emb = embed(item['desc'])
        embeddings.append(emb)
    
    embeddings_array = np.vstack(embeddings)
    
    emoji_bank = EmojiBank(EMOJI_BANK_DATA, embeddings_array)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    emoji_bank.save(
        output_path / "emoji_bank.json",
        output_path / "emoji_bank.npy"
    )
    
    print(f"\nEmoji bank saved to {output_path}")
    print(f"  Total emojis: {len(emoji_bank)}")
    print(f"  Embedding dimension: {embeddings_array.shape[1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='emoji_bank'
    )
    
    args = parser.parse_args()
    
    build_emoji_embeddings(args.model, args.output)


if __name__ == '__main__':
    main()
