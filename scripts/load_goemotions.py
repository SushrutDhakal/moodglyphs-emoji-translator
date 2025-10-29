import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

GOEMOTIONS_EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def load_goemotions_data(split='train'):
    data_dir = script_dir / 'goemotions_data'
    file_path = data_dir / f'{split}.tsv'
    
    examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            text = parts[0]
            label_ids = [int(x) for x in parts[1].split(',') if x.strip()]
            comment_id = parts[2]
            
            emotion_vector = [0.0] * len(GOEMOTIONS_EMOTIONS)
            for idx in label_ids:
                if 0 <= idx < len(GOEMOTIONS_EMOTIONS):
                    emotion_vector[idx] = 1.0
            
            examples.append({
                'text': text,
                'emotions': emotion_vector,
                'id': comment_id
            })
    
    return examples

def convert_to_jsonl(output_dir):
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'dev', 'test']:
        examples = load_goemotions_data(split)
        
        output_file = output_dir / f'{split}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                json.dump({'text': ex['text'], 'emotions': ex['emotions']}, f)
                f.write('\n')

if __name__ == '__main__':
    data_dir = script_dir / 'data_goemotions'
    convert_to_jsonl(data_dir)

