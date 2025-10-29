import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.emotion_model import EMOTION_LABELS


TEMPLATES = {
    'nostalgia': [
        "I miss those {adjective} {time_period}.",
        "Remember when we used to {activity}? Those were the days.",
        "There's something about {object} that takes me back.",
        "I wish we could go back to {time_period}.",
        "{time_period} feel like a different lifetime now.",
    ],
    'joy': [
        "I'm so {adjective} about {event}!",
        "This is absolutely {adjective}!",
        "I can't stop smiling about {event}.",
        "Everything feels so {adjective} right now.",
        "What a {adjective} day this has been!",
    ],
    'sadness': [
        "I feel so {adjective} about {event}.",
        "It's hard to {action} when everything feels so {adjective}.",
        "Why does everything have to be so {adjective}?",
        "I can't shake this {adjective} feeling.",
        "Nothing seems to {action} anymore.",
    ],
    'anxiety': [
        "I'm so worried about {event}.",
        "What if {event} goes wrong?",
        "I can't stop thinking about {worry}.",
        "My mind won't stop {action}.",
        "I feel so {adjective} about everything.",
    ],
    'sarcasm': [
        "Oh great, {event}. Just what I needed.",
        "Wow, {event}. I'm so impressed.",
        "Because {event} is exactly what makes sense.",
        "Sure, {event}. That'll definitely work.",
        "Nothing says quality like {event}.",
    ],
    'hope': [
        "Maybe {event} will be different.",
        "I'm hopeful that {event} will work out.",
        "Things are starting to look {adjective}.",
        "I believe {event} is possible.",
        "Tomorrow could bring {event}.",
    ],
}

FILLERS = {
    'adjective': [
        'beautiful', 'quiet', 'peaceful', 'chaotic', 'perfect', 'simple',
        'empty', 'heavy', 'bright', 'dark', 'overwhelming', 'gentle'
    ],
    'time_period': [
        'summer nights', 'childhood days', 'college years', 'those mornings',
        'winter evenings', 'lazy afternoons', 'early days', 'simpler times'
    ],
    'activity': [
        'stay up talking', 'explore the city', 'laugh at nothing',
        'dream about the future', 'waste time together', 'make plans'
    ],
    'object': [
        'old songs', 'faded photographs', 'handwritten letters',
        'that place', 'these memories', 'forgotten dreams'
    ],
    'event': [
        'this', 'everything', 'the meeting', 'the news', 'tomorrow',
        'the project', 'the situation', 'what happened', 'the plan'
    ],
    'action': [
        'focus', 'sleep', 'move forward', 'care', 'matter', 'help'
    ],
    'worry': [
        'what they think', 'tomorrow', 'failing', 'the future',
        'making mistakes', 'letting people down'
    ],
}


def fill_template(template):
    text = template
    for key, values in FILLERS.items():
        if '{' + key + '}' in text:
            text = text.replace('{' + key + '}', random.choice(values))
    return text


def generate_emotion_vector(primary_emotions, intensity_range=(0.6, 0.9)):
    vector = {}
    
    for emotion, base_intensity in primary_emotions.items():
        intensity = random.uniform(
            max(base_intensity - 0.2, 0.0),
            min(base_intensity + 0.1, 1.0)
        )
        vector[emotion] = round(intensity, 3)
    
    all_emotions = set(EMOTION_LABELS)
    remaining = all_emotions - set(primary_emotions.keys())
    
    n_secondary = random.randint(2, 4)
    secondary = random.sample(list(remaining), min(n_secondary, len(remaining)))
    
    for emotion in secondary:
        vector[emotion] = round(random.uniform(0.05, 0.3), 3)
    
    return vector


EMOTION_PROFILES = {
    'nostalgia': {'nostalgia': 0.9, 'melancholy': 0.6, 'wistfulness': 0.7, 'yearning': 0.5},
    'joy': {'joy': 0.9, 'amusement': 0.7, 'hope': 0.6},
    'sadness': {'sadness': 0.8, 'melancholy': 0.7, 'despair': 0.5},
    'anxiety': {'anxiety': 0.8, 'fear': 0.6, 'worry': 0.7},
    'sarcasm': {'sarcasm': 0.9, 'irony': 0.8, 'amusement': 0.5},
    'hope': {'hope': 0.85, 'relief': 0.5, 'calm': 0.4, 'determination': 0.6},
    'melancholy': {'melancholy': 0.8, 'sadness': 0.6, 'nostalgia': 0.5, 'wistfulness': 0.7},
    'anger': {'anger': 0.85, 'frustration': 0.7, 'disgust': 0.4},
    'relief': {'relief': 0.85, 'calm': 0.7, 'peace': 0.6, 'joy': 0.4},
    'awe': {'awe': 0.9, 'wonder': 0.8, 'admiration': 0.6, 'surprise': 0.5},
}


def generate_dataset(n_samples, output_path):
    samples = []
    
    for i in range(n_samples):
        category = random.choice(list(TEMPLATES.keys()))
        template = random.choice(TEMPLATES[category])
        text = fill_template(template)
        
        emotion_profile = EMOTION_PROFILES.get(category, {category: 0.8})
        labels = generate_emotion_vector(emotion_profile)
        
        sample = {
            'id': f'syn_{i:05d}',
            'text': text,
            'labels': labels
        }
        
        samples.append(sample)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(samples)} samples to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=1000)
    parser.add_argument('--val-size', type=int, default=200)
    parser.add_argument('--test-size', type=int, default=200)
    parser.add_argument('--output-dir', type=str, default='data')
    
    args = parser.parse_args()
    
    print("Generating synthetic emotion dataset...")
    
    generate_dataset(args.train_size, f"{args.output_dir}/train.jsonl")
    generate_dataset(args.val_size, f"{args.output_dir}/val.jsonl")
    generate_dataset(args.test_size, f"{args.output_dir}/test.jsonl")
    
    print(f"\nDataset generated in {args.output_dir}/")
    print(f"  Train: {args.train_size} samples")
    print(f"  Val: {args.val_size} samples")
    print(f"  Test: {args.test_size} samples")


if __name__ == '__main__':
    main()
