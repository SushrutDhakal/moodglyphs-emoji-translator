import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.emotion_model import EMOTION_LABELS

TEMPLATES = {
    'joy': [
        "I'm so happy about {event}!",
        "This is wonderful! {event}",
        "I'm thrilled that {event}",
        "What amazing news about {event}!",
        "I couldn't be happier!",
    ],
    'sadness': [
        "I feel so sad about {event}",
        "This makes me cry",
        "I'm heartbroken about {event}",
        "It's devastating that {event}",
        "I can't stop feeling down",
    ],
    'anger': [
        "I'm so angry about {event}!",
        "This makes me furious!",
        "I'm outraged that {event}",
        "I can't believe how mad this makes me",
        "This is infuriating!",
    ],
    'fear': [
        "I'm terrified of {event}",
        "This is really scary",
        "I'm afraid that {event}",
        "I'm frightened by {event}",
        "This scares me so much",
    ],
    'surprise': [
        "Wow, I didn't expect {event}!",
        "I'm shocked that {event}",
        "What a surprise!",
        "I never saw {event} coming",
        "I'm amazed by {event}",
    ],
    'disgust': [
        "This is disgusting",
        "I'm repulsed by {event}",
        "That's gross and {event}",
        "I can't stand {event}",
        "This makes me sick",
    ],
    'nostalgia': [
        "I miss those days when {event}",
        "Remember when {event}? Those were the days",
        "I wish we could go back to {event}",
        "There's something about {event} that takes me back",
        "Those memories of {event} feel like yesterday",
    ],
    'melancholy': [
        "There's a sadness in {event}",
        "I feel a deep sorrow about {event}",
        "Everything feels heavy and {event}",
        "There's a quiet sadness to {event}",
        "I'm filled with melancholy thinking about {event}",
    ],
    'yearning': [
        "I long for {event}",
        "I deeply desire {event}",
        "I'm yearning for {event}",
        "I crave {event} so much",
        "My heart aches for {event}",
    ],
    'longing': [
        "I wish for {event}",
        "I keep hoping for {event}",
        "I miss {event} desperately",
        "I long to have {event} back",
        "There's an emptiness without {event}",
    ],
    'wistfulness': [
        "I think wistfully about {event}",
        "There's a bittersweet feeling to {event}",
        "I remember {event} with a gentle sadness",
        "I feel wistful when I think of {event}",
        "There's a tender longing for {event}",
    ],
    'sarcasm': [
        "Oh great, {event}. Just perfect.",
        "Wow, {event}. I'm so impressed.",
        "Sure, {event}. That makes total sense.",
        "Because {event} is exactly what we needed.",
        "Oh wonderful, {event}. Just wonderful.",
    ],
    'irony': [
        "Isn't it ironic that {event}?",
        "Of course {event} happened. Of course.",
        "Naturally {event}. How fitting.",
        "The irony of {event} isn't lost on me",
        "How perfectly ironic that {event}",
    ],
    'playfulness': [
        "Let's have fun with {event}!",
        "This is playful and {event}",
        "I'm in a playful mood about {event}",
        "Let's be silly and {event}",
        "I feel lighthearted about {event}",
    ],
    'amusement': [
        "This is hilarious!",
        "I'm so amused by {event}",
        "This cracks me up!",
        "I can't stop laughing at {event}",
        "How funny that {event}!",
    ],
    'humor': [
        "That's really funny!",
        "The humor in {event} is perfect",
        "I find {event} quite amusing",
        "There's something humorous about {event}",
        "I appreciate the comedy of {event}",
    ],
    'anxiety': [
        "I'm so worried about {event}",
        "What if {event} goes wrong?",
        "I'm anxious that {event}",
        "I can't stop worrying about {event}",
        "This gives me so much anxiety",
    ],
    'relief': [
        "I'm so relieved that {event}",
        "Thank goodness {event}",
        "What a relief about {event}!",
        "I can finally breathe knowing {event}",
        "The relief I feel about {event} is immense",
    ],
    'calm': [
        "I feel peaceful about {event}",
        "Everything is calm and {event}",
        "I'm at peace with {event}",
        "There's a tranquility to {event}",
        "I feel serene about {event}",
    ],
    'peace': [
        "I'm at peace with {event}",
        "There's a peacefulness to {event}",
        "I feel harmonious about {event}",
        "Everything feels balanced with {event}",
        "I'm content and peaceful",
    ],
    'serenity': [
        "I feel serene about {event}",
        "There's a serenity in {event}",
        "I'm tranquil and {event}",
        "Everything feels still and {event}",
        "I'm in a state of serenity",
    ],
    'hope': [
        "I'm hopeful that {event}",
        "Maybe {event} will work out",
        "I believe {event} is possible",
        "There's hope for {event}",
        "I'm optimistic about {event}",
    ],
    'despair': [
        "I've lost all hope about {event}",
        "I'm in despair over {event}",
        "Everything feels hopeless",
        "I can't see a way out of {event}",
        "There's no hope for {event}",
    ],
    'resignation': [
        "I've accepted that {event}",
        "I'm resigned to {event}",
        "I guess {event} is just how it is",
        "I've given up on changing {event}",
        "I'm passively accepting {event}",
    ],
    'determination': [
        "I'm determined to {event}",
        "I won't give up on {event}",
        "I'm committed to {event}",
        "Nothing will stop me from {event}",
        "I'm resolved to {event}",
    ],
    'admiration': [
        "I admire {event} so much",
        "I'm in awe of {event}",
        "I have such respect for {event}",
        "I look up to {event}",
        "I'm impressed by {event}",
    ],
    'awe': [
        "I'm in awe of {event}",
        "This is breathtaking!",
        "I'm stunned by the magnificence of {event}",
        "I'm overwhelmed by {event}",
        "The grandeur of {event} amazes me",
    ],
    'wonder': [
        "I wonder about {event}",
        "I'm curious about {event}",
        "There's something magical about {event}",
        "I'm filled with wonder at {event}",
        "I marvel at {event}",
    ],
    'embarrassment': [
        "I'm so embarrassed about {event}",
        "I feel awkward about {event}",
        "This is mortifying!",
        "I'm ashamed that {event}",
        "I want to hide after {event}",
    ],
    'pride': [
        "I'm so proud of {event}",
        "I feel accomplished about {event}",
        "I take pride in {event}",
        "I'm honored by {event}",
        "I feel great about {event}",
    ],
}

FILLERS = {
    'event': [
        'this', 'everything', 'what happened', 'the situation', 'it all',
        'the outcome', 'this moment', 'the news', 'the result', 'what occurred',
        'the experience', 'what transpired', 'this development', 'these events',
    ],
}

CORRELATIONS = {
    'joy': {'joy': 1.0, 'amusement': 0.5, 'hope': 0.4, 'playfulness': 0.3},
    'sadness': {'sadness': 1.0, 'melancholy': 0.7, 'despair': 0.4, 'longing': 0.3},
    'anger': {'anger': 1.0, 'disgust': 0.5, 'frustration': 0.6},
    'fear': {'fear': 1.0, 'anxiety': 0.7, 'worry': 0.5},
    'surprise': {'surprise': 1.0, 'awe': 0.3, 'wonder': 0.3},
    'disgust': {'disgust': 1.0, 'anger': 0.4},
    'nostalgia': {'nostalgia': 1.0, 'melancholy': 0.5, 'yearning': 0.6, 'wistfulness': 0.7},
    'melancholy': {'melancholy': 1.0, 'sadness': 0.6, 'wistfulness': 0.5},
    'yearning': {'yearning': 1.0, 'longing': 0.8, 'nostalgia': 0.5},
    'longing': {'longing': 1.0, 'yearning': 0.8, 'sadness': 0.4},
    'wistfulness': {'wistfulness': 1.0, 'nostalgia': 0.7, 'melancholy': 0.5},
    'sarcasm': {'sarcasm': 1.0, 'irony': 0.7, 'humor': 0.3},
    'irony': {'irony': 1.0, 'sarcasm': 0.6},
    'playfulness': {'playfulness': 1.0, 'joy': 0.5, 'amusement': 0.6},
    'amusement': {'amusement': 1.0, 'joy': 0.6, 'humor': 0.7},
    'humor': {'humor': 1.0, 'amusement': 0.7, 'playfulness': 0.4},
    'anxiety': {'anxiety': 1.0, 'fear': 0.6, 'worry': 0.7},
    'relief': {'relief': 1.0, 'calm': 0.6, 'peace': 0.5},
    'calm': {'calm': 1.0, 'peace': 0.8, 'serenity': 0.7},
    'peace': {'peace': 1.0, 'calm': 0.8, 'serenity': 0.8},
    'serenity': {'serenity': 1.0, 'peace': 0.8, 'calm': 0.7},
    'hope': {'hope': 1.0, 'optimism': 0.7, 'determination': 0.4},
    'despair': {'despair': 1.0, 'sadness': 0.7, 'hopelessness': 0.9},
    'resignation': {'resignation': 1.0, 'acceptance': 0.6, 'melancholy': 0.4},
    'determination': {'determination': 1.0, 'resolve': 0.8, 'hope': 0.5},
    'admiration': {'admiration': 1.0, 'respect': 0.8, 'awe': 0.5},
    'awe': {'awe': 1.0, 'wonder': 0.7, 'admiration': 0.6},
    'wonder': {'wonder': 1.0, 'curiosity': 0.6, 'awe': 0.5},
    'embarrassment': {'embarrassment': 1.0, 'shame': 0.7, 'awkwardness': 0.6},
    'pride': {'pride': 1.0, 'satisfaction': 0.7, 'joy': 0.5},
}


def generate_example(primary_emotion):
    template = random.choice(TEMPLATES[primary_emotion])
    
    text = template
    for placeholder in FILLERS:
        if f'{{{placeholder}}}' in text:
            text = text.replace(f'{{{placeholder}}}', random.choice(FILLERS[placeholder]))
    
    labels = {emotion: 0.0 for emotion in EMOTION_LABELS}
    
    if primary_emotion in CORRELATIONS:
        for emotion, score in CORRELATIONS[primary_emotion].items():
            if emotion in labels:
                labels[emotion] = score
    else:
        labels[primary_emotion] = 1.0
    
    labels = {k: round(v + random.uniform(-0.1, 0.1), 2) for k, v in labels.items() if v > 0}
    labels = {k: max(0.0, min(1.0, v)) for k, v in labels.items()}
    
    return {'text': text, 'labels': labels}


def generate_dataset(num_samples, emotions):
    examples = []
    for _ in range(num_samples):
        emotion = random.choice(emotions)
        examples.append(generate_example(emotion))
    return examples


def main():
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    covered_emotions = list(TEMPLATES.keys())
    
    print(f"Generating data for {len(covered_emotions)} emotions...")
    print(f"Emotions: {', '.join(covered_emotions)}")
    
    train_data = generate_dataset(1000, covered_emotions)
    val_data = generate_dataset(200, covered_emotions)
    test_data = generate_dataset(200, covered_emotions)
    
    with open(output_dir / 'train.jsonl', 'w') as f:
        for i, ex in enumerate(train_data, 1):
            f.write(json.dumps({'id': str(i), **ex}) + '\n')
    
    with open(output_dir / 'val.jsonl', 'w') as f:
        for i, ex in enumerate(val_data, 1):
            f.write(json.dumps({'id': str(i), **ex}) + '\n')
    
    with open(output_dir / 'test.jsonl', 'w') as f:
        for i, ex in enumerate(test_data, 1):
            f.write(json.dumps({'id': str(i), **ex}) + '\n')
    
    print(f"\nGenerated datasets:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")


if __name__ == '__main__':
    main()

