# moodglyphs-emoji-translator
# MoodGlyphs

Text to emoji translator that detects emotions and picks relevant emojis.

## Install

```bash
pip install -r requirements.txt
```

## Setup

First time only:

```bash
python cli.py setup
```

This generates training data, builds the emoji bank, and trains the model. Takes about 5-10 minutes.

## Usage

```bash
python cli.py translate "your text here"
```

Show emotions:
```bash
python cli.py translate "your text" --show-emotions
```

Interactive mode:
```bash
python cli.py interactive
```

## Commands

- `translate TEXT` - convert text to emojis
- `reverse EMOJI` - interpret emoji sequence
- `interactive` - start a session that learns your preferences
- `train` - train model on custom data
- `eval` - evaluate model
- `profile` - view user stats

Run `python cli.py --help` for all options.

## How it works

- Encodes text with a transformer model
- Predicts emotion vectors (30 dimensions)
- Matches to emoji embeddings using cosine similarity
- Uses MMR algorithm to pick diverse emojis
- Learns user preferences over time

## File structure

```
core/          - main code
scripts/       - training and setup
data/          - training data
models/        - trained weights
emoji_bank/    - emoji embeddings
profiles/      - user preferences
```

## Training on your own data

Format as JSONL:
```json
{"id": "1", "text": "example", "labels": {"joy": 0.9, "nostalgia": 0.3}}
```

Then run:
```bash
python cli.py train --train data.jsonl --val val.jsonl
```
