# MoodGlyphs

Text-to-emoji translator using emotion detection and semantic matching.

## Quick Start

Clone and setup:

```bash
git clone https://github.com/SushrutDhakal/moodglyphs-emoji-translator.git
cd moodglyphs-emoji-translator
pip install -r requirements.txt
python cli.py setup
```

The setup downloads and trains on the GoEmotions dataset (43K human-labeled Reddit comments), builds an emoji bank with 1816 emojis, and trains the emotion detection model. Takes 5-10 minutes.

Test with your own input:

```bash
python demo.py
```

Or translate directly:

```bash
python cli.py translate "I'm so excited about this!"
python cli.py translate "Feeling grateful today" --show-emotions
```

## Commands

- `translate TEXT` - convert text to emojis
- `reverse EMOJI` - interpret emoji sequence
- `interactive` - interactive session with personalization
- `train` - train model on custom data
- `eval` - evaluate model performance
- `profile` - view usage statistics

Run `python cli.py --help` for all options.

## How it Works

1. **Text Encoding**: Transformer model (MiniLM-L6-v2) encodes input text
2. **Emotion Detection**: Trained classifier predicts 28 emotion scores (GoEmotions labels)
3. **Emoji Matching**: Cosine similarity between text and 1816 emoji embeddings
4. **Diversity Selection**: MMR algorithm picks relevant but diverse emojis
5. **Personalization**: Learns user preferences over time (optional)

## Tech Stack

- **Model**: Fine-tuned sentence-transformers/all-MiniLM-L6-v2
- **Training Data**: [GoEmotions](https://www.kaggle.com/datasets/debarshichanda/goemotions) (43K labeled Reddit comments)
- **Emoji Bank**: [Full Emoji Dataset](https://www.kaggle.com/datasets/subinium/emojiimage-dataset) (1816 emojis)
- **Framework**: PyTorch + HuggingFace Transformers

## Model Performance

**Emotion Detection Accuracy** (Validation MAE: 0.0525)

| Input | Top Emotion | Confidence |
|-------|-------------|------------|
| "This is terrifying!" | Fear | 0.808 |
| "I'm so angry" | Anger | 0.659 |
| "I'm super happy" | Joy | 0.832 |
| "Feeling really sad" | Sadness | 0.840 |
| "This makes me so grateful" | Gratitude | 0.839 |
| "This is hilarious" | Amusement | 0.855 |

**Content-Aware Translation**
- "play soccer" → ⚽
- "ate pizza" → 🍕
- "feeling confused" → 🤔

## Custom Training

Format your data as JSONL with emotion vectors:

```json
{"text": "example text", "emotions": [0.0, 0.8, 0.0, ...]}
```

Train:

```bash
python scripts/train.py --train data.jsonl --val val.jsonl --epochs 5
```

## Project Structure

```
core/               - emotion model, translator, emoji bank
scripts/            - training and data processing
  ├── train.py
  ├── download_goemotions.py
  └── load_goemotions.py
models/             - trained model weights
emoji_bank/         - emoji embeddings (1816 emojis)
profiles/           - user personalization data
```
