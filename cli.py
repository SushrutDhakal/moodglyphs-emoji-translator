import sys
import click
from pathlib import Path
import os

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from core import Translator, UserProfile
from core.emotion_model import EMOTION_LABELS
from core.utils import top_k_emotions


@click.group()
def cli():
    pass


@cli.command()
@click.argument('text')
@click.option('--user', default='default')
@click.option('--k', default=3)
@click.option('--lambda-param', default=0.7)
@click.option('--no-personalization', is_flag=True)
@click.option('--show-emotions', is_flag=True)
def translate(text, user, k, lambda_param, no_personalization, show_emotions):
    try:
        translator = Translator.load()
        
        emoji_seq, emotion_vec = translator.translate(
            text,
            username=user,
            k=k,
            lambda_param=lambda_param,
            use_personalization=not no_personalization
        )
        
        print(emoji_seq)
        
        if show_emotions:
            top_emotions = top_k_emotions(emotion_vec, EMOTION_LABELS, k=5)
            print("\nDetected emotions:")
            for emotion, score in top_emotions:
                bar = "█" * int(score * 20)
                print(f"  {emotion:15s} {bar} {score:.3f}")
    
    except FileNotFoundError as e:
        click.echo(f"Error: Model not found. Run 'python cli.py setup' first.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('emoji_seq')
@click.option('--k', default=3)
def reverse(emoji_seq, k):
    try:
        translator = Translator.load()
        
        data_path = Path("data/train.jsonl")
        if data_path.exists():
            import json
            texts = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line)
                    texts.append(sample['text'])
            translator.load_dataset_for_reverse(texts[:500])
        
        results = translator.reverse(emoji_seq, k=k)
        
        for i, (text, emotions) in enumerate(results, 1):
            print(f"\n{i}. {text}")
            if emotions:
                emotion_str = ", ".join([f"{k}: {v:.2f}" for k, v in list(emotions.items())[:3]])
                print(f"   Emotions: {emotion_str}")
    
    except FileNotFoundError as e:
        click.echo(f"Error: Model not found.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--user', default='default')
def interactive(user):
    try:
        translator = Translator.load()
        translator.interactive_session(username=user)
    except FileNotFoundError as e:
        click.echo(f"Error: Model not found.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--train', type=click.Path(exists=True), required=True)
@click.option('--val', type=click.Path(exists=True), required=True)
@click.option('--model-name', default='sentence-transformers/all-MiniLM-L6-v2')
@click.option('--output', default='models/emotion_model.pt')
@click.option('--epochs', default=5)
@click.option('--batch-size', default=16)
@click.option('--lr', default=2e-5)
def train(train, val, model_name, output, epochs, batch_size, lr):
    import subprocess
    
    cmd = [
        sys.executable,
        'scripts/train.py',
        '--train', train,
        '--val', val,
        '--model-name', model_name,
        '--output', output,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(lr)
    ]
    
    subprocess.run(cmd)


@cli.command()
@click.option('--model', default='sentence-transformers/all-MiniLM-L6-v2')
@click.option('--output', default='emoji_bank')
def build_emoji_bank(model, output):
    import subprocess
    
    cmd = [
        sys.executable,
        'scripts/build_emoji_bank.py',
        '--model', model,
        '--output', output
    ]
    
    subprocess.run(cmd)


@cli.command()
@click.option('--test', type=click.Path(exists=True), required=True)
@click.option('--output', type=click.Path(), default='eval_results.json')
def eval(test, output):
    import json
    import numpy as np
    from tqdm import tqdm
    
    try:
        translator = Translator.load()
        
        samples = []
        with open(test, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        
        click.echo(f"Evaluating on {len(samples)} samples...")
        
        all_true = []
        all_pred = []
        
        for sample in tqdm(samples):
            text = sample['text']
            true_labels = sample['labels']
            
            _, pred_vec = translator.translate(text, use_personalization=False)
            
            true_vec = np.zeros(len(EMOTION_LABELS))
            for i, emotion in enumerate(EMOTION_LABELS):
                if emotion in true_labels:
                    true_vec[i] = true_labels[emotion]
            
            all_true.append(true_vec)
            all_pred.append(pred_vec)
        
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        
        mae = np.mean(np.abs(all_pred - all_true))
        mse = np.mean((all_pred - all_true) ** 2)
        
        per_emotion_mae = {}
        for i, emotion in enumerate(EMOTION_LABELS):
            per_emotion_mae[emotion] = float(np.mean(np.abs(all_pred[:, i] - all_true[:, i])))
        
        results = {
            'overall_mae': float(mae),
            'overall_mse': float(mse),
            'per_emotion_mae': per_emotion_mae,
            'n_samples': len(samples)
        }
        
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"\nResults:")
        click.echo(f"  MAE: {mae:.4f}")
        click.echo(f"  MSE: {mse:.4f}")
        click.echo(f"  Results saved to {output}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--user', help='Specific user to show')
def profile(user):
    if user:
        p = UserProfile(user)
        click.echo(f"\nProfile: {user}")
        click.echo(f"Total interactions: {p.total_interactions}")
        
        if p.emoji_usage:
            click.echo(f"\nFavorite emojis:")
            for emoji, count in p.get_favorite_emojis(10):
                click.echo(f"  {emoji}  {count:.0f} uses")
        else:
            click.echo("No emoji usage data yet.")
    else:
        profiles_dir = Path("profiles")
        if profiles_dir.exists():
            profile_files = list(profiles_dir.glob("*.json"))
            if profile_files:
                click.echo(f"Found {len(profile_files)} user profiles:")
                for pf in profile_files:
                    username = pf.stem
                    p = UserProfile(username)
                    click.echo(f"  {username}: {p.total_interactions} interactions")
            else:
                click.echo("No user profiles found.")
        else:
            click.echo("No profiles directory found.")


@cli.command()
@click.option('--user', required=True)
@click.confirmation_option(prompt='Are you sure?')
def delete_profile(user):
    if UserProfile.delete_profile(user):
        click.echo(f"Profile '{user}' deleted.")
    else:
        click.echo(f"Profile '{user}' not found.")


@cli.command()
@click.option('--train-size', default=1000)
@click.option('--val-size', default=200)
@click.option('--test-size', default=200)
@click.option('--output-dir', default='data')
def generate_data(train_size, val_size, test_size, output_dir):
    import subprocess
    
    cmd = [
        sys.executable,
        'scripts/generate_synthetic_data.py',
        '--train-size', str(train_size),
        '--val-size', str(val_size),
        '--test-size', str(test_size),
        '--output-dir', output_dir
    ]
    
    subprocess.run(cmd)


@cli.command()
@click.option('--model', default='sentence-transformers/all-MiniLM-L6-v2')
def setup(model):
    import subprocess
    
    click.echo("=== MoodGlyphs Setup ===\n")
    
    click.echo("Step 1/3: Generating synthetic dataset...")
    subprocess.run([
        sys.executable,
        'scripts/generate_synthetic_data.py',
        '--train-size', '1000',
        '--val-size', '200',
        '--test-size', '200',
        '--output-dir', 'data'
    ])
    
    click.echo("\nStep 2/3: Building emoji bank...")
    subprocess.run([
        sys.executable,
        'scripts/build_emoji_bank.py',
        '--model', model,
        '--output', 'emoji_bank'
    ])
    
    click.echo("\nStep 3/3: Training emotion model...")
    subprocess.run([
        sys.executable,
        'scripts/train.py',
        '--train', 'data/train.jsonl',
        '--val', 'data/val.jsonl',
        '--model-name', model,
        '--output', 'models/emotion_model.pt',
        '--epochs', '5',
        '--batch-size', '16'
    ])
    
    click.echo("\n=== Setup Complete! ===")
    click.echo("Try: python cli.py translate \"your text here\"")


if __name__ == '__main__':
    cli()
