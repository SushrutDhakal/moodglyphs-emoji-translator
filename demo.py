import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    if not Path("models/emotion_model.pt").exists():
        subprocess.run([sys.executable, "cli.py", "setup"])
    
    from core import Translator
    from core.emotion_model import EMOTION_LABELS
    from core.utils import top_k_emotions
    
    translator = Translator.load()
    
    while True:
        try:
            text = input("\nEnter text to translate (or 'quit' to exit): ").strip()
            
            if text.lower() in ['quit', 'q', 'exit']:
                break
            
            if not text:
                continue
            
            emoji_seq, emotion_vec = translator.translate(text)
            
            print(f"\nEmojis: {emoji_seq}")
            
            top_emotions = top_k_emotions(emotion_vec, EMOTION_LABELS, k=5)
            print("\nTop emotions:")
            for emotion, score in top_emotions:
                bar = "█" * int(score * 20)
                print(f"  {emotion:15s} {bar} {score:.3f}")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


if __name__ == '__main__':
    main()

