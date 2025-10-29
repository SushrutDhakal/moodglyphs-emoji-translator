import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import requests
from pathlib import Path

script_dir = Path(__file__).parent.parent
data_dir = script_dir / "goemotions_data"
data_dir.mkdir(exist_ok=True)

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()
    
    api.dataset_download_files(
        'debarshichanda/goemotions',
        path=str(data_dir),
        unzip=True
    )
    
except Exception as e:
    urls = {
        'train.tsv': 'https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv',
        'dev.tsv': 'https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv',
        'test.tsv': 'https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv',
        'emotions.txt': 'https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt'
    }
    
    for filename, url in urls.items():
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            output_path = data_dir / filename
            with open(output_path, 'wb') as f:
                f.write(response.content)
        except Exception as err:
            print(f"Error downloading {filename}: {err}")

