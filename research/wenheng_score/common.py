import os
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(os.environ.get('WENHENG_WORKSPACE', str(Path.cwd() / 'workspace'))).resolve()
SEED = 20261022
DATA_REVISION = '933489269eb9dbf5f6f00f8e8be6ba0ccebc3dbb'
PROCESSED = ROOT / 'data/processed/cfms34-v1'
PRETRAINED = ROOT / 'checkpoints/pretrained/bge-small-zh-v1.5'


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_split(name):
    return [json.loads(line) for line in (PROCESSED / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]


def text_input(row):
    # No scores, IDs, traits or generated feedback enter model features.
    return row['text']


def redact(text):
    text = unicodedata.normalize('NFKC', text).strip()
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
    text = re.sub(r'(?<!\d)1[3-9]\d{9}(?!\d)', '[PHONE]', text)
    return text


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if c.isalnum()).lower()


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
