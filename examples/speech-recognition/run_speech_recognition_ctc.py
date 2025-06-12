import os
from datasets import load_dataset

DATASET_NAME = "librispeech_asr"
CONFIG_NAME = "clean"
SPLIT = "train.100"
DATA_DIR = "/software/data/librispeech_asr/LibriSpeech"
SCRIPT_PATH = "/root/librispeech_asr.py"

# Pobierz tylko plik datasetu .py (jeśli nie istnieje)
if not os.path.exists(SCRIPT_PATH):
    print(f"Downloading dataset script to {SCRIPT_PATH}...")
    os.system(
        f"wget https://raw.githubusercontent.com/huggingface/datasets/main/datasets/librispeech_asr/librispeech_asr.py -O {SCRIPT_PATH}"
    )

# Użycie lokalnego datasetu
print("Loading dataset using local script...")
dataset = load_dataset(
    path=SCRIPT_PATH,
    name=CONFIG_NAME,
    split=SPLIT,
    data_dir=DATA_DIR,
    trust_remote_code=True,
)

print("Dataset loaded successfully.")
print(dataset)
