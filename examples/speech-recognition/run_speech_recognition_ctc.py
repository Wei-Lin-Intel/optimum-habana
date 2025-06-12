import os
from datasets import load_dataset

DATASET_NAME = "librispeech_asr"
CONFIG_NAME = "clean"
SPLIT = "train.100"
DATA_DIR = "/software/data/librispeech_asr/LibriSpeech"
SCRIPT_DIR = "/root/local_librispeech_asr"

# Klonowanie definicji lokalnej jeśli nie istnieje
if not os.path.exists(SCRIPT_DIR):
    print(f"Cloning dataset definition to {SCRIPT_DIR}...")
    os.system(f"git clone https://huggingface.co/datasets/openslr/librispeech_asr {SCRIPT_DIR}")

# Wczytanie datasetu
print("Loading dataset from local path...")
dataset = load_dataset(
    path=SCRIPT_DIR,
    name=CONFIG_NAME,
    split=SPLIT,
    data_dir=DATA_DIR,
    trust_remote_code=True,
)

print("Dataset loaded successfully.")
print(dataset)
