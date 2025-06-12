import os
from datasets import load_dataset

DATASET_NAME = "librispeech_asr"
CONFIG_NAME = "clean"
SPLIT = "train.100"
DATA_DIR = "/software/data/librispeech_asr/LibriSpeech"
SCRIPT_DIR = "/root/local_librispeech_asr"

# Klonowanie tylko raz
if not os.path.exists(SCRIPT_DIR):
    print(f"Cloning dataset definition to {SCRIPT_DIR}...")
    os.system(f"git clone https://huggingface.co/datasets/openslr/librispeech_asr {SCRIPT_DIR}")

# Skrypt datasetu znajduje się głębiej w strukturze repozytorium
DATASET_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "librispeech_asr.py")

if not os.path.exists(DATASET_SCRIPT_PATH):
    raise FileNotFoundError(f"Expected dataset script not found at: {DATASET_SCRIPT_PATH}")

# Wczytanie lokalnego datasetu
print("Loading dataset using local dataset script...")
dataset = load_dataset(
    path=DATASET_SCRIPT_PATH,
    name=CONFIG_NAME,
    split=SPLIT,
    data_dir=DATA_DIR,
    trust_remote_code=True,
)

print("Dataset loaded successfully.")
print(dataset)
