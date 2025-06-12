import os
import shutil
from datasets import load_dataset

# Ustawienia
DATASET_NAME = "librispeech_asr"
CONFIG_NAME = "clean"
SPLIT = "train.100"
SOURCE_DATA = "/software/data/librispeech_asr/LibriSpeech"
LOCAL_DATA = "/root/librispeech_asr/LibriSpeech"
LOCAL_SCRIPT = "/root/local_librispeech_asr"

# Skopiuj dane jeśli trzeba
if not os.path.exists(LOCAL_DATA):
    print(f"Copying dataset from {SOURCE_DATA} to {LOCAL_DATA}...")
    shutil.copytree(SOURCE_DATA, LOCAL_DATA)

# Sklonuj definicję datasetu jeśli trzeba
if not os.path.exists(LOCAL_SCRIPT):
    print(f"Cloning dataset definition to {LOCAL_SCRIPT}...")
    os.system(f"git clone https://huggingface.co/datasets/openslr/librispeech_asr {LOCAL_SCRIPT}")

# Wczytaj dataset lokalnie
print("Loading dataset...")
ds = load_dataset(
    path=LOCAL_SCRIPT,
    name=CONFIG_NAME,
    split=SPLIT,
    data_dir=LOCAL_DATA,
    trust_remote_code=True,
)

print("Loaded dataset:", ds)
