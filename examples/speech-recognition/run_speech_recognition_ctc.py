import os
import sys
import shutil
import datasets
from datasets import load_dataset
import subprocess

# Informacje diagnostyczne
print("=== Environment Diagnostics ===")
print(f"datasets version: {datasets.__version__}")
print(f"datasets file: {datasets.__file__}")
print(f"sys.executable: {sys.executable}")
print(f"sys.path: {sys.path}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '(not set)')}")
print("which python:", subprocess.getoutput("which python"))
print()

# Ścieżki do cache, które chcemy usunąć
hf_home = os.path.expanduser("~/.cache/huggingface")
paths_to_clear = [
    f"{hf_home}/datasets/modules/datasets/librispeech_asr",
    f"{hf_home}/modules/datasets_modules/datasets/librispeech_asr",
    f"{hf_home}/datasets/downloads",
]

print("=== Clearing HuggingFace cache ===")
for path in paths_to_clear:
    if os.path.exists(path):
        print(f"Removing: {path}")
        shutil.rmtree(path)
    else:
        print(f"Not found: {path}")
print()

# Próba wczytania datasetu lokalnie
print("=== Attempting to load local LibriSpeech dataset ===")
DATASET_NAME = "librispeech_asr"
CONFIG_NAME = "clean"
SPLIT = "train.100"
LOCAL_PATH = "/software/data/librispeech_asr/LibriSpeech"

print("== Permissions Check ==")
print("Can read data_dir:", os.access(LOCAL_PATH, os.R_OK))
print("Can execute data_dir:", os.access(LOCAL_PATH, os.X_OK))

print(f"DATASET_NAME = {DATASET_NAME}")
print(f"CONFIG_NAME  = {CONFIG_NAME}")
print(f"SPLIT        = {SPLIT}")
print(f"LOCAL_PATH   = {LOCAL_PATH}")
print()

try:
    ds = load_dataset(
        path=DATASET_NAME,
        name=CONFIG_NAME,
        split=SPLIT,
        data_dir=LOCAL_PATH,
        trust_remote_code=True,
    )
    print("\n=== Dataset loaded successfully ===")
    print(ds)
except Exception as e:
    print("\n!!! Dataset loading failed !!!")
    print(str(e))
