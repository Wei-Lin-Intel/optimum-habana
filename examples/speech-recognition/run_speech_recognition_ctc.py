from datasets import load_dataset

DATASET_SCRIPT = "/software/data/librispeech_asr/librispeech_asr.py"
DATA_DIR = "/software/data/librispeech_asr/LibriSpeech"
CONFIG_NAME = "clean"
SPLIT = "train.100"

dataset = load_dataset(
    path=DATASET_SCRIPT,
    name=CONFIG_NAME,
    split=SPLIT,
    data_dir=DATA_DIR,
    trust_remote_code=True,    
)

print(dataset)