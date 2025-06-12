from datasets import load_dataset

DATASET_NAME = "librispeech_asr"
CONFIG_NAME = "clean"
SPLIT = "train.100"
LOCAL_PATH = "/software/data/librispeech_asr/LibriSpeech"

print("Loading local LibriSpeech dataset...")
ds = load_dataset(
    path=DATASET_NAME,
    name=CONFIG_NAME,
    split=SPLIT,
    data_dir=LOCAL_PATH,
    trust_remote_code=True,
)
print(ds)
