
from datasets import load_dataset
import os

dataset = load_dataset(
            'lambdalabs/pokemon-blip-captions',
            None,
            cache_dir='/root/software/data/pytorch/huggingface/sdxl',
        )

out_loc = 'dataset_pokemon'
if not os.path.exists(out_loc):
    os.mkdir(out_loc)
with open(f'{out_loc}/label.txt', 'w') as f:
    for idx, dt in enumerate(dataset['train']):
        dt['image'].save(f'{out_loc}/{idx}.jpg')
        f.write(dt['text']+'\n')
