from datasets import load_dataset
import os

dataset = load_dataset(
            'lambdalabs/pokemon-blip-captions',
            None,
        )

dir = 'dataset_pokemon1'
os.mkdir(dir)
with open(f'{dir}/label.txt', 'w') as f:
    for idx, dt in enumerate(dataset['train']):
        dt['image'].save(f'{dir}/{idx}.jpg')
        f.write(dt['text'] + '\n')

