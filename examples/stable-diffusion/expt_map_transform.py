from datasets import load_dataset
import pdb
tr = pdb.set_trace
from torchvision import transforms
import torch, random

image_column = 'image'
resolution = 1024
center_crop = True
train_resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
train_crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
train_flip = transforms.RandomHorizontalFlip(p=1.0)
train_transforms = transforms.Compose([transforms.ToTensor(), transforms.RandomErasing(), transforms.Normalize([0.5], [0.5])])

def preprocess_train(examples):
        print('in preproc')
        images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if center_crop:
                y1 = max(0, int(round((image.height - resolution) / 2.0)))
                x1 = max(0, int(round((image.width - resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (resolution, resolution))
                image = crop(image, y1, x1, h, w)
            if True and random.random() < 0.5:
                # flip
                image = train_flip(image)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        #import pdb; pdb.set_trace()
        return examples


def mapper(batch):
    print('in mapper', flush=True)
    images = batch["pixel_values"]
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    with torch.no_grad():
        model_input = pixel_values * 2
    return {"model_input": model_input.cpu()}

dataset = load_dataset(
            'lambdalabs/pokemon-blip-captions',
            None,
            cache_dir='/root/software/data/pytorch/huggingface/sdxl',
        )
tr_dt = dataset['train'].select(range(64)).with_transform(preprocess_train)
tr_dt = tr_dt.map(mapper, batched=True, batch_size=16, load_from_cache_file=False)
print('map done')

def collate_fn(examples):
    #import pdb; pdb.set_trace()
    model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples])
    pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
    original_sizes = [example["original_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]
    #prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
    #pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

    return {
        "model_input": model_input,
        #"prompt_embeds": prompt_embeds,
        #"pooled_prompt_embeds": pooled_prompt_embeds,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        "pixel_values": pixel_values
    }

train_dataloader = torch.utils.data.DataLoader(
        tr_dt,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=16,
        num_workers=0,
    )


for epoch in range(5):
    inp_sum = []
    inp_sum_1 = []
    for i, dt in enumerate(train_dataloader):
        inp_sum += [i.item() for i in dt['model_input'].sum((1,2,3))]
        inp_sum_1 += [i.item() for i in dt['pixel_values'].sum((1,2,3))]
        print(i, epoch)
    print(sorted(inp_sum))
    print(sorted(inp_sum_1))
    print('*'*50)
