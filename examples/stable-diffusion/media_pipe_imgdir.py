'''


this is dummy "encoding"
[ord(i) for i in x[1]]


TODO right now the data pipe might be static in its cropping. make it random

'''
import numpy as np
import time
import os
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import Dataset
from datasets import Dataset as DatasetHF

from transformers.trainer_pt_utils import DistributedSampler

import torch
from optimum.utils import logging
from torch.distributed import get_rank, get_world_size

logger = logging.get_logger(__name__)


try:
    from habana_frameworks.mediapipe import fn
    from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
    from habana_frameworks.mediapipe.backend.operator_specs import schema
    from habana_frameworks.mediapipe.media_types import dtype, ftype, imgtype, randomCropType, readerOutType
    from habana_frameworks.mediapipe.mediapipe import MediaPipe
    from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
    from habana_frameworks.mediapipe.operators.reader_nodes.read_image_from_dir import get_max_file
    from habana_frameworks.torch.hpu import get_device_name
    from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
except ImportError:
    pass



class PokemonDataset(Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.loaded_dataset = self.load_dt()
        self.column_names = ['image', 'text']

    def load_dt(self):
        labels = open(f'{self.dataset_dir}/label.txt').readlines()
        dct = {'image': [], 'text': []}
        for item in sorted([i for i in os.listdir(self.dataset_dir) if 'txt' not in i], key=lambda x : int(x.split('.')[0])):
            key = int(item.split('.')[0])
            dct['image'] += [f'{self.dataset_dir}/{item}']
            dct['text'] += [labels[key]]
            if len(dct['image']) >= 16*6*8: # TODO get rid of later
                break
        return dct

    def map(self, fn):
        pass

    def __len__(self):
        return len(self.loaded_dataset['text'])

    def __getitem__(self, idx):
        return {'image': self.loaded_dataset['image'][idx], 'text': self.loaded_dataset['text'][idx]}


def get_dataset_for_pipeline(img_dir):
    def create_gen(torch_dataset):
        def gen():
            for ex in torch_dataset:
                yield ex
        return gen
    dt = PokemonDataset(img_dir) #TODO PokemonDataset is not needed. just create a generator
    return DatasetHF.from_generator(create_gen(dt))


class read_image_text_from_dataset(MediaReaderNode):
    """
    Class defining read image/text from directory node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr):
        super().__init__(name, guid, device, inputs, params, cparams, node_attr)
        self.meta_dtype = params["label_dtype"]  # TODO sasarkar.. clean up unnecesary args, add args that are hardcoded etc
        self.dataset = params["dataset"]

        self.dataset_image = []
        self.dataset_text = [] # TODO can be removed later
        self.dataset_prompt_embeds = []
        self.dataset_pooled_prompt_embeds = []
        self.dataset_original_sizes = []
        self.dataset_crop_top_lefts = []
        for k in self.dataset:
            self.dataset_image += [k['image']]
            self.dataset_text += [k['text']]
            self.dataset_prompt_embeds += [k['prompt_embeds']]
            self.dataset_pooled_prompt_embeds += [k['pooled_prompt_embeds']]
            self.dataset_original_sizes += [k['original_sizes']]
            self.dataset_crop_top_lefts += [k['crop_top_lefts']]

        self.dataset_image = np.array(self.dataset_image)
        self.dataset_prompt_embeds = np.array(self.dataset_prompt_embeds, dtype=np.float32)
        self.dataset_pooled_prompt_embeds = np.array(self.dataset_pooled_prompt_embeds, dtype=np.float32)
        self.dataset_original_sizes = np.array(self.dataset_original_sizes, dtype=np.uint32)
        self.dataset_crop_top_lefts = np.array(self.dataset_crop_top_lefts, dtype=np.uint32)
        #import pdb; pdb.set_trace()
        self.epoch = 0
        self.batch_sampler = params["batch_sampler"]

        #self.num_imgs_slice = len(SDXLMediaPipe.batch_sampler.sampler)
        #self.num_batches_slice = len(SDXLMediaPipe.batch_sampler)
        self.num_imgs_slice = len(self.batch_sampler.sampler)
        self.num_batches_slice = len(self.batch_sampler)

        #print(self.num_imgs_slice)
        #print(self.num_batches_slice)
        #print('XXXXX here....')

        logger.info("Finding largest file ...")
        self.max_file = max(self.dataset['image'], key= lambda x : len(x))
        #self.max_file_length = max([len(i) for i in self.dataset['image']])

        self.max_label_len = len(max(self.dataset['text'], key= lambda x : len(x))) # TODO remove

        #logger.info(f"The largest file is {self.max_file_length}.")


    def set_params(self, params):
        self.batch_size = params.batch_size

    def gen_output_info(self):
        out_info = []
        o = opnode_tensor_info(dtype.NDT, np.array([self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        o = opnode_tensor_info(
            self.meta_dtype, np.array([2048, 77, self.batch_size], dtype=np.uint32), ""
        ) # TODO sarkar: whats 77? max tokenized len of labels ? remove hardcode
        out_info.append(o)

        o = opnode_tensor_info(
            self.meta_dtype, np.array([1280, self.batch_size], dtype=np.uint32), ""
        )  # TODO sasarkar: hardcoded shapes. remove later
        out_info.append(o)


        o = opnode_tensor_info(
            'uint32', np.array([2, self.batch_size], dtype=np.uint32), ""
        )
        out_info.append(o)
        #import pdb; pdb.set_trace()
        o = opnode_tensor_info(
            'uint32', np.array([2, self.batch_size], dtype=np.uint32), ""
        )
        out_info.append(o)

        #o = opnode_tensor_info(
        #    'uint32', np.array([self.max_label_len, self.batch_size], dtype=np.uint32), ""
        #)
        #out_info.append(o) # TODO can remove this later. this is text label
        return out_info

    def get_largest_file(self):
        return self.max_file

    def get_media_output_type(self):
        return readerOutType.FILE_LIST

    def __len__(self):
        #import pdb; pdb.set_trace()
        return self.num_batches_slice
        #return len(self.dataset)

    def __iter__(self):
        self.iter_loc = 0
        self.epoch += 1
        try:
            self.batch_sampler.sampler.set_epoch(self.epoch) # Without this dist sampler will create same batches every epoch
        except:
            pass
        self.batch_sampler_iter = iter(self.batch_sampler)
        return self

    def __next__(self):
        t0 = time.time()
        #print('enter reader')
        if self.iter_loc > (self.num_imgs_slice - 1):
            raise StopIteration

        data_idx = next(self.batch_sampler_iter)
        if True:
            img_list = [i for i in self.dataset_image[data_idx]]
            prompt_embeds_np = self.dataset_prompt_embeds[data_idx]
            pooled_prompt_embeds_np = self.dataset_pooled_prompt_embeds[data_idx]
            original_sizes = self.dataset_original_sizes[data_idx]
            crop_top_lefts = self.dataset_crop_top_lefts[data_idx]
        else: # TODO get rid of the else section

            #try:
            #    print(f'{data_idx} , {get_rank()}. ,,,,,,,,, idx')
            #except:
            #    print(f'{data_idx} , {0}. ,,,,,,,,, idx')
            data = [self.dataset[i] for i in data_idx]
            # each item of data has keys: dict_keys(['image', 'text', 'prompt_embeds', 'pooled_prompt_embeds'])

            img_list = [d['image'] for d in data]
            prompt_embeds_np = np.array([d['prompt_embeds'] for d in data], dtype=np.float32)
            pooled_prompt_embeds_np = np.array([d['pooled_prompt_embeds'] for d in data], dtype=np.float32)
            original_sizes = np.array([d['original_sizes'] for d in data], dtype=np.uint32)
            crop_top_lefts = np.array([d['crop_top_lefts'] for d in data], dtype=np.uint32)

            #import pdb; pdb.set_trace()

            #text_label = np.zeros([self.batch_size, self.max_label_len], dtype=np.uint32)
            #for idxx, d in enumerate(data):
            #    text_label[idxx,:len(d['text'])] = np.array([ord(kk) for kk in d['text']], dtype=np.uint32)

            #return img_list, prompt_embeds_np, pooled_prompt_embeds_np, original_sizes, crop_top_lefts, text_label

        self.iter_loc = self.iter_loc + self.batch_size
        #print('exit reader', time.time()-t0)
        return img_list, prompt_embeds_np, pooled_prompt_embeds_np, original_sizes, crop_top_lefts


read_image_text_from_dataset_params = {
    "label_dtype": dtype.FLOAT32,
    "dataset": None,
    'batch_sampler': []
}
#name, guid, device, inputs, params, cparams, node_attr
'''
 def add_operator(self,
                     name,
                     guid,
                     min_inputs,
                     max_inputs,
                     input_keys,
                     num_outputs,
                     params,
                     cparams,
                     op_class,
                     dtype):
        """
'''
schema.add_operator(
    "SDXLDataReader",
    None,
    0,
    0,
    [],
    5, #5, #6
    read_image_text_from_dataset_params,
    None,
    read_image_text_from_dataset,
    dtype.NDT,
)
op_class = fn.operator_add("SDXLDataReader", False)
op_class.__module__ = fn.__name__
setattr(fn, "SDXLDataReader", op_class)

class RandomFlipFunction(media_function):
    """
    Class to randomly generate input for RandomFlip media node.

    """

    def __init__(self, params):
        """
        :params params: random_flip_func specific params.
                        shape: output shape
                        dtype: output data type
                        seed: seed to be used
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed']
        self.rng = np.random.default_rng(self.seed)
    def __call__(self):
        """
        :returns : randomly generated binary output per image.
        """
        probabilities = [1.0 - 0.5, 0.5]
        random_flips = self.rng.choice([0, 1], p=probabilities, size=self.np_shape)
        random_flips = np.array(random_flips, dtype=self.np_dtype)
        #print(random_flips)
        return random_flips

class SDXLMediaPipe(MediaPipe):
    """
    Class defining SDXL media pipe:
        read data --> image decoding (include crop and resize) --> crop mirror normalize

    Original set of PyTorch transformations:
        aspect ratio preserving resize -> center crop -> normalize

    """

    #batch_sampler = None
    instance_count = 0

    def __init__(self, dataset=None, sampler=None, batch_size=512, drop_last=False, queue_depth=5):
        self.device = get_device_name()
        self.dataset = dataset

        
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)

        self.image_size = 512 # TODO .. hardcoded size

        pipe_name = "{}:{}".format(self.__class__.__name__, SDXLMediaPipe.instance_count)
        pipe_name = str(pipe_name)

        super(SDXLMediaPipe, self).__init__(
            device=self.device, batch_size=batch_size, prefetch_depth=queue_depth, pipe_name=pipe_name
        )

        self.input = fn.SDXLDataReader(label_dtype=dtype.FLOAT32, dataset=self.dataset, batch_sampler=self.batch_sampler)
        def_output_image_size = [self.image_size, self.image_size]
        res_pp_filter = ftype.BI_LINEAR
        self.decode = fn.ImageDecoder(
            device=self.device,
            output_format=imgtype.RGB_P,
            #random_crop_type=randomCropType.CENTER_CROP,
            resize=def_output_image_size,
            resampling_mode=res_pp_filter,
        )
        normalize_mean = np.array([255/2, 255/2, 255/2]).astype(np.float32)
        normalize_std = 1 / (np.array([255/2, 255/2, 255/2]).astype(np.float32))
        norm_mean = fn.MediaConst(data=normalize_mean, shape=[1, 1, 3], dtype=dtype.FLOAT32)
        norm_std = fn.MediaConst(data=normalize_std, shape=[1, 1, 3], dtype=dtype.FLOAT32)
        self.cmn = fn.CropMirrorNorm(
            crop_w=self.image_size,
            crop_h=self.image_size,
            crop_pos_x=0,
            crop_pos_y=0,
            crop_d=0,
            dtype=dtype.FLOAT32,
        )
        self.mean = norm_mean()
        self.std = norm_std()

        self.random_flip_input = fn.MediaFunc(func=RandomFlipFunction,
                                                  shape=[16], # TODO hardcoded to batch=2
                                                  dtype=dtype.UINT8,
                                                  seed=100)
        self.random_flip = fn.RandomFlip(horizontal=1,
                                        device=self.device)

        SDXLMediaPipe.instance_count += 1

    def definegraph(self):
        #print(self.input.batch_sampler, 'DEFINEGRAPH')
        #jpegs, prompt_embeds, pooled_prompt_embeds, original_sizes, crop_top_lefts, text_label = self.input() # TODO remove
        jpegs, prompt_embeds, pooled_prompt_embeds, original_sizes, crop_top_lefts = self.input()
        images = self.decode(jpegs)
        flip = self.random_flip_input()
        images = self.random_flip(images, flip)  # TODO enable flip
        images = self.cmn(images, self.mean, self.std)
        #return images, prompt_embeds, pooled_prompt_embeds, original_sizes, crop_top_lefts, text_label
        return images, prompt_embeds, pooled_prompt_embeds, original_sizes, crop_top_lefts


class MediaApiDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        sampler=None, # TODO ignored. remove
        collate_fn=None,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=None,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.fallback_activated = False

        from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUGenericPytorchIterator

        try:
            world_size = get_world_size()
        except:
            world_size = 1
        # TODO use DistributedSamplerwithLoop.. if droplast=True
        if world_size > 1:
            process_index = get_rank()
            #print(f'CREATING DISTSAMPLER world_size = {world_size}, process_index = {process_index}', flush=True)
            sampler = DistributedSampler(
                            self.dataset,
                            num_replicas=world_size,
                            rank=process_index,
                            seed=1,
                        )
        else:
            sampler = torch.utils.data.sampler.RandomSampler(self.dataset)

        self.sampler = sampler

        pipeline = SDXLMediaPipe(
            dataset=dataset,
            sampler=self.sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            queue_depth=5,
        )
        self.iterator = HPUGenericPytorchIterator(mediapipe=pipeline)
        self.epoch = 0
        

    def __len__(self):
        if self.fallback_activated:
            return super().__len__()
        else:
            return len(self.iterator)

    def __iter__(self):
        if self.fallback_activated:
            return super().__iter__()
        else:
            #try:
            #    print(f'SET EPOCH: {self.epoch} {get_rank()}', flush=True)
            #    #self.iterator.pipe.sampler.sampler.set_epoch(self.epoch) # Without this dist sampler will create same batches every epoch
            #    #self.iterator.pipe.sampler.sampler.seed += self.epoch
            #except:
            #    pass
            self.iterator.__iter__()
        self.epoch += 1
        return self

    def __next__(self):
        if self.fallback_activated:
            return super().__next__()
        #print('call next')
        #t0 = time.time()
        data = next(self.iterator)
        #print('next done', time.time()-t0)
        #import pdb; pdb.set_trace()
        #txtenc = data[5].to('cpu').numpy() # TODO remove
        return {
            "pixel_values": data[0],
            "prompt_embeds": data[1],
            "pooled_prompt_embeds": data[2],
            "original_sizes": data[3],  ## 2nd num is ZERO here
            "crop_top_lefts": data[4],
            #'text': [''.join([chr(i) for i in k]).strip('\x00') for k in txtenc] # TODO remove
        }
        #TODO sasarkar: at end of each iter, shuffle indexes



if __name__ == '__main__':
    
    #params = {'shuffle': True, 'seed': 42, 'drop_remainder': True, 'pad_remainder': True, 'dataset': 'dataset_pokemon', 'label_dtype': dtype.NDT, 'num_slices': 1, 'slice_index': 0, }
    #cparams = None
    #node_attr = None
    #rdr = read_image_text_from_dataset("RDR", None, 'cpu', [], params, cparams, node_attr)

    dataset_dir = 'dataset_pokemon'

    train_dataset = get_dataset_for_pipeline(dataset_dir)
    dataloader_params = {
                "batch_size": 16,
                #"collate_fn": data_collator,
                "num_workers": 8,
                "pin_memory": True,
                "sampler": None
            }
    def attach_metadata(batch):
        import imagesize
        return {"original_sizes" : imagesize.get(batch['image']), "crop_top_lefts" : (0,0)}
    train_dataset = train_dataset.map(attach_metadata)
    dataloader = MediaApiDataLoader(train_dataset, **dataloader_params)
    for idx, dt in enumerate(dataloader):
        import pdb; pdb.set_trace()
        print()
