from habana_frameworks.mediapipe import fn
from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.backend.operator_specs import schema
from habana_frameworks.mediapipe.media_types import dtype, ftype, imgtype, randomCropType, readerOutType
from habana_frameworks.mediapipe.mediapipe import MediaPipe
from habana_frameworks.mediapipe.operators.reader_nodes.read_image_from_dir import get_max_file
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function

from torch.utils.data.sampler import BatchSampler, RandomSampler
import numpy as np


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
        print(random_flips)
        return random_flips

class ClipMediaPipe(MediaPipe):

    batch_sampler = None
    instance_count = 0

    def __init__(self, dataset=None, sampler=None, batch_size=512, drop_last=False, queue_depth=1):
        self.device = 'Gaudi2'
        self.drop_last = drop_last
        self.sampler = sampler
        ClipMediaPipe.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.image_size = 1024
        pipe_name = "{}:{}".format(self.__class__.__name__, ClipMediaPipe.instance_count)
        pipe_name = str(pipe_name)

        super(ClipMediaPipe, self).__init__(
            device=self.device, batch_size=batch_size, prefetch_depth=queue_depth, pipe_name=pipe_name
        )
        self.input = fn.ReadImageDatasetFromDir(shuffle=False, dir='MNIST-JPG/MNIST/test', format='jpg')
        def_output_image_size = [self.image_size, self.image_size]
        res_pp_filter = ftype.BICUBIC
        self.decode = fn.ImageDecoder(
            device=self.device,
            output_format=imgtype.RGB_P,
            #random_crop_type=randomCropType.CENTER_CROP,
            resize=def_output_image_size,
            resampling_mode=res_pp_filter,
        )
        self.coin_flip = fn.CoinFlip(seed=100,
                                    device=self.device)
        self.random_flip = fn.RandomFlip(horizontal=1,
                                        device=self.device)
        self.const = fn.Constant(constant=0.5,
                                dtype=dtype.FLOAT32,
                                device=self.device)


        self.random_flip_input = fn.MediaFunc(func=RandomFlipFunction,
                                                  shape=[16],
                                                  dtype=dtype.UINT8,
                                                  seed=100)


        ClipMediaPipe.instance_count += 1

    def definegraph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        if True:
            flip = self.random_flip_input()
            images = self.random_flip(images, flip)
        else:
            probability = self.const()
            predicate = self.coin_flip(probability)
            #images = self.random_flip(images, predicate)
        return images, labels 

print('here 0')
pipeline = ClipMediaPipe(
            dataset=None,
            sampler=None,
            batch_size=16,
            drop_last=True,
            queue_depth=2,
        )
pipeline.build()
pipeline.iter_init()
for i in range(3):
    print('here 4')
    images, labels = pipeline.run()
    images = images.as_cpu().as_nparray()
    labels = labels.as_cpu().as_nparray()
    import pdb; pdb.set_trace()
    print('here 5')
