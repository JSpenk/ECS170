'''
Concrete IO class for the stage 3 image datasets (MNIST, ORL, CIFAR pickle files).

Every file is a pickled dict {'train': [...], 'test': [...]}, each item being
{'image': pixel matrix, 'label': int}. Set `split` to 'train' or 'test' to choose
which part this loader object returns.

Output: X with shape (n, channels, height, width), scaled to [0, 1], and y with
labels starting at 0.
'''

# Based on the ECS 170 code template (Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>)
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import pickle


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None   # 'MNIST', 'ORL' or 'CIFAR'
    split = 'train'                   # 'train' or 'test'
    normalize = True                  # scale pixels from [0, 255] into [0, 1]

    # the pickle files are big, so the train and test loader objects share one read
    _cache = {}

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading', self.split, 'data from', self.dataset_source_file_name, '...')
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        if path not in Dataset_Loader._cache:
            with open(path, 'rb') as f:
                Dataset_Loader._cache[path] = pickle.load(f)
        instances = Dataset_Loader._cache[path][self.split]
        name = self.dataset_source_file_name.upper()

        images = np.stack([np.asarray(inst['image']) for inst in instances]).astype(np.float32)
        y = np.array([inst['label'] for inst in instances], dtype=np.int64)

        if images.ndim == 3:
            # MNIST: (n, 28, 28) -> (n, 1, 28, 28)
            X = images[:, None, :, :]
        else:
            if name == 'ORL':
                # the three channels hold identical values, so keep just one
                images = images[:, :, :, :1]
            # (n, H, W, C) -> (n, C, H, W), the layout PyTorch convolutions expect
            X = images.transpose(0, 3, 1, 2)

        if name == 'ORL':
            # ORL labels run 1..40, the loss function needs 0..39
            y = y - 1

        if self.normalize:
            X = X / 255.0
        X = np.ascontiguousarray(X, dtype=np.float32)

        print('--loaded', X.shape[0], 'instances, image shape', X.shape[1:], ',',
              len(np.unique(y)), 'classes (labels', y.min(), 'to', str(y.max()) + ')')
        return {'X': X, 'y': y}