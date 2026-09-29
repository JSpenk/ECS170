'''
Concrete IO class for the stage 2 dataset (pre-partitioned train.csv / test.csv)
'''

# Based on the ECS 170 code template (Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>)
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    # scale the raw pixel features from [0, 255] into [0, 1]
    normalize = True

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        '''
        Each line of the file is one instance: the first number is the class label,
        the remaining numbers are the features of the instance (comma separated).
        '''
        print('loading data from', self.dataset_source_file_name, '...')
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        # np.loadtxt is much faster than a python loop for 60k x 785 values
        raw = np.loadtxt(path, delimiter=',', dtype=np.float32)
        y = raw[:, 0].astype(np.int64)
        X = raw[:, 1:]
        if self.normalize:
            X = X / 255.0
        print('--loaded', X.shape[0], 'instances with', X.shape[1], 'features,',
              len(np.unique(y)), 'classes')
        return {'X': X, 'y': y}
