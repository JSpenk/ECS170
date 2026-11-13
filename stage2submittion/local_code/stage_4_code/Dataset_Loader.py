'''
Base IO class for all datasets
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD


from local_code.base_class.dataset import dataset
import os
import pandas as pd

from pathlib import Path

class Dataset_Loader(dataset):
    dataset_name = None
    dataset_description = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    data = None
    split = 'train'

    # initialization function
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    # information print function
    def print_dataset_information(self):
        '''
        Print the basic information about the dataset class
        inclduing the dataset name, and dataset description
        '''
        print('Dataset Name: ' + self.dataset_name)
        print('Dataset Description: ' + self.dataset_description)

    def load(self):
        print('loading', self.split, 'data from',
              self.dataset_source_file_name, '...')

        path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name
        x = []
        y = []

        folder = path / self.split

        # Looping through positive and negative reviews and assigning it a label (0 or 1)
        for sentiment, label in [('neg', 0), ('pos', 1)]:
            review_folder = folder / sentiment

            for file in review_folder.glob('*.txt'):
                with open(file, 'r', encoding='utf-8') as f:
                    review = f.read()

                # Splitting up
                x.append(review)
                y.append(label)

        print('--loaded', len(x), 'reviews,', len(set(y)), 'classes')
        return {'X': x, 'y': y}

    def load_text_generation_data(self):
        print('loading text generation data...')

        file_path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name / 'data'

        df = pd.read_csv(file_path)

        jokes = df['Joke'].dropna().tolist()

        print('--loaded', len(jokes), 'jokes')

        return {'X': jokes}
