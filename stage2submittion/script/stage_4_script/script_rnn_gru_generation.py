
'''
Stage 4: train an RNN on the text generation dataset
and generate text from three starting words.

Run from this folder (script/stage_4_script).
'''

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
))

from local_code.stage_4_code.Plotter import plot_generation_learning_curve
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.METHOD_RNN_Generation import METHOD_RNN_Generation

import numpy as np
import torch
import json

data_folder = '../../data/stage_4_data/'
result_folder = '../../result/stage_4_result/'

if 1:
    np.random.seed(2)
    torch.manual_seed(2)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    os.makedirs(result_folder, exist_ok=True)

train_obj = Dataset_Loader(
    'Text Generation',
    'Joke text generation dataset'
)

train_obj.dataset_source_folder_path = data_folder
train_obj.dataset_source_file_name = 'text_generation'

data = train_obj.load_text_generation_data()

method_obj = METHOD_RNN_Generation(
    'RNN Generation',
    'Text generation using RNN',
    rnn_type='gru',
    vocab_size=10000,
    embedding_dim=100,
    rnn_hidden=128,
    max_epoch=15,
    batch_size=32,
    bidirectional=False
)

method_obj.data = data

print('************ Start ************')
sequences = method_obj.run()

plot_generation_learning_curve(
    method_obj.history,
    result_folder + 'RNN__GRU_Generation_learning_curve.png'
)
print('************ Finish ************')
