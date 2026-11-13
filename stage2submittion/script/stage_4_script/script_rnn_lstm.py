'''
Stage 4: train an RNN on the IMDb training set, evaluate on the testing set,
save the predictions, the metrics and the learning-curve plot.

Run from this folder (script/stage_4_script).
'''

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
))

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import METHOD_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Plotter import plot_learning_curves
import numpy as np
import torch
import json

data_folder = '../../data/stage_4_data/'
result_folder = '../../result/stage_4_result/'

if 1:
    np.random.seed(2)
    torch.manual_seed(2)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    data_folder = '../../data/stage_4_data/'
    result_folder = '../../result/stage_4_result/'

    os.makedirs(result_folder, exist_ok=True)

# ---- objection initialization section ---------------
train_obj = Dataset_Loader('IMDb train', 'IMDb training reviews')
train_obj.dataset_source_folder_path = data_folder
train_obj.dataset_source_file_name = 'text_classification'
train_obj.split = 'train'

test_obj = Dataset_Loader('IMDb test', 'IMDb testing reviews')
test_obj.dataset_source_folder_path = data_folder
test_obj.dataset_source_file_name = 'text_classification'
test_obj.split = 'test'

method_obj = METHOD_RNN(
    'RNN',
    'IMDb sentiment classification',
    rnn_type='lstm',
    vocab_size=10000,
    embedding_dim=100,
    rnn_hidden=128,
    max_length=300,
    max_epoch=15,
    batch_size=128,
)

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = result_folder
result_obj.result_destination_file_name = 'RNN_LSTM_prediction_result'
result_obj.fold_count = 1

setting_obj = Setting_Train_Test('pre-partitioned train/test', '')
evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')

print('************ Start ************')
setting_obj.prepare(
    train_obj,
    method_obj,
    result_obj,
    evaluate_obj,
    test_obj
)

setting_obj.print_setup_summary()
print(method_obj)

scores, _ = setting_obj.load_run_save_evaluate()

print('************ Overall Performance ************')
for k, v in scores.items():
    print('RNN %-18s %.4f' % (k, v))

print('training time: %.1fs' % method_obj.method_training_time)
print('testing time: %.1fs' % method_obj.method_testing_time)

plot_learning_curves(
        method_obj.history,
        result_folder + 'RNN_LSTM_learning_curve.png'
    )

with open(result_folder + 'RNN_LSTM_final_metrics.json', 'w') as f:
    json.dump({
        'scores': scores,
        'history': method_obj.history,
        'training_time_s': method_obj.method_training_time,
        'testing_time_s': method_obj.method_testing_time,
        'n_parameters': sum(
            p.numel() for p in method_obj.parameters()
        )
    }, f, indent=2)

print('************ Finish ************')