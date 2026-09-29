'''
Stage 2: train the final MLP on the training set, evaluate on the testing set,
save the predictions, the metrics and the learning-curve plot.

Run from this folder (script/stage_2_script), as with the stage 1 scripts.
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plotter import plot_learning_curves
import numpy as np
import torch
import json

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    data_folder = '../../data/stage_2_data/'
    result_folder = '../../result/stage_2_result/'
    os.makedirs(result_folder, exist_ok=True)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    train_obj = Dataset_Loader('stage 2 train', '60,000 hand-written digit images, 784 pixel features')
    train_obj.dataset_source_folder_path = data_folder
    train_obj.dataset_source_file_name = 'train.csv'

    test_obj = Dataset_Loader('stage 2 test', '10,000 hand-written digit images, 784 pixel features')
    test_obj.dataset_source_folder_path = data_folder
    test_obj.dataset_source_file_name = 'test.csv'

    # final model (best setting in the ablation study, script_ablation.py):
    # 784 -> 256 -> 128 -> 10, ReLU, dropout 0.2, Adam (lr 1e-3), cross-entropy, batch 128
    method_obj = Method_MLP('multi-layer perceptron', '', input_dim=784, hidden_dims=(256, 128),
                            n_classes=10, activation='relu', dropout=0.2, optimizer='adam',
                            learning_rate=1e-3, loss='cross_entropy', max_epoch=30, batch_size=128)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = result_folder + 'MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('pre-partitioned train/test', '')

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj, test_obj)
    setting_obj.print_setup_summary()
    print(method_obj)
    scores, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    for k, v in scores.items():
        print('MLP %-18s %.4f' % (k, v))
    print('training time: %.1fs' % method_obj.method_training_time)

    plot_learning_curves(method_obj.history, result_folder + 'MLP_learning_curve.png')
    with open(result_folder + 'MLP_final_metrics.json', 'w') as f:
        json.dump({'scores': scores, 'history': method_obj.history,
                   'training_time_s': method_obj.method_training_time,
                   'n_parameters': sum(p.numel() for p in method_obj.parameters())}, f, indent=2)
    print('************ Finish ************')
    # ------------------------------------------------------
