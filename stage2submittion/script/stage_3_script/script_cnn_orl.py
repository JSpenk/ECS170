'''
Stage 3: train a CNN on the ORL training set, evaluate on the testing set,
save the predictions, the metrics and the learning-curve plot.

Run from this folder (script/stage_3_script).
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Plotter import plot_learning_curves
import numpy as np
import torch
import json

#---- CNN on ORL script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    data_folder = '../../data/stage_3_data/'
    result_folder = '../../result/stage_3_result/'
    os.makedirs(result_folder, exist_ok=True)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    train_obj = Dataset_Loader('stage 3 ORL train', '360 face images, 1x112x92, 1x112x92')
    train_obj.dataset_source_folder_path = data_folder
    train_obj.dataset_source_file_name = 'ORL'
    train_obj.split = 'train'

    test_obj = Dataset_Loader('stage 3 ORL test', '40 face images, 1x112x92')
    test_obj.dataset_source_folder_path = data_folder
    test_obj.dataset_source_file_name = 'ORL'
    test_obj.split = 'test'

    # 2 conv blocks (32, 64 filters, 3x3, padding 1, max-pool 2) -> FC 128 -> 40 classes
    method_obj = Method_CNN_ORL('CNN on ORL', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = result_folder + 'CNN_ORL_'
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
        print('CNN-ORL %-18s %.4f' % (k, v))
    print('training time: %.1fs' % method_obj.method_training_time)

    plot_learning_curves(method_obj.history, result_folder + 'CNN_ORL_learning_curve.png')
    with open(result_folder + 'CNN_ORL_final_metrics.json', 'w') as f:
        json.dump({'scores': scores, 'history': method_obj.history,
                   'training_time_s': method_obj.method_training_time,
                   'n_parameters': sum(p.numel() for p in method_obj.parameters())}, f, indent=2)
    print('************ Finish ************')
    # ------------------------------------------------------