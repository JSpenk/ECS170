from local_code.stage_4_code.Method_RNN import METHOD_RNN
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_4_code.Result_Saver import Result_Saver
import os

result_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'result')
)
os.makedirs(result_path, exist_ok=True)

train_loader = Dataset_Loader()
train_loader.dataset_source_folder_path = r"C:\Users\theXr\PycharmProjects\ECS170\stage2submittion\data\stage_4_data"
train_loader.dataset_source_file_name = "text_classification"
train_loader.split = "train"

test_loader = Dataset_Loader()
test_loader.dataset_source_folder_path = r"C:\Users\theXr\PycharmProjects\ECS170\stage2submittion\data\stage_4_data"
test_loader.dataset_source_file_name = "text_classification"
test_loader.split = "test"

method = METHOD_RNN(
    'RNN',
    'IMDb sentiment classification',
    rnn_type='rnn',
    vocab_size=10000,
    embedding_dim=128,
    rnn_hidden=128,
    max_length=300,
    max_epoch=1,
    batch_size=128
)

result = Result_Saver()
result.result_destination_folder_path = result_path + os.sep
result.result_destination_file_name = 'stage_4_result'
result.fold_count = 1

evaluate = Evaluate_Metrics()

setting = Setting_Train_Test()

setting.prepare(
    train_loader,
    method,
    result,
    evaluate,
    test_loader
)

scores, _ = setting.load_run_save_evaluate()

print("RNN Evaluation Results:")
print(scores)