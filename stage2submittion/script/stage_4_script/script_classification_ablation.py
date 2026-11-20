
import os
import sys
import random
import numpy as np
import pandas as pd
import torch

# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

sys.path.insert(0, PROJECT_ROOT)

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN

SEED = 2

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

# Load dataset
data_folder = os.path.join(
    PROJECT_ROOT, 'data', 'stage_4_data'
)

result_folder = os.path.join(
    PROJECT_ROOT, 'result', 'stage_4_result'
)

os.makedirs(result_folder, exist_ok=True)

def load_classification_data(split):

    loader = Dataset_Loader(
        'Text Classification',
        'IMDb movie review sentiment classification'
    )

    loader.dataset_source_folder_path = data_folder
    loader.dataset_source_file_name = 'text_classification'
    loader.split = split

    return loader.load()

dataset_path = os.path.join(
    data_folder,
    'text_classification'
)

print("\n========== DATASET PATH CHECK ==========")
print("Dataset path:", dataset_path)
print("Dataset exists:", os.path.exists(dataset_path))

for split in ['train', 'test']:
    for sentiment in ['neg', 'pos']:

        folder = os.path.join(
            dataset_path,
            split,
            sentiment
        )

        print("\nFolder:", folder)
        print("Exists:", os.path.exists(folder))

        if os.path.exists(folder):
            print("Number of files:", len(os.listdir(folder)))


train_data = load_classification_data('train')
test_data = load_classification_data('test')

train_reviews = train_data['X']
train_labels = train_data['y']

test_reviews = test_data['X']
test_labels = test_data['y']

print("\nTraining reviews:", len(train_reviews))
print("Testing reviews:", len(test_reviews))

# Baseline
baseline = {
    'vocab_size': 10000,
    'embedding_dim': 100,
    'max_length': 300,
    'n_classes': 2,

    'rnn_hidden': 128,
    'num_layers': 1,
    'rnn_type': 'rnn',
    'bidirectional': False,
    'rnn_dropout': 0.4,

    'hidden_dim': 64,
    'activation': 'relu',
    'dropout': 0.5,

    'optimizer': 'adam',
    'learning_rate': 0.001,
    'weight_decay': 0.0001,

    'max_epoch': 15,
    'batch_size': 128
}

# Ablation

experiments = [
    ('Baseline', {}),

    ('Embedding', {'embedding_dim': 64}),
    ('Embedding', {'embedding_dim': 128}),

    ('Hidden Dimension', {'rnn_hidden': 64}),
    ('Hidden Dimension', {'rnn_hidden': 256}),

    ('Depth', {'num_layers': 2}),

    ('Learning Rate', {'learning_rate': 0.01}),
    ('Learning Rate', {'learning_rate': 0.0001})
]
architectures = ['rnn', 'lstm', 'gru']

def run_experiment(architecture, factor, changes):

    set_seed()

    config = baseline.copy()
    config.update(changes)
    config['rnn_type'] = architecture

    model = Method_RNN(
        architecture.upper() + ' Classification',
        'Sentiment classification ablation experiment',
        **config
    )

    model._build_vocabulary(train_reviews)

    X_train, train_lengths = model._encode_reviews(
        train_reviews
    )

    X_test, test_lengths = model._encode_reviews(
        test_reviews
    )

    print(
        f"\n{architecture.upper()} | {factor} | "
        f"Setting: {changes if changes else 'Baseline'}"
    )

    model.train(
        X_train,
        train_labels,
        train_lengths
    )

    predictions = model.test(
        X_test,
        test_lengths
    )

    accuracy = (
        np.asarray(predictions) == np.asarray(test_labels)
    ).mean() * 100

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Test Accuracy: {accuracy:.2f}%")

    return {
        'Architecture': architecture.upper(),
        'Factor': factor,
        'Setting': str(changes) if changes else 'Baseline',
        'Accuracy': accuracy,
        'Parameters': parameter_count
    }

results = []

for factor, changes in experiments:

    for architecture in architectures:

        result = run_experiment(
            architecture,
            factor,
            changes
        )

        results.append(result)


# Save
results_df = pd.DataFrame(results)

comparison = results_df.pivot(
    index=['Factor', 'Setting'],
    columns='Architecture',
    values='Accuracy'
).reset_index()

comparison.to_csv(
    os.path.join(
        result_folder,
        'RNN_classification_ablation_comparison.csv'
    ),
    index=False
)

print("\nClassification ablation completed!")
