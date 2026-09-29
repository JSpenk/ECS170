'''
Stage 2 ablation study: change one setting of the MLP at a time (depth, width, activation,
optimizer, learning rate, loss function, dropout) and compare the test performance.

Results are appended to result/stage_2_result/ablation_results.json after every run, so the
script can be stopped and restarted: finished runs are skipped.
Run from this folder (script/stage_2_script).
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plotter import plot_ablation_groups
import numpy as np
import torch
import json
import csv

data_folder = '../../data/stage_2_data/'
result_folder = '../../result/stage_2_result/'
os.makedirs(result_folder, exist_ok=True)
out_json = result_folder + 'ablation_results.json'
EPOCHS = 20  # every ablation run uses the same budget so the comparison is fair

# the reference configuration; every ablation run changes exactly one of these settings
BASE = dict(hidden_dims=(256, 128), activation='relu', dropout=0.2, optimizer='adam',
            learning_rate=1e-3, loss='cross_entropy', softmax_output=False, batch_size=128)

# (group, label shown in the report, changes relative to BASE)
RUNS = [
    ('Depth', '0 hidden (784-10)', dict(hidden_dims=())),
    ('Depth', '1 hidden (256)', dict(hidden_dims=(256,))),
    ('Depth', '2 hidden (256-128)', dict()),
    ('Depth', '3 hidden (256-128-64)', dict(hidden_dims=(256, 128, 64))),
    ('Depth', '4 hidden (256-128-64-32)', dict(hidden_dims=(256, 128, 64, 32))),
    ('Width', '64-32', dict(hidden_dims=(64, 32))),
    ('Width', '128-64', dict(hidden_dims=(128, 64))),
    ('Width', '256-128', dict()),
    ('Width', '512-256', dict(hidden_dims=(512, 256))),
    ('Width', '1024-512', dict(hidden_dims=(1024, 512))),
    ('Activation', 'Sigmoid', dict(activation='sigmoid')),
    ('Activation', 'Tanh', dict(activation='tanh')),
    ('Activation', 'ReLU', dict()),
    ('Optimizer', 'SGD+momentum (lr 0.01)', dict(optimizer='sgd', learning_rate=1e-2)),
    ('Optimizer', 'Adagrad (lr 0.01)', dict(optimizer='adagrad', learning_rate=1e-2)),
    ('Optimizer', 'Adam (lr 0.001)', dict()),
    ('Learning rate (Adam)', 'lr 0.01', dict(learning_rate=1e-2)),
    ('Learning rate (Adam)', 'lr 0.001', dict()),
    ('Learning rate (Adam)', 'lr 0.0001', dict(learning_rate=1e-4)),
    ('Loss function', 'Cross-entropy on logits', dict()),
    ('Loss function', 'Softmax + cross-entropy (template)', dict(softmax_output=True)),
    ('Loss function', 'MSE on softmax', dict(loss='mse')),
    ('Dropout', 'No dropout', dict(dropout=0.0)),
    ('Dropout', 'Dropout 0.2', dict()),
    ('Dropout', 'Dropout 0.5', dict(dropout=0.5)),
]


def run_key(changes):
    cfg = dict(BASE, max_epoch=EPOCHS, **changes)
    return json.dumps(cfg, sort_keys=True, default=list)


if __name__ == '__main__':
    only = sys.argv[1:]  # optional: group names to run
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    results = json.load(open(out_json)) if os.path.exists(out_json) else {}

    todo = [r for r in RUNS if (not only or r[0] in only) and run_key(r[2]) not in results]
    if todo:
        tr = Dataset_Loader('train', ''); tr.dataset_source_folder_path = data_folder; tr.dataset_source_file_name = 'train.csv'
        te = Dataset_Loader('test', ''); te.dataset_source_folder_path = data_folder; te.dataset_source_file_name = 'test.csv'
        data = {'train': tr.load(), 'test': te.load()}

    for group, label, changes in todo:
        key = run_key(changes)
        if key in results:  # the shared reference run appears in several groups
            continue
        print('=' * 20, group, '|', label, '=' * 20)
        np.random.seed(2)
        torch.manual_seed(2)
        m = Method_MLP('mlp', label, max_epoch=EPOCHS, **dict(BASE, **changes))
        m.data = data
        out = m.run()
        ev = Evaluate_Metrics('metrics', '')
        ev.data = out
        scores = ev.evaluate()
        results[key] = {'scores': scores, 'history': m.history, 'train_time_s': m.method_training_time,
                        'n_parameters': sum(p.numel() for p in m.parameters())}
        json.dump(results, open(out_json, 'w'), indent=1)
        print(label, {k: round(v, 4) for k, v in scores.items() if k in ('accuracy', 'f1_macro')})

    # ---- summary table and plot ----
    rows, groups = [], {}
    for group, label, changes in RUNS:
        r = results.get(run_key(changes))
        if r is None:
            continue
        s = r['scores']
        rows.append([group, label, '%.4f' % s['accuracy'], '%.4f' % s['precision_weighted'],
                     '%.4f' % s['recall_weighted'], '%.4f' % s['f1_weighted'], '%.4f' % s['f1_macro'],
                     r['n_parameters'], '%.0f' % r['train_time_s']])
        groups.setdefault(group, []).append((label, r['history']))
    with open(result_folder + 'ablation_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'setting', 'accuracy', 'precision_w', 'recall_w', 'f1_w', 'f1_macro', 'params', 'train_s'])
        w.writerows(rows)
    for r in rows:
        print(' | '.join(str(x) for x in r))
    if groups:
        plot_ablation_groups(list(groups.items()), result_folder + 'ablation_curves.png', ymin=0.9)
