'''
Stage 3 ablation study: change one setting of the CNN at a time (depth, kernel size, padding,
stride, pooling layer, hidden layer dimension, loss function) and compare the test performance.

Results are rewritten to one of 3 files after every run, so the
script can be stopped and restarted: finished runs are skipped.
Results files:
    for MNIST runs: result/stage_3_result/ablation_results_MNIST.json
    for ORL runs: result/stage_3_result/ablation_results_ORL.json
    for CIFAR runs: result/stage_3_result/ablation_results_CIFAR.json
Each setting is run with every seed in SEEDS; the table reports mean +- std.
Usage: python script_ablation.py [MNIST|ORL|CIFAR] [group names...]
'''

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Plotter import plot_ablation_groups
import numpy as np
import torch
import json
import csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_folder = os.path.join(ROOT, 'data', 'stage_3_data') + os.sep
result_folder = os.path.join(ROOT, 'result', 'stage_3_result') + os.sep
os.makedirs(result_folder, exist_ok=True)

SEEDS = (0, 1, 2)

# base settings that differ by dataset
DATASETS = {
    'MNIST': dict(base=dict(in_channels=1, input_size=(28, 28), n_classes=10, dropout=0.25, batch_size=128),
                  epochs=10, ymin=0.9),
    'ORL':   dict(base=dict(in_channels=1, input_size=(112, 92), n_classes=40, dropout=0.5, batch_size=16),
                  epochs=30, ymin=0.0),
    'CIFAR': dict(base=dict(in_channels=3, input_size=(32, 32), n_classes=10, dropout=0.25, batch_size=128),
                  epochs=10, ymin=0.3),
}

# reference configurations shared by all three datasets; every ablation run changes exactly one of these settings
SHARED = dict(conv_channels=(32, 64), kernel_size=3, padding=1, stride=1, pool='max',
              hidden_dim=128, activation='relu', optimizer='adam', learning_rate=1e-3,
              loss='cross_entropy', device='cpu')

# (group, label shown in the report, changes relative to SHARED)
RUNS = [
    ('Depth', '1 conv block', dict(conv_channels=(32,))),
    ('Depth', '2 conv blocks (baseline)', dict()),
    ('Depth', '3 conv blocks', dict(conv_channels=(32, 64, 128))),
    ('Kernel size', '3x3 (baseline)', dict()),
    ('Kernel size', '5x5', dict(kernel_size=5, padding=2)),
    ('Padding', 'padding 1 (baseline)', dict()),
    ('Padding', 'padding 0', dict(padding=0)),
    ('Stride', 'stride 1 (baseline)', dict()),
    ('Stride', 'stride 2', dict(stride=2)),
    ('Pooling', 'max (baseline)', dict()),
    ('Pooling', 'average', dict(pool='avg')),
    ('Hidden dim', '128 (baseline)', dict()),
    ('Hidden dim', '512', dict(hidden_dim=512)),
    ('Loss function', 'cross-entropy (baseline)', dict()),
    ('Loss function', 'MSE on softmax', dict(loss='mse')),
]


def save_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def load_data(name):
    splits = {}
    for split in ('train', 'test'):
        obj = Dataset_Loader('stage 3 %s %s' % (name, split), '')
        obj.dataset_source_folder_path = data_folder
        obj.dataset_source_file_name = name      # 'MNIST', 'ORL' or 'CIFAR'
        obj.split = split
        splits[split] = obj.load()
    return {'train': splits['train'], 'test': splits['test']}


def run_dataset(name, only):
    cfg = DATASETS[name]
    epochs = cfg['epochs']
    base = dict(SHARED, **cfg['base'])
    out_json = result_folder + 'ablation_results_' + name + '.json'

    def run_key(changes, seed):
        return json.dumps(dict(base, max_epoch=epochs, seed=seed, **changes), sort_keys=True, default=list)

    if os.path.exists(out_json):
        with open(out_json) as f:
            results = json.load(f)
    else:
        results = {}
    todo = [(g, l, c, s) for g, l, c in RUNS if (not only or g in only)
            for s in SEEDS if run_key(c, s) not in results]

    data = None
    if todo:
        data = load_data(name)

    for group, label, changes, seed in todo:
        key = run_key(changes, seed)
        if key in results:  # baseline appears in several groups
            continue
        print('=' * 20, name, group, '|', label, '| seed', seed, '=' * 20)
        np.random.seed(seed)
        torch.manual_seed(seed)
        m = Method_CNN('cnn', label, max_epoch=epochs, **dict(base, **changes))
        m.data = data
        out = m.run()
        ev = Evaluate_Metrics('metrics', '')
        ev.data = out
        scores = ev.evaluate()
        results[key] = {'scores': scores, 'history': m.history, 'train_time_s': m.method_training_time,
                        'n_parameters': sum(p.numel() for p in m.parameters())}
        save_json(results, out_json)
        print(label, {k: round(v, 4) for k, v in scores.items() if k in ('accuracy', 'f1_macro')})

    rows, groups = [], {}
    for group, label, changes in RUNS:
        rs = [results[run_key(changes, s)] for s in SEEDS if run_key(changes, s) in results]
        if not rs:
            continue

        def ms(k):
            v = [r['scores'][k] for r in rs]
            return '%.4f +- %.4f' % (np.mean(v), np.std(v))

        rows.append([group, label, len(rs), ms('accuracy'), ms('precision_macro'),
                     ms('recall_macro'), ms('f1_macro'), rs[0]['n_parameters'],
                     '%.0f' % np.mean([r['train_time_s'] for r in rs])])
        groups.setdefault(group, []).append((label, rs[0]['history']))  # curves from the first seed
    with open(result_folder + 'ablation_table_%s.csv' % name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'setting', 'n_seeds', 'accuracy', 'precision_macro', 'recall_macro',
                    'f1_macro', 'params', 'train_s'])
        w.writerows(rows)
    if groups:
        plot_ablation_groups(list(groups.items()), result_folder + 'ablation_curves_%s.png' % name,
                             ymin=cfg['ymin'])


if __name__ == '__main__':
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    args = sys.argv[1:]
    names = [args[0]] if args and args[0] in DATASETS else list(DATASETS)
    only = args[1:] if args and args[0] in DATASETS else args

    valid = {r[0] for r in RUNS}
    bad = [g for g in only if g not in valid]
    if bad:
        sys.exit('Unknown group(s) %s. Choose from: %s' % (bad, sorted(valid)))
    for n in names:
        run_dataset(n, only)