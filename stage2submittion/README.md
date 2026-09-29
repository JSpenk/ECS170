# ECS 170 Fall 2026 - Project Stage 2: MLP classification with PyTorch

Multi-layer perceptron that classifies 28x28 hand-written digit images (784 pixel features, 10 classes).
Final model: 784 -> 256 -> 128 -> 10, ReLU, dropout 0.2, Adam (lr 0.001), cross-entropy, batch 128, 30 epochs.
**Test accuracy 98.30%, macro F1 0.9829** (10,000 test instances).

## Folder layout (same structure as the course code template)
```
stage2submittion/
├── Stage2_Report.docx / .pdf        report (team information still to be filled in)
├── local_code/
│   ├── base_class/                  template base classes (dataset, method, setting, result, evaluate)
│   └── stage_2_code/
│       ├── Dataset_Loader.py        reads train.csv / test.csv, scales pixels to [0,1]
│       ├── Method_MLP.py            configurable PyTorch MLP + mini-batch training loop
│       ├── Evaluate_Metrics.py      accuracy + macro/weighted/micro precision, recall, F1
│       ├── Setting_Train_Test.py    pre-partitioned train/test experiment setting
│       ├── Result_Saver.py / Result_Loader.py
│       └── Plotter.py               learning-curve and ablation plots
├── script/stage_2_script/
│   ├── script_mlp.py                trains + evaluates the final model
│   └── script_ablation.py           24 ablation runs (depth, width, activation, optimizer, lr, loss, dropout)
├── data/stage_2_data/               put train.csv and test.csv here (not included, 128 MB)
└── result/stage_2_result/           outputs of the two scripts (already generated)
```

## How to run
1. `pip install -r requirements.txt`
2. Copy `train.csv` and `test.csv` from the course's stage_2_data.zip into `data/stage_2_data/`.
3. Run from inside `script/stage_2_script/` (in PyCharm: set that folder as the working directory):
   - `python script_mlp.py` - about 35 s on a CPU
   - `python script_ablation.py` - about 10 min on a CPU (finished runs are cached in ablation_results.json and skipped;
     delete that file to re-run everything, or pass group names, e.g. `python script_ablation.py Depth Width`)

## Results (result/stage_2_result/)
| File | Content |
|---|---|
| MLP_final_metrics.json | final scores, per-epoch loss/accuracy history, training time |
| MLP_prediction_result_None | pickled test predictions and true labels |
| MLP_learning_curve.png | training loss and train/test accuracy per epoch |
| ablation_results.json / ablation_table.csv | scores and learning curves of all ablation runs |
| ablation_curves.png | test accuracy per epoch, all 7 ablation groups |
| ablation_selected.png / model_architecture.png | figures used in the report |

| Metric | Macro | Weighted | Micro |
|---|---|---|---|
| Precision | 0.9831 | 0.9830 | 0.9830 |
| Recall | 0.9828 | 0.9830 | 0.9830 |
| F1 | 0.9829 | 0.9830 | 0.9830 |
| Accuracy | 0.9830 | | |
