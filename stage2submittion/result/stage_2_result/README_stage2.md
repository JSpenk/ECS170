# Stage 2 - MLP classification (results summary)

## How to run (PyCharm: working directory = script/stage_2_script)
- `script_mlp.py`      final model -> MLP_prediction_result_None, MLP_final_metrics.json, MLP_learning_curve.png
- `script_ablation.py` 24 ablation runs -> ablation_results.json, ablation_table.csv, ablation_curves.png
  (resumable: finished runs are skipped; pass group names, e.g. `python script_ablation.py Depth`, to run a subset)

## Data
data/stage_2_data/train.csv (60,000 rows) and test.csv (10,000 rows), pre-partitioned.
Each row: label (0-9), then 784 integer pixel features (28x28 digit image). Features scaled to [0,1] (divide by 255).
Classes are roughly balanced (5,421-6,742 training instances per digit).

## Final model (script_mlp.py)
784 -> Linear 256 -> ReLU -> Dropout 0.2 -> Linear 128 -> ReLU -> Dropout 0.2 -> Linear 10 (logits)
235,146 parameters. Cross-entropy loss on logits, Adam (lr 0.001), mini-batch 128, 30 epochs,
PyTorch default init, seed 2. Training time ~35 s on CPU.

| Metric    | Macro  | Weighted | Micro  |
|-----------|--------|----------|--------|
| Precision | 0.9831 | 0.9830   | 0.9830 |
| Recall    | 0.9828 | 0.9830   | 0.9830 |
| F1        | 0.9829 | 0.9830   | 0.9830 |
Accuracy: 0.9830. Training loss 0.386 (epoch 1) -> 0.014 (epoch 30).

## Ablation (20 epochs each, change one setting from the 256-128 / ReLU / dropout 0.2 / Adam 1e-3 / CE reference)
| Group | Setting | Test acc | Macro F1 |
|---|---|---|---|
| Depth | 0 hidden (784-10) | 0.9284 | 0.9273 |
| Depth | 1 hidden (256) | 0.9827 | 0.9826 |
| Depth | 2 hidden (256-128) *reference* | 0.9837 | 0.9837 |
| Depth | 3 hidden (256-128-64) | 0.9835 | 0.9834 |
| Depth | 4 hidden (256-128-64-32) | 0.9796 | 0.9795 |
| Width | 64-32 | 0.9748 | 0.9746 |
| Width | 128-64 | 0.9803 | 0.9802 |
| Width | 512-256 | 0.9825 | 0.9823 |
| Width | 1024-512 | 0.9826 | 0.9825 |
| Activation | Sigmoid | 0.9802 | 0.9801 |
| Activation | Tanh | 0.9800 | 0.9799 |
| Optimizer | SGD + momentum 0.9 (lr 0.01) | 0.9814 | 0.9813 |
| Optimizer | Adagrad (lr 0.01) | 0.9814 | 0.9813 |
| Learning rate | Adam lr 0.01 | 0.9746 | 0.9744 |
| Learning rate | Adam lr 0.0001 | 0.9759 | 0.9757 |
| Loss | Softmax layer + cross-entropy (stage 1 template) | 0.9789 | 0.9788 |
| Loss | MSE on softmax vs one-hot | 0.9829 | 0.9828 |
| Dropout | none | 0.9784 | 0.9782 |
| Dropout | 0.5 | 0.9817 | 0.9816 |

Takeaways: a hidden layer matters most (+5.5 pts over a linear softmax model); beyond 2 hidden layers / 256-128
width there is no gain. ReLU > sigmoid/tanh; Adam at 1e-3 beats SGD/Adagrad and other Adam learning rates.
The stage 1 template applies Softmax before CrossEntropyLoss (which already applies log-softmax), squashing the
gradients; feeding raw logits fixes it (+0.5 pts). Dropout 0.2 reduces overfitting (train 99.5% vs test 98.3%).
