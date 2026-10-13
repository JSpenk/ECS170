'''
Plotting helpers for the stage 2 learning curves
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc='left')
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(labelsize=8, colors=INK2, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)


def plot_learning_curves(history, path, title='MLP'):
    '''Two panels: training loss per epoch, and train / test accuracy per epoch.'''
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ep = history['epoch']
    axes[0].plot(ep, history['train_loss'], color=SERIES[0], linewidth=2)
    _style(axes[0], 'Training loss', 'Epoch', 'Cross-entropy loss')
    axes[1].plot(ep, history['train_acc'], color=SERIES[0], linewidth=2, label='Train')
    if history.get('test_acc'):
        axes[1].plot(ep, history['test_acc'], color=SERIES[1], linewidth=2, label='Test')
    _style(axes[1], 'Accuracy', 'Epoch', 'Accuracy')
    axes[1].legend(fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ablation_groups(groups, path, ncols=3, metric='test_acc', ylabel='Test accuracy', ymin=None):
    '''groups: list of (group_title, [(label, history), ...]); one panel per group.
    Test accuracy is used by default because the losses of different loss functions
    (cross-entropy vs MSE) are not on a comparable scale.'''
    n = len(groups)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.9 * nrows), squeeze=False)
    for i, (gtitle, runs) in enumerate(groups):
        ax = axes[i // ncols][i % ncols]
        for j, (label, hist) in enumerate(runs):
            ax.plot(hist['epoch'], hist[metric], color=SERIES[j % len(SERIES)], linewidth=1.6, label=label)
        _style(ax, gtitle, 'Epoch', ylabel)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        # legend under the panel so it never covers the curves
        ax.legend(fontsize=6.5, frameon=False, loc='upper left', bbox_to_anchor=(0.0, -0.3),
                  ncol=1, handlelength=1.5, borderaxespad=0)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
