import matplotlib.pyplot as plt
import seaborn as sns


def plot_learning_curves_separate(df):
    fig, axes = plt.subplots(ncols=2, figsize=(15,5))
    sns.despine()

    fig.suptitle('Learning curves')

    for ax, metric, color in zip(axes, ['error', 'loss'], ['r', 'b']):
        for split, line_style in zip(['train', 'valid'], ['--', '-']):
            ax.plot(df['%s_%s' % (split,metric)], color + line_style, label=split)
        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    return fig, axes


def plot_learning_curves_together(df):
    fig, ax1 = plt.subplots(figsize=(10,5))
    sns.despine()
    ax2 = ax1.twinx()
    axes = (ax1, ax2)

    fig.suptitle('Learning curves')

    for ax, metric, color in zip(axes, ['error', 'loss'], ['r', 'b']):
        for split, line_style in zip(['train', 'valid'], ['--', '-']):
            ax.plot(df['%s_%s' % (split,metric)], color + line_style, label=split)
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric, color=color)
        ax.legend()

    return fig, axes
