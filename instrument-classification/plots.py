import matplotlib as mpl
# do not use Qt/X that require $DISPLAY, must be called before importing pyplot
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_learning_curves_separate(df):
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    sns.despine()

    fig.suptitle('Learning curves')

    for ax, metric, color in zip(axes, ['error', 'loss'], ['r', 'b']):
        for split, lw in zip(['train', 'valid'], [1, 2]):
            ax.plot(df['%s_%s' % (split,metric)], color=color, lw=lw, label=split)
        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    fig.tight_layout()

    return fig, axes


def plot_learning_curves_together(df):
    fig, ax1 = plt.subplots(figsize=(10, 5))
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


def plot_confusion_matrix(cm, labels, split):
    fig, ax = plt.subplots()
    fig.suptitle('Confusion matrix (%s)' % split)
    ax.grid(False)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im)
    tick_marks = list(range(len(labels)))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()

    return fig, ax


def plot_error_by_midi_bin(df, split):
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    fig.suptitle('Error (absolute, relative) by pitch bins')

    df[['correct', 'incorrect']].plot(
        kind='bar', stacked=True, color=['green', 'red'], ax=axes[0])
    df[['correct_perc', 'incorrect_perc']].plot(
        kind='bar', stacked=True, color=['green', 'red'], ax=axes[1])

    fig.tight_layout()

    return fig, axes
