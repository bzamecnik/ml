"""
Evaluates the model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
# do not use Qt/X that require $DISPLAY, must be called before importing pyplot
mpl.use('Agg')
import matplotlib.pyplot as plt

from prepare_training_data import load_indexes, load_transformers
from plots import plot_learning_curves_separate


def evaluate_model(data_dir, model_dir, evaluation_dir):
    ix = load_indexes(data_dir)

    predictions = pd.read_csv(evaluation_dir +  '/predictions.csv')
    y_proba_pred = np.load(evaluation_dir +  '/predictions_proba.npz')['y_proba_pred']

    instr_family_le, scaler, _ = load_transformers(model_dir)

    training_history = pd.read_csv(evaluation_dir + '/learning_curves.csv')

    final_metrics = pd.read_csv(evaluation_dir + '/final_metrics.csv', index_col=0)

    splits = list(final_metrics.index)

    def plot_learning_curves(training_history):
        fig, axes = plot_learning_curves_separate(training_history)
        fig.savefig(evaluation_dir + '/learning_curves.png')

    def compute_confusion_matrix(split):
        print('confusion matrix (%s): rows = truth, columns = predictions' % split)
        y_true = predictions['y_true'][ix[split]]
        y_pred = predictions['y_pred'][ix[split]]
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm,
            columns=instr_family_le.classes_,
            index=instr_family_le.classes_)
        cm_df.to_csv(evaluation_dir + '/confusion_matrix_%s.csv' % split, float_format='%.3f')
        print(cm_df)
        return cm

    def plot_confusion_matrix(cm, labels, split):
        plt.figure()
        plt.grid(False)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix (%s)' % split)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(evaluation_dir + '/confusion_matrix_%s.png' % split)
        plt.clf()

    def per_class_metrics(cm, split):
        per_class_metrics = pd.DataFrame(np.diag(cm) / cm.sum(axis=1),
            columns=['accuracy'], index=instr_family_le.classes_)
        per_class_metrics['error'] = 1.0 - per_class_metrics['accuracy']
        print(per_class_metrics)
        csv_file = evaluation_dir + '/per_class_metrics_%s.csv' % split
        per_class_metrics.to_csv(csv_file, float_format='%.3f')

    def analyze_error_by_pitch():
        parameters_with_targets = pd.read_csv(data_dir + '/parameters_with_targets.csv', index_col=0)

        data_points = parameters_with_targets.join(predictions)

        def compute_error_by_midi_bin(split='valid'):
            correct_by_midi_valid = data_points[data_points['split'] == split][['midi_number', 'accurate']]

            df = pd.DataFrame(index=np.arange(128))
            hist = correct_by_midi_valid.groupby(['midi_number', 'accurate']).size().reset_index().set_index('midi_number')
            df['correct'] = hist[hist['accurate']][0]
            df['incorrect'] = hist[~hist['accurate']][0]
            df.fillna(0, inplace=True)

            df['bin'] = df.index.map(lambda i: (i // 8) * 8)
            df['total'] = df['correct'] + df['incorrect']

            df_bins = df.groupby('bin').sum()

            df_bins['correct_perc'] = df_bins['correct'] / df_bins['total']
            df_bins['incorrect_perc'] = df_bins['incorrect'] / df_bins['total']

            csv_file = evaluation_dir + '/error_by_midi_bins_%s.csv' % split
            df_bins.to_csv(csv_file)

            return df_bins

        def plot_error_by_midi_bin(df, split):
            fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
            df[['correct', 'incorrect']].plot(
                kind='bar', stacked=True, color=['green', 'red'], ax=axes[0])
            df[['correct_perc', 'incorrect_perc']].plot(
                kind='bar', stacked=True, color=['green', 'red'], ax=axes[1])

            plt.suptitle('Error (absolute, relative) by pitch bins')

            plt.savefig(evaluation_dir + '/error_by_midi_bins_%s.pdf' % split)

        for split in splits:
            plot_error_by_midi_bin(compute_error_by_midi_bin(split), split)

    plot_learning_curves(training_history)

    for split in splits:
        print('split: ', split)
        cm = compute_confusion_matrix(split)
        plot_confusion_matrix(cm, instr_family_le.classes_, split)
        per_class_metrics(cm, split)

    analyze_error_by_pitch()

if __name__ == '__main__':
    base_dir = 'data/working/single-notes-2000'
    data_dir = base_dir + '/ml-inputs'
    model_dir = base_dir + '/model'
    evaluation_dir = base_dir + '/evaluation'

    evaluate_model(data_dir, model_dir, evaluation_dir)
