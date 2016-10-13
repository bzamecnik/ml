"""
Creates and trains an ML model and evaluates it.
"""

import numpy as np
import os
import pandas as pd
import shutil

# model training
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

# evaluation
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib as mpl
# do not use Qt/X that require $DISPLAY, must be called before importing pyplot
mpl.use('Agg')
import matplotlib.pyplot as plt

from prepare_training_data import load_data, load_transformers


def create_model(input_shape, class_count):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(class_count))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics=['accuracy'])

    model.summary()

    return model


def train_model(model, x, y, ix, output_dir, batch_size=32, epoch_count=30):
    with open(output_dir + '/model_arch.yaml', 'w') as f:
        f.write(model.to_yaml())

    training_hist = model.fit(
        x[ix['train']], y[ix['train']],
        validation_data=(x[ix['valid']], y[ix['valid']]),
        batch_size=batch_size, nb_epoch=epoch_count,
        verbose=1)

    model.save_weights(output_dir + '/model_weights.h5') # HDF5

    return model, training_hist


def predict(model, x, y, ix, output_dir):
    """
    Store predictions in a CSV file and predicted probabilities in an NPZ file.
    """

    y_proba_pred = model.predict_proba(x)
    np.savez_compressed(output_dir + '/predictions_proba.npz',
        y_proba_pred=y_proba_pred)

    df = pd.DataFrame({
        'y_pred': model.predict_classes(x),
        'y_true': np_utils.categorical_probas_to_classes(y)})

    df['accurate'] = df['y_true'] == df['y_pred']

    df['split'] = ''
    for key, indexes in ix.items():
        df.ix[indexes, 'split'] = key

    df = df[['split', 'y_true', 'y_pred', 'accurate']]

    df.to_csv(output_dir + '/predictions.csv', index=None)


def evaluate_model(model, x, y, ix, instr_family_le, scaler, training_hist, output_dir):
    y_valid_pred = model.predict_classes(x[ix['valid']])
    y_valid_pred_proba = model.predict_proba(x[ix['valid']])

    def evaluation_metrics():
        splits = ['train', 'valid', 'test']
        metrics = pd.DataFrame([
                model.evaluate(x[ix[split]], y[ix[split]], verbose=0)
                for split in splits
            ],
            columns=model.metrics_names,
            index=splits)
        metrics['error'] = 1.0 - metrics['acc']
        metrics['count'] = [len(ix[split]) for split in splits]
        metrics['abs_error'] = (metrics['error'] * metrics['count']).astype(int)
        print(metrics)
        metrics.to_csv(output_dir + '/metrics.csv', float_format='%.3f')
        metrics

    def plot_learning_curves(training_hist):
        plt.figure()
        for key in training_hist.history.keys():
            plt.plot(training_hist.history[key], label=key)
        plt.axhline(1.0)
        plt.legend()
        plt.savefig(output_dir + '/learning_curves.png')
        plt.clf()

    def auc():
        auc = roc_auc_score(y[ix['valid']], y_valid_pred_proba)
        print('AUC (valid):', auc)
        return auc

    def compute_confusion_matrix():
        print('confusion matrix (valid): rows = truth, columns = predictions')
        y_valid_true = np_utils.categorical_probas_to_classes(y[ix['valid']])
        cm = confusion_matrix(y_valid_true, y_valid_pred)
        cm_df = pd.DataFrame(cm,
            columns=instr_family_le.classes_,
            index=instr_family_le.classes_)
        cm_df.to_csv(output_dir + '/confusion_matrix_valid.csv', float_format='%.3f')
        print(cm_df)
        return cm

    def plot_confusion_matrix(cm, labels, title='Confusion matrix'):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(output_dir + '/confusion_matrix_valid.png')
        plt.clf()

    def per_class_metrics(cm):
        per_class_metrics = pd.DataFrame(np.diag(cm) / cm.sum(axis=1),
            columns=['accuracy'], index=instr_family_le.classes_)
        per_class_metrics['error'] = 1.0 - per_class_metrics['accuracy']
        print(per_class_metrics)
        per_class_metrics.to_csv(output_dir + '/per_class_metrics_valid.csv',
            float_format='%.3f')

    evaluation_metrics()
    plot_learning_curves(training_hist)
    auc()
    cm = compute_confusion_matrix()
    plot_confusion_matrix(cm, instr_family_le.classes_)
    per_class_metrics(cm)


def prepare_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    base_dir = 'data/working/single-notes-2000'
    data_dir = base_dir + '/ml-inputs'
    model_dir = base_dir + '/model'
    evaluation_dir = base_dir + '/evaluation'

    prepare_dirs([data_dir, model_dir, evaluation_dir])

    x, y, ix = load_data(data_dir)
    instr_family_le, scaler, _ = load_transformers(data_dir)

    model = create_model(input_shape=x.shape[1:], class_count=y.shape[1])
    model, training_hist = train_model(model,
        x, y, ix,
        model_dir,
        epoch_count=20)

    shutil.copy(
        data_dir + '/preproc_transformers.json',
        model_dir + '/preproc_transformers.json')

    predict(model, x, y, ix, evaluation_dir)

    evaluate_model(model, x, y, ix, instr_family_le, scaler, training_hist,
        evaluation_dir)
