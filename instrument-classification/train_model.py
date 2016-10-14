"""
Creates and trains an ML model and makes predictions on the data.
"""

import numpy as np
import os
import pandas as pd
import shutil
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

from prepare_training_data import load_data, load_transformers


def create_model(input_shape, class_count):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(class_count))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics=['accuracy'])

    model.summary()

    return model


def train_model(model, x, y, ix, model_dir, evaluation_dir,
    batch_size=32, epoch_count=30):
    with open(model_dir + '/model_arch.yaml', 'w') as f:
        f.write(model.to_yaml())

    training_hist = model.fit(
        x[ix['train']], y[ix['train']],
        validation_data=(x[ix['valid']], y[ix['valid']]),
        batch_size=batch_size, nb_epoch=epoch_count,
        verbose=1)

    model.save_weights(model_dir + '/model_weights.h5') # HDF5

    store_learning_curves(training_hist, evaluation_dir)

    return model


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

    return y_proba_pred


def compute_final_metrics(model, x, y, ix, y_proba_pred, evaluation_dir):
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

    metrics['auc'] = [roc_auc_score(y[ix[split]], y_proba_pred[ix[split]])
        for split in splits]

    print(metrics)
    metrics.to_csv(evaluation_dir + '/final_metrics.csv', float_format='%.5f')


def store_learning_curves(training_hist,  evaluation_dir):
    df = pd.DataFrame(training_hist.history)
    df.rename(columns={
        'acc': 'train_acc', 'loss': 'train_loss',
        'val_acc': 'valid_acc', 'val_loss': 'valid_loss'
    }, inplace=True)
    df['train_error'] = 1.0 - df['train_acc']
    df['valid_error'] = 1.0 - df['valid_acc']
    df.to_csv(evaluation_dir + '/learning_curves.csv', index=None)


def prepare_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    base_dir = 'data/working/single-notes-2000'
    data_dir = base_dir + '/ml-inputs'
    model_dir = base_dir + '/model'
    evaluation_dir = base_dir + '/evaluation'

    prepare_dirs([data_dir, model_dir, evaluation_dir])

    shutil.copy(
        data_dir + '/preproc_transformers.json',
        model_dir + '/preproc_transformers.json')

    x, y, ix = load_data(data_dir)
    instr_family_le, scaler, _ = load_transformers(data_dir)

    model = create_model(input_shape=x.shape[1:], class_count=y.shape[1])
    model = train_model(model,
        x, y, ix,
        model_dir, evaluation_dir,
        epoch_count=50)

    y_proba_pred = predict(model, x, y, ix, evaluation_dir)

    compute_final_metrics(model, x, y, ix, y_proba_pred, evaluation_dir)
