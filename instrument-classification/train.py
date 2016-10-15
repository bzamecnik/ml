"""
Trains an ML model, makes predictions on the data and evaluates it.
"""

import arrow
from keras.utils import np_utils
import numpy as np
import os
import pandas as pd
import random
import shutil
from sklearn.metrics import roc_auc_score

from capture import CaptureStdout
from evaluate import evaluate_model
from model_arch import create_model
from prepare_training_data import load_data, load_transformers

def train_model(model, x, y, ix, model_dir, evaluation_dir,
    batch_size=32, epoch_count=30):
    with open(model_dir + '/model_arch.yaml', 'w') as f:
        f.write(model.to_yaml())

    with open(model_dir + '/model_summary.txt', 'w') as f:
        with CaptureStdout() as output:
            model.summary()
        f.write(str(output))

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

    y_proba_pred = model.predict(x)
    np.savez_compressed(output_dir + '/predictions_proba.npz',
        y_proba_pred=y_proba_pred)

    df = pd.DataFrame({
        'y_pred': np_utils.probas_to_classes(y_proba_pred),
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
    metrics.index.name = 'split'
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


def store_model_files(input_dir, model_dir):
    shutil.copy(
        input_dir + '/preproc_transformers.json',
        model_dir + '/preproc_transformers.json')
    shutil.copy('model_arch.py', model_dir + '/model_arch.py')

def generate_model_id():
    """
    Returns a model id based on timestamp with some random part to prevent potential collisions.
    """
    date_part = arrow.utcnow().format('YYYY-MM-DD_HH-mm-ss')
    random_part = random.randint(0, 2<<31)
    return '%s_%x' % (date_part, random_part)

if __name__ == '__main__':
    model_id = generate_model_id()
    print('model id:', model_id)

    base_dir = 'data/working/single-notes-2000'
    input_dir = base_dir + '/training-data'
    model_dir = base_dir + '/models/' + model_id
    output_dir = model_dir + '/output-data'
    evaluation_dir = model_dir + '/evaluation'

    prepare_dirs([input_dir, model_dir, output_dir, evaluation_dir])

    store_model_files(input_dir, model_dir)

    x, y, ix = load_data(input_dir)
    instr_family_le, scaler, _ = load_transformers(input_dir)

    model = create_model(input_shape=x.shape[1:], class_count=y.shape[1])
    model.summary()
    model = train_model(model,
        x, y, ix,
        model_dir, evaluation_dir,
        epoch_count=10)

    y_proba_pred = predict(model, x, y, ix, output_dir)

    compute_final_metrics(model, x, y, ix, y_proba_pred, evaluation_dir)

    evaluate_model(input_dir, model_dir)
