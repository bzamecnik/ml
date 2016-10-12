"""
Prepares features and target inputs, creates and trains an ML model and
evaluates it.
"""

import os

# preprocessing
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

# model training
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

# evaluation
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib as mpl
# do not use Qt/X that require $DISPLAY, must be called before importing pyplot
mpl.use('Agg')
import matplotlib.pyplot as plt

from instruments import midi_instruments
from preprocessing import ChromagramTransformer

jsonpickle_numpy.register_handlers()

def prepare_inputs(input_dir, output_dir, model_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    ## Load features

    chromagrams = np.load(input_dir + '/chromagrams.npz')['arr_0']
    # axes: data point, block, chroma vector
    print('chromagrams.shape:', chromagrams.shape)
    print('chromagrams.size:', chromagrams.size)

    # Reshape for the convolution filtering - add one dimension of size 1
    # that will be further used for multiple convolution filters.
    # TODO: maybe this should done in the model
    x = chromagrams.reshape(chromagrams.shape + (1,))
    print('x.shape:', x.shape)

    input_shape = x.shape[1:]
    print('input shape (rows, cols, filters):', input_shape)

    ## Load targets

    parameters = pd.read_csv(input_dir + '/parameters.csv', index_col=0)
    print('parameters.shape:', parameters.shape)

    instruments = midi_instruments()

    # Add the instrument family which will be our target.
    parameters = parameters.join(instruments[['family_name']], on='midi_instrument')

    ### Encode the targets

    instr_family_le = LabelEncoder()
    parameters['family_id'] = instr_family_le.fit_transform(parameters['family_name'])

    parameters.to_csv(output_dir + '/parameters_with_targets.csv')

    y = np_utils.to_categorical(parameters['family_id'])

    print('output shape: (samples, encoded targets):', y.shape)
    print('target labels:', instr_family_le.classes_,
        instr_family_le.transform(instr_family_le.classes_))
    class_count = len(instr_family_le.classes_)
    print('number of classes:', class_count)

    ## Split the dataset

    def split_dataset(index, random_state, test_ratio=0.2, valid_ratio=0.2):
        index = list(index)
        ix_train, ix_test = train_test_split(index, test_size=test_ratio,
            random_state=random_state)
        ix_train, ix_valid = train_test_split(ix_train,
            test_size=valid_ratio / (1 - test_ratio), random_state=random_state)
        return {'train': ix_train, 'valid': ix_valid, 'test': ix_test}

    split_seed = 42
    ix = split_dataset(parameters.index, random_state=split_seed)

    with open(output_dir + '/splits.json', 'w') as f:
        json = jsonpickle.encode({'indices': ix, 'seed': split_seed})
        f.write(json)

    def splits_to_df(split_incides):
        df = pd.DataFrame([(v, key) for (key, values) in split_incides.items() for v in values], columns=['index', 'split'])
        df.sort_values('index', inplace=True)
        return df

    splits_to_df(ix).to_csv(output_dir + '/splits.csv', index=None)

    print('split sizes:')
    for key, value in ix.items():
        print(key, len(value))

    ## Scale the features

    scaler = MinMaxScaler()
    scaler.fit(x[ix['train']].reshape(len(ix['train']), -1))
    x = scaler.transform(x.reshape(len(x), -1)).reshape(-1, *x.shape[1:])

    np.savez_compressed(
        '{}/features_targets_split_seed_{}.npz'.format(output_dir, split_seed),
        x=x, y=y)

    with open(input_dir + '/chromagram_transformer.json', 'r') as f:
        chromagram_transformer = ChromagramTransformer(**jsonpickle.decode(f.read()))

    with open(model_dir + '/preproc_transformers.json', 'w') as f:
        json = jsonpickle.encode((instr_family_le, scaler,
            chromagram_transformer))
        f.write(json)

    return (x, y, ix, instr_family_le, scaler, input_shape, class_count)


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
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir + '/model_arch.yaml', 'w') as f:
        f.write(model.to_yaml())

    training_hist = model.fit(
        x[ix['train']], y[ix['train']],
        validation_data=(x[ix['valid']], y[ix['valid']]),
        batch_size=batch_size, nb_epoch=epoch_count,
        verbose=1)

    model.save_weights(output_dir + '/model_weights.h5') # HDF5

    return model, training_hist


def evaluate_model(model, x, y, ix, training_hist, output_dir):
    y_valid_pred = model.predict_classes(x[ix['valid']])
    y_valid_pred_proba = model.predict_proba(x[ix['valid']])

    os.makedirs(output_dir, exist_ok=True)

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

if __name__ == '__main__':
    model_dir = 'data/working/single-notes-2000/model'

    x, y, ix, instr_family_le, scaler, input_shape, class_count = \
        prepare_inputs(
            'data/prepared/single-notes-2000',
            'data/working/single-notes-2000/ml-inputs',
            model_dir)

    model = create_model(input_shape, class_count)
    model, training_hist = train_model(model,
        x, y, ix,
        model_dir,
        epoch_count=20)

    evaluate_model(model, x, y, ix, training_hist, 'data/working/single-notes-2000/evaluation')
