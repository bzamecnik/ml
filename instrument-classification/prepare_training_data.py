"""
Prepares features and targets and creates split indexes.

Chromagram features are scaled and reshaped, targets are encoded.
"""

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from keras.utils import np_utils
import numpy as np
import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from instruments import midi_instruments
from preprocessing import ChromagramTransformer, ConvolutionReshaper

jsonpickle_numpy.register_handlers()


def prepare_inputs(input_dir, output_dir, split_seed=42):
    os.makedirs(output_dir, exist_ok=True)

    ## Load features

    x = np.load(input_dir + '/chromagrams.npz')['arr_0']
    # axes: data point, block, chroma vector
    print('x.shape:', x.shape)
    print('x.size:', x.size)

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
    # NOTE: The reshapes are necessary since the trainsformer expects 1D values
    # and we have 2D values.
    x = scaler.transform(x.reshape(len(x), -1)).reshape(-1, *x.shape[1:])

    ## Reshape for the convolution filtering

    x = ConvolutionReshaper().transform(x)
    print('x.shape (for convolution):', x.shape)

    input_shape = x.shape[1:]
    print('input shape (rows, cols, filters):', input_shape)

    np.savez_compressed(
        '{}/features_targets.npz'.format(output_dir, split_seed),
        x=x, y=y)

    with open(input_dir + '/chromagram_transformer.json', 'r') as f:
        chromagram_transformer = ChromagramTransformer(**jsonpickle.decode(f.read()))

    with open(output_dir + '/preproc_transformers.json', 'w') as f:
        json = jsonpickle.encode((instr_family_le, scaler,
            chromagram_transformer))
        f.write(json)


def load_data(data_dir):
    data = np.load(data_dir + '/features_targets.npz')
    x, y = data['x'], data['y']

    with open(data_dir + '/splits.json', 'r') as f:
        ix = jsonpickle.decode(f.read())['indices']

    return x, y, ix


def load_transformers(data_dir):
    with open(data_dir + '/preproc_transformers.json', 'r') as f:
        # instr_family_le, scaler, chromagram_transformer
        return jsonpickle.decode(f.read())


if __name__ == '__main__':
    prepare_inputs(
        'data/prepared/single-notes-2000',
        'data/working/single-notes-2000/ml-inputs')
