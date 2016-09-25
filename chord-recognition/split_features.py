'''
This script allows to split the particular features (chromagram) and labels
according to previously computed song-level splits (given by an index).

All features (X) and labels (Y) in each split are concatenated. All splits are
stored in a single compressed numpy file. It contains X and Y arrays for each
split: X_train, Y_train, X_valid, Y_valid, X_test, Y_test.

The reason is to prepare input data for the machine learning phase that is most
convenient.
'''

from collections import OrderedDict
import numpy as np
import os
import pandas as pd

def load_index(file_name):
    df_index = pd.read_csv(file_name, sep='\t')
    df_index.sort('order', inplace=True)
    return df_index

def load_song(song_path):
    df_labels = pd.read_csv(labels_dir + song_path + '.pcs', sep='\t')
    features_npz = np.load(features_dir + song_path + '.npz')
    x = features_npz['X']
    y = df_labels[df_labels.columns[1:]].as_matrix()
    return x, y

def load_songs(df_index):
    X = []
    Y = []
    for path in df_index['path']:
        print(path)
        x, y = load_song(path)
        X.append(x)
        Y.append(y)
    return np.vstack(X), np.vstack(Y)

def split_songs_and_save_to_single_file(index_file, labels_dir, features_dir, target_file):
    df_index = load_index(index_file)
    splits = {}
    for split in ('train', 'valid', 'test'):
        print('--- split: ', split, '---')
        X, Y = load_songs(df_index[df_index['split'] == split])
        splits['X_' + split] = X
        splits['Y_' + split] = Y

    print('saving to:', target_file)
    target_dir = os.path.dirname(target_file)
    os.makedirs(target_dir, exist_ok=True)
    np.savez_compressed(target_file, **splits)

if __name__ == '__main__':
    params = OrderedDict([
        ('block', '4096'), ('hop', '2048'), ('bins', '-48,67'), ('div', '1')
    ])
    feature_param_path = '_'.join('%s=%s' % (k, v) for (k, v) in params.items())

    data_dir = '../data/beatles'
    index_file = data_dir + '/songs-dataset-split.tsv'
    labels_dir = data_dir + '/chord-pcs/%s_%s/' % (params['block'], params['hop'])
    features_dir = '%s/chromagram/%s/' % (data_dir, feature_param_path)
    target_file = '%s/ml_dataset/%s/dataset.npz' % (data_dir, feature_param_path)

    split_songs_and_save_to_single_file(index_file, labels_dir, features_dir, target_file)
