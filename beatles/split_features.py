'''
This script allows to split the particular features (chromagram) and labels
according to previously computed song-level splits (given by an index).

All features (X) and labels (Y) in each split are concatenated. All splits are
stored in a single compressed numpy file. It contains X and Y arrays for each
split: X_train, Y_train, X_valid, Y_valid, X_test, Y_test.

The reason is to prepare input data for the machine learning phase that is most
convenient.
'''

import numpy as np
import pandas as pd
import os

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
    index_file = 'data/beatles/songs-dataset-split.tsv'
    labels_dir = 'data/beatles/chord-pcs/4096_2048/'
    features_dir = 'data/beatles/chromagram/block=4096_hop=2048_bins=-48,67_div=1/'
    target_file = 'data/beatles/ml_dataset/chromagram_block=4096_hop=2048_bins=-48,67_div=1/dataset.npz'

    split_songs_and_save_to_single_file(index_file, labels_dir, features_dir, target_file)
