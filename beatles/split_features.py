'''
This script allows to split the particular features (chromagram) and labels
according to previously computed song-level splits.

All features (X) and labels (Y) in each split are concatenated and stored as
single files.

The reason is to prepare input data for the machine learning phase that is most
convenient.
'''

import numpy as np
import pandas as pd
import os

index_file = 'data/beatles/songs-dataset-split.tsv'
df_index = pd.read_csv(index_file, sep='\t')

df_index.sort('order', inplace=True)

labels_dir = 'data/beatles/chord-pcs/4096_2048/'
features_dir = 'data/beatles/chromagram/block=4096_hop=2048_bins=-48,67_div=1/'

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

target_dir = 'data/beatles/ml_dataset/chromagram_block=4096_hop=2048_bins=-48,67_div=1/'

def merge_split(split):
    print('--- split: ', split, '---')
    X, Y = load_songs(df_index[df_index['split'] == split])
    target_file = target_dir + split + '.npz'
    print('saving to:', target_file)
    np.savez_compressed(target_file, X=X, Y=Y)

os.makedirs(target_dir, exist_ok=True)

for split in ('train', 'valid', 'test'):
    merge_split(split)
