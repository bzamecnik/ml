'''
This script allows to split the dataset of songs into training/validation/test
splits. It is done at the song granularity in order to prevent leaking
information within each song (compared to splitting at block level).

Also this approach is invariant to the block/hop size of features like
chromagram. This allows to compare various feature types.

The output is a TSV file containing information on which song is in which split
and its relative order within the split.
'''

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

def split_songs(song_file, song_index_file, split_index_file, random_state):
    df = pd.read_csv(song_file, sep='\t', header=None, names=['path'])
    songs = np.array([p.split('/') for p in df['path']])
    df['artist'] = songs[:, 0]
    df['album'] = songs[:, 1]
    df['song'] = songs[:, 2]

    def split_dataset(index, random_state):
        index = list(index)
        ix_train, ix_test = train_test_split(index, test_size=0.2, random_state=random_state)
        ix_train, ix_valid = train_test_split(ix_train, test_size=0.2 / (1 - 0.2), random_state=random_state)
        return {'train': ix_train, 'valid': ix_valid, 'test': ix_test}

    split_incides = split_dataset(df.index, random_state)

    df['split'] = ''
    for name in split_incides:
        df['split'].ix[split_incides[name]] = name

    df['order'] = np.hstack([split_incides['train'], split_incides['valid'], split_incides['test']])

    df.to_csv(song_index_file, sep='\t', index=None)

    with open(split_index_file, 'w') as file:
        for name in split_incides:
            print(name + '\t' + ','.join([str(i) for i in split_incides[name]]), file=file)

if __name__ == '__main__':
    data_dir = '../data/beatles'

    song_file = data_dir + '/isophonic-songs.txt'
    song_index_file = data_dir + '/songs-dataset-split.tsv'
    split_index_file = data_dir + '/dataset-split-indexes.tsv'

    split_songs(song_file, song_index_file, split_index_file, random_state=42)
