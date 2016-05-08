# Prepares data for simple chord classification.
# It computes the chromagram from the audio file and splits the labels to blocks.

import numpy as np
import pandas as pd
import sys
import math
import os

sys.path.append('music-processing-experiments')

from analysis import split_to_blocks
from files import load_wav
from time_intervals import block_labels
from spectrogram import create_window
from reassignment import chromagram

def prepare_chomagram_and_labels(
    album,
    song_title,
    block_size,
    hop_size,
    bin_range,
    bin_division):

    song = 'The_Beatles/'+album+'/'+song_title
    data_dir = 'data/beatles'
    audio_file = data_dir + '/audio-cd/' + song + '.wav'
    chord_file = data_dir  + '/chordlab/' + song + '.lab.pcs.tsv'
    audio_file, chord_file

    # ## Load audio
    print('loading audio:', audio_file)

    x, fs = load_wav(audio_file)

    print('sampling rate:', fs, 'Hz')
    print('number of samples:', len(x))
    print('duration in audio:', len(x) / fs, 'sec')

    # ## Load chords
    print('loading chords:', chord_file)
    chords = pd.read_csv(chord_file, sep='\t')
    print('shape:', chords.shape)
    print('duration in chords:', chords['end'].iloc[-1])

    pcs_cols = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
    label_cols = ['label','root','bass'] + pcs_cols

    # ## Split audio to blocks

    x_blocks, x_times = split_to_blocks(x, block_size, hop_size, fs)
    print('blocks shape:', x_blocks.shape)
    print('number of blocks:', len(x_blocks))
    # start times for each block
    print('last block starts at:', x_times[-1], 'sec')

    # ## Mapping of chords to blocks

    def chords_to_blocks(chords, block_center_times):
        chord_ix = 0
        for t in block_center_times:
            yield chords.iloc[i][pcs_cols]

    def time_to_samples(time):
        return np.round(time * fs)

    chords['start_sample'] = time_to_samples(chords['start'])
    chords['end_sample'] = time_to_samples(chords['end'])
    df_blocks = pd.DataFrame({'start': time_to_samples(x_times).astype(np.int64)})
    df_blocks['end'] = df_blocks['start'] + block_size

    label_dict = chords[label_cols].drop_duplicates().set_index('label')

    df_labels = chords[['start_sample', 'end_sample', 'label']].copy()
    df_labels.rename(columns={'start_sample': 'start', 'end_sample': 'end'}, inplace=True)

    df_labelled_blocks = block_labels(df_blocks, df_labels)

    df_block_pcs = df_labelled_blocks[['label']].join(label_dict, on='label')[['label'] + pcs_cols]

    assert len(df_block_pcs) == len(df_blocks)

    block_labels_file = '{}/chord-pcs/{}_{}/{}.pcs'.format(data_dir, block_size, hop_size, song)
    print('block labels file:', block_labels_file)

    os.makedirs(os.path.dirname(block_labels_file), exist_ok=True)
    df_block_pcs.to_csv(block_labels_file, sep='\t', index=False)

    # ## Chromagram features

    w = create_window(block_size)
    X_chromagram = chromagram(x_blocks, w, fs, to_log=True, bin_range=bin_range, bin_division=bin_division)

    chromagram_file = '{}/chromagram/block={}_hop={}_bins={},{}_div={}/{}.npz'.format(
        data_dir, block_size, hop_size, bin_range[0], bin_range[1], bin_division, song)

    print('chomagram file:', chromagram_file)

    os.makedirs(os.path.dirname(chromagram_file), exist_ok=True)
    np.savez_compressed(chromagram_file, X=X_chromagram, times=x_times)

def transform_all():
    # takes an hour - could be optimized...
    import timeit

    block_size = 4096
    hop_size = 2048
    bin_range = (-48, 67)
    bin_division = 1

    with open('data/beatles/isophonic-songs.txt') as file:
        for line in file.readlines():
            band, album, song_title = line.rstrip('\n').split('/')
            print(album, song_title)
            prepare_chomagram_and_labels(album, song_title, block_size, hop_size, bin_range, bin_division)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare data for chord classification - chromagram and labels.')
    parser.add_argument('ALBUM')
    parser.add_argument('SONG_TITLE')
    parser.add_argument('-b', '--block-size', type=int, default=4096, help='block size')
    parser.add_argument('-p', '--hop-size', type=int, default=2048, help='hop size')
    parser.add_argument('-r', '--bin-range', type=int, nargs=2, default=[-48, 67], help='chromagram bin range')
    parser.add_argument('-d', '--bin-division', type=int, default=1, help='bins per semitone in chromagram')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    prepare_chomagram_and_labels(
        album=args.ALBUM,
        song_title=args.SONG_TITLE,
        block_size=args.block_size,
        hop_size=args.hop_size,
        bin_range=args.bin_range,
        bin_division=args.bin_division)

# example usage:
# $ python prepare_data_for_chord_classification.py \
#   "03_-_A_Hard_Day's_Night" "05_-_And_I_Love_Her" \
#   -b 4096 -p 2048 -r -48 67 -d 1
