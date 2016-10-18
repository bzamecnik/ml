"""
Extracts chromagram features from audio data.

Input: a 2D tensor with set of audio samples
Output: A tensor with a chromagram for each sample
"""

import argparse
import jsonpickle
import numpy as np
from tfr.preprocessing import ChromagramTransformer

from generate_audio_samples import SingleToneDataset


def extract_chromagrams(data_dir, block_size, hop_size, bin_range, bin_division):
    print('loading dataset from:', data_dir)
    dataset = SingleToneDataset(data_dir)
    print('dataset shape:', dataset.samples.shape)

    ch = ChromagramTransformer(dataset.sample_rate,
        block_size, hop_size, bin_range, bin_division)

    with open(data_dir + '/chromagram_transformer.json', 'w') as f:
        json = jsonpickle.encode(ch.get_params())
        f.write(json)

    chromagrams = np.dstack(ch.transform(dataset.samples[i])
        for i in range(len(dataset.samples)))
    chromagrams = np.rollaxis(chromagrams, 2)

    print('chomagrams shape:', chromagrams.shape)

    chromagram_file = data_dir + '/chromagrams.npz'
    print('saving chromagrams to:', chromagram_file)
    np.savez_compressed(chromagram_file, chromagrams)

    return chromagrams


def parse_args():
    parser = argparse.ArgumentParser(description='Extract chromagram features')
    parser.add_argument('data_dir', metavar='DATA_DIR', type=str,
        help='data directory (both audio/features)')
    parser.add_argument('-b', '--block-size', type=int, default=4096,
        help='block size')
    parser.add_argument('-p', '--hop-size', type=int, default=2048,
        help='hop size')
    parser.add_argument('-r', '--bin-range', type=int, nargs=2,
        default=[-48, 67], help='chromagram bin range')
    parser.add_argument('-d', '--bin-division', type=int, default=1,
        help='bins per semitone in chromagram')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    extract_chromagrams(args.data_dir,
        args.block_size, args.hop_size, args.bin_range, args.bin_division)
