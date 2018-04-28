"""
Extracts pitchgram features from audio data.

Input: a 2D tensor with set of audio samples
Output: A tensor with a pitchgram for each sample

The data is requantized to the non-overlapping output frames of size same as
the input frame hop size.
"""

import argparse
import jsonpickle
import numpy as np
import os
from tfr import PitchgramTransformer

from generate_audio_samples import SingleToneDataset


def extract_pitchgrams(audio_dir, feature_dir, block_size, hop_size, bin_range, bin_division):
    print('loading dataset from:', audio_dir)
    dataset = SingleToneDataset(audio_dir)
    print('dataset shape:', dataset.samples.shape)

    ch = PitchgramTransformer(sample_rate=dataset.sample_rate,
        frame_size=block_size, hop_size=hop_size,
        bin_range=bin_range, bin_division=bin_division)

    os.makedirs(feature_dir)

    with open(feature_dir + '/pitchgram_transformer.json', 'w') as f:
        json = jsonpickle.encode(ch.get_params())
        f.write(json)

    pitchgrams = np.dstack(ch.transform(dataset.samples[i])
        for i in range(len(dataset.samples)))
    pitchgrams = np.rollaxis(pitchgrams, 2)

    print('chomagrams shape:', pitchgrams.shape)

    pitchgram_file = feature_dir + '/pitchgrams.npz'
    print('saving pitchgrams to:', pitchgram_file)
    np.savez_compressed(pitchgram_file, pitchgrams)

    return pitchgrams


def parse_args():
    parser = argparse.ArgumentParser(description='Extract pitchgram features')
    parser.add_argument('audio_dir', metavar='AUDIO_DIR', type=str,
        help='input directory with audio files')
    parser.add_argument('feature_dir', metavar='FEATURE_DIR', type=str,
        help='output directory with features')
    parser.add_argument('-b', '--block-size', type=int, default=4096,
        help='block size')
    parser.add_argument('-p', '--hop-size', type=int, default=2048,
        help='hop size')
    parser.add_argument('-r', '--bin-range', type=int, nargs=2,
        default=[-48, 67], help='pitchgram bin range')
    parser.add_argument('-d', '--bin-division', type=int, default=1,
        help='bins per semitone in pitchgram')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    extract_pitchgrams(args.audio_dir, args.feature_dir,
        args.block_size, args.hop_size, args.bin_range, args.bin_division)
