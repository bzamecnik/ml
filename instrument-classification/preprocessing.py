import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys

# TODO: package the chromagram script
sys.path.append('../tools/music-processing-experiments/')

from analysis import split_to_blocks
from spectrogram import create_window
from reassignment import chromagram


class ChromagramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate=44100, block_size=4096, hop_size=2048,
        bin_range=[-48, 67], bin_division=1):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.hop_size = hop_size
        self.bin_range = bin_range
        self.bin_division = bin_division

        self.window = create_window(block_size)

    def transform(self, X, **transform_params):
        """
        Transforms audio clip X into a normalized chromagram.
        Input: X - mono audio clip - numpy array of shape (samples,)
        Ooutput: X_chromagram - numpy array of shape (blocks, bins)
        """
        X_blocks, X_times = split_to_blocks(X,
            self.block_size, self.hop_size, self.sample_rate)
        X_chromagram = chromagram(
            X_blocks,
            self.window,
            self.sample_rate,
            to_log=True,
            bin_range=self.bin_range,
            bin_division=self.bin_division)
        # map from raw dB [-120.0, bin_count] to [0.0, 1.0]
        bin_count = X_blocks.shape[1]
        X_chromagram = (X_chromagram + 120) / (120 + bin_count)
        return X_chromagram

    def fit(self, X, y=None, **fit_params):
        return self
