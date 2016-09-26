import numpy as np
import pandas as pd

# TODO
# - convert segments to frames
# - convert frames to segments

def frames_to_segments(df_frames, total_duration=None):
    """
    Converts a dataframe with labelled frames to segments.
    It merges adjacent frames with equal labels.
    Both input and output dataframes have columns ['start', 'end', 'label'].
    The exact total duration (end time) can be optionally specified.
    Times are in seconds.
    """
    df = df_frames.copy()
    labels = df['label']
    segment_start = labels != labels.shift(1)
    df_segments = df[segment_start].copy()
    end_time = total_duration if total_duration else df['end'].iloc[-1]
    df_segments['end'] = df_segments['start'].shift(-1).fillna(end_time)
    return df_segments

def frame_label_tsv_to_segment_tsv(input_file, output_file, fs=44100., hop_size=2048, total_duration=None):
    # no time, just 12 PCS labels
    def read_label_file(file_name):
        with open(file_name) as file:
            return [line.replace('\n', '').replace('\t', '') for line in file.readlines()]

    def read_labels_as_df(file_name):
        labels = read_label_file(file_name)
        # compute frame start and end times
        hop_duration = hop_size / fs
        start_times = hop_duration * np.arange(len(labels))
        return pd.DataFrame({
            'start': start_times,
            'end': hop_duration + start_times,
            'label': labels},
            columns=['start', 'end', 'label'])

    def save_tsv(df, file_name):
        df.to_csv(file_name, sep='\t', index=None, float_format='%.6f')

    def explode_pitch_classes(df):
        df = df.copy()
        labels = df['label']
        pcs = np.array([[p for p in label] for label in df_segments['label']]).T
        pcs_cols = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
        for i, col in enumerate(pcs_cols):
            df[col] = pcs[i]
        del df['label']
        return df

    df_frames = read_labels_as_df(input_file)
    df_segments = frames_to_segments(df_frames)
    save_tsv(df_segments)
