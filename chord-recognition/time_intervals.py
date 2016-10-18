import pandas as pd
import numpy as np
import collections

def block_labels(df_blocks, df_labels):
    '''
    Given fixed-size overlapping blocks and variable-sized non-overlapping
    labels select most suitable label for each block.
    This can be useful eg. to assign chord labels to audio blocks.

    All times are measured in samples and represented by integers.

    Inputs:
    - df_blocks: pandas DataFrame with columns start, end (in samples)
    - df_labels: pandas DataFrame with columns start, label

    Outputs:
    - df_blocks: pandas DataFrame with columns start, end, label

    In case multiple labels span a single block the label with most coverage is
    selected.
    '''
    def merge_events(df_blocks, df_labels):
        df_events = pd.merge(
            pd.concat([df_blocks[['start']], df_blocks[['end']].rename(columns={'end': 'start'})]).drop_duplicates(),
            df_labels, how='outer')
        df_events.sort('start', inplace=True)
        df_events.fillna(method='pad', inplace=True)
        df_events['duration'] = abs(df_events['start'].diff(-1))
        df_events.set_index('start', inplace=True)
        return df_events.dropna()

    df_events = merge_events(df_blocks, df_labels)

    def label_for_block(start, end):
        labels = df_events['label'].ix[start:end]
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            durations = df_events['duration'].ix[start:end]
            cnt = collections.Counter()
            for l, d in zip(labels, durations):
                cnt[l] += d
            return cnt.most_common(1)[0][0]
        else:
            return labels.iloc[0]

    def add_labels(df_blocks):
        block_labels = (label_for_block(start, end) for (i, start, end) in df_blocks.itertuples())
        df_block_labels = pd.DataFrame(block_labels, columns=['label'])
        return df_blocks.join(df_block_labels)

    return add_labels(df_blocks)

def test():
    block_size = 10
    hop_size = 5
    sample_count = 90
    block_count = (sample_count - block_size) / hop_size
    block_starts = hop_size * np.arange(block_count + 1).astype(np.int32)
    block_ends = block_starts + block_size
    blocks = list(zip(block_starts, block_ends))
    df_blocks = pd.DataFrame(blocks, columns=['start', 'end'])

    # label segment start times (the last element is the end of the last segment)
    label_times = [0, 25, 38, 50, 60, 64, 68, 80, 81, 84, 89]
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'H', 'N']
    df_labels = pd.DataFrame({'start': label_times, 'label': labels}, columns=['start', 'label'])

    df_labelled_blocks = block_labels(df_blocks, df_labels)

    expected_labels = ['A','A','A','A','B','B','B','C','C','D','D','D','G','G','G','H','H']
    actual_labels = list(df_labelled_blocks['label'])
    for s, e, a in zip(block_starts, expected_labels, actual_labels):
        print(s, e, a, '*' if e != a else '')
    assert actual_labels == expected_labels
