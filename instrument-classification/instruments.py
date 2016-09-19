import pandas as pd

def midi_instruments(definition_file='data/midi-instruments.csv'):
    """Provides a Pandas DataFrame with MIDI insturments definitions."""
    return pd.read_csv(definition_file, sep=',')
