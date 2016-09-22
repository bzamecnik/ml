"""
This generates a single-tone audio clip for instrument classification.
"""

import argparse
from music21.chord import Chord
from music21.duration import Duration
from music21.instrument import Instrument
from music21.note import Note, Rest
from music21.stream import Stream
from music21.tempo import MetronomeMark
from music21.volume import Volume
import numpy as np
import os
import pandas as pd
import soundfile as sf

from fluidsynth import FluidSynth
from instruments import midi_instruments

def make_instrument(id):
    i = Instrument()
    i.midiProgram = id
    return i

def chord_with_volume(chord, volume):
    chord.volume = Volume(velocityScalar=volume)
    return chord

def write_midi(stream, output_file):
    stream.write('midi', output_file)

def generate_single_note(midi_number, midi_instrument=0, volume=1.0, duration=1.0, tempo=120):
    """
    Generates a stream containing a single note with given parameters.
    midi_number - MIDI note number, 0 to 127
    midi_instrument - MIDI intrument number, 0 to 127
    duration - floating point number (in quarter note lengths)
    volume - 0.0 to 1.0
    tempo - number of quarter notes per minute (eg. 120)

    Note that there's a quarter note rest at the beginning and at the end.
    """
    return Stream([
        MetronomeMark(number=tempo),
        make_instrument(int(midi_instrument)),
        chord_with_volume(Chord([
            Note(midi=int(midi_number), duration=Duration(duration))
        ]), volume)
    ])

def generate_separete_notes(note_params_df, midi_dir, audio_dir, audio_format='flac'):
    """
    Generates a batch of single note samples from the given table of parameters.

    `note_params_df` - a Pandas Dataframe with columns:
    `midi_number, midi_instrument, volume, duration, tempo`. Their meaning is the same as in generate_single_note.
    `output_dir` - output directory for the MIDI files

    Each sample goes to a single MIDI file named by the numeric index. Also each synthesized audio sample goes to a
    """
    for d in [midi_dir, audio_dir]:
        os.makedirs(d, exist_ok=True)
    fs = FluidSynth()
    for i, row in note_params_df.iterrows():
        midi_file = '{0}/{1:06d}.midi'.format(midi_dir, i)
        audio_file = '{0}/{1:06d}.{2}'.format(audio_dir, i, audio_format)

        print(row, midi_file, audio_file)

        stream = generate_single_note(**row)
        write_midi(stream, midi_file)
        fs.midi_to_audio(midi_file, audio_file)

def random_params(n, note_range=None, volume_range=(0.5, 1.0), duration=1.0, tempo=120, seed=None):
    """
    Generate note parameters randomly as a DataFrame.

    n - number of samples
    """

    if seed is not None:
        np.random.seed(seed)

    instruments = midi_instruments()

    def instrument_range(i):
        instr = instruments.ix[i]
        instr_range = np.array([instr['min_pitch'], instr['max_pitch']])
        if note_range is not None:
            instr_range = np.clip(instr_range, *note_range)
        return instr_range

    allowed_instruments = np.hstack([
            np.arange(0, 8), # piano
            np.arange(16, 32), # organ, guitar
            np.arange(40, 48), # strings
            np.arange(56, 80), # brass, reed, pipe
        ])

    def random_note_for_instrument(instr):
        instr_range = instrument_range(instr)
        return np.random.random_integers(low=instr_range[0], high=instr_range[1], size=1)[0]

    df = pd.DataFrame()
    df['midi_instrument'] = np.random.choice(allowed_instruments, size=n)
    df['midi_number'] = df['midi_instrument'].apply(random_note_for_instrument)
    df['volume'] = np.random.uniform(low=volume_range[0], high=volume_range[1], size=n)
    # TODO: allow varying duration while maintaining constant audio length
    df['duration'] = duration
    df['tempo'] = tempo

    return df

def generate_notes_in_batch(note_params_df, midi_dir, audio_dir, audio_format='flac'):
    """
    Generates a batch of single note samples from the given table of parameters.

    `note_params_df` - a Pandas Dataframe with columns:
    `midi_number, midi_instrument, volume, duration, tempo`. Their meaning is the same as in generate_single_note.
    `output_dir` - output directory for the MIDI files

    Each sample goes to a single MIDI file named by the numeric index. Also each synthesized audio sample goes to a
    """
    for d in [midi_dir, audio_dir]:
        os.makedirs(d, exist_ok=True)

    fs = FluidSynth()

    stream = Stream()
    stream.append(MetronomeMark(number=60))

    for i, row in note_params_df.iterrows():
        stream.append(make_instrument(int(row['midi_instrument'])))
        duration = Duration(1.0)
        # TODO: use duration from the parameters DF
        stream.append(chord_with_volume(Chord([
            Note(midi=int(row['midi_number']), duration=duration)
        ]), row['volume']))
        stream.append(Rest(duration=Duration(2.0)))

    midi_file = '{0}/batch.midi'.format(midi_dir)
    audio_file = '{0}/batch_all.{1}'.format(audio_dir, audio_format)

    write_midi(stream, midi_file)
    fs.midi_to_audio(midi_file, audio_file)
    cut_audio_samples(audio_file, audio_dir, len(note_params_df), 3.0, 0.5, audio_format)

def cut_audio_samples(input_file, output_dir, n_parts, part_duration, margin_duration, audio_format):
    x, fs = sf.read(input_file)
    x = x.mean(axis=1) # convert to mono
    for i in range(n_parts):
        # let's have larger margin to prevent spilling the content
        # 1 second margin, 1 second note, 1 second margin
        part_samples = part_duration * fs # in samples
        # then cut the margin down a bit
        margin_samples = int(margin_duration * fs)
        start = i * part_samples + margin_samples
        end = (i + 1) * part_samples - margin_samples
        x_part = x[start:end]
        audio_file = output_dir + '/batch_{0:03d}.{1}'.format(i, audio_format)
        print(start/fs, end/fs, audio_file)
        sf.write(audio_file, x_part, fs)

def generate_random_samples(args):
    params_df = random_params(args.count, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    params_df.to_csv(args.output_dir + '/parameters.csv')
    midi_dir = args.output_dir + '/midi'
    audio_dir = args.output_dir + '/' + args.audio_format
    # generate_separete_notes(params_df, midi_dir, audio_dir, args.audio_format)
    generate_notes_in_batch(params_df, midi_dir, audio_dir, args.audio_format)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate random audio samples.')
    parser.add_argument('-c', '--count', type=int, help='number of samples')
    parser.add_argument('-s', '--seed', type=int, help='random seed')
    parser.add_argument('-o', '--output-dir', type=str, help='output directory')
    parser.add_argument('-f', '--audio-format', type=str, default='flac', help='audio format (flac, wav)')

    return parser.parse_args()

if __name__ == '__main__':
    generate_random_samples(parse_args())

# TODO: split into two parts with separate responsibilities:
# - randomly generate the parameters to a CSV file
# - synthesize sounds from a given CSV file
