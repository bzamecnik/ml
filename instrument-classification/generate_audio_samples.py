"""
This generates a single-tone audio clip for instrument classification.
"""

import music21
from music21.chord import Chord
from music21.duration import Duration
from music21.instrument import Instrument
from music21.note import Note
from music21.stream import Stream
from music21.tempo import MetronomeMark
from music21.volume import Volume

def make_instrument(id):
    i = Instrument()
    i.midiProgram = id
    return i

def chord_with_volume(chord, volume):
    chord.volume = Volume(velocityScalar=volume)
    return chord

def generate_single_note(midi_number, midi_instrument, volume, duration, tempo):
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
        make_instrument(midi_instrument),
        chord_with_volume(Chord([
            Note(midi=midi_number, duration=Duration(duration))
        ]), volume)
    ])

def write_midi(stream, output_file):
    stream.write('midi', output_file)

if __name__ == '__main__':
    # example
    stream = generate_single_note(midi_number=60, midi_instrument=2, volume=1.0, duration=0.5, tempo=120)
    write_midi(stream, 'data/working/example-parametric-note/01.midi')

# TODO:
# - create a better API
# - make a random generator of the parameters
# - produce audio samples in batch
#   - either one file per samples (many runs of FS - may be slow)
#   - or make a big MIDI (then audio) and then split (synthesize in one run)
