"""
An example script that generates a simple sequence of chords in the all available MIDI instruments - into MIDI and FLAC.
"""

import music21
from music21 import chord, stream
from midi2audio import FluidSynth
import os.path
from instruments import midi_instruments

def make_instrument(midi_instrument_id):
    i = music21.instrument.Instrument()
    i.midiProgram = midi_instrument_id
    return i

def generate_chords(midi_instrument_id, output_file):
    s = stream.Stream()
    s.append(make_instrument(midi_instrument_id))
    s.append(chord.Chord(["G","B","D", "F"]))
    s.append(chord.Chord(["D", "F", "A", "C"]))
    s.append(chord.Chord(["C", "E", "G", "B"]))
    s.write('midi', output_file)

if __name__ == '__main__':
    data_dir = 'data/working/chords-in-many-instruments'
    midi_dir = '{}/midi'.format(data_dir)
    audio_dir = '{}/flac'.format(data_dir)
    for path in [midi_dir, audio_dir]:
        os.makedirs(path, exist_ok=True)
    for index, instrument in midi_instruments().iterrows():
        file_name = 'instrument_{0:03d}_{1}'.format(instrument['id'], instrument['name'])
        midi_file = '{}/{}.mid'.format(midi_dir, file_name)
        audio_file = '{}/{}.flac'.format(audio_dir, file_name)
        generate_chords(instrument['id'], midi_file)
        FluidSynth().midi_to_audio(midi_file, audio_file)
