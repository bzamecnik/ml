"""
Synthesizes an audio file from MIDI using some basic sound font.

It's a wrapper over fluidsynth which provides a bit easier interface.

Input: MIDI file
Output: audio file (WAV, FLAC, ...) - type determined from the extension

Usage in shell:

# use some default sound font
$ python fluidsynth.py input.mid output.flac
# use a custom sound font
$ python fluidsynth.py -s sound_font.sf2 input.mid output.flac

Usage in Python:

FluidSynth('sound_font.sf2').midi_to_audio('input.mid', 'output.wav')
"""

import argparse
from os.path import expanduser
import subprocess

default_sound_font = expanduser("~/Library/Audio/Sounds/Banks/fluid_r3_gm.sf2")

class FluidSynth():
    def __init__(self, sound_font=default_sound_font):
        self.sound_font = sound_font

    def midi_to_audio(self, midi_file, audio_file):
        subprocess.call(['fluidsynth', '-ni', self.sound_font, midi_file, '-F', audio_file])

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MIDI to audio via FluidSynth')
    parser.add_argument('midi_file', metavar='MIDI', type=str)
    parser.add_argument('audio_file', metavar='AUDIO', type=str)
    parser.add_argument('-s', '--sound-font', type=str,
        default=default_sound_font)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    FluidSynth(args.sound_font).midi_to_audio(args.midi_file, args.audio_file)
