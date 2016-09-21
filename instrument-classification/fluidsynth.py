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

If no sound font is specified explicitly the path of a default one is searched in the `~/.fluidsynth/default_sound_font` file.
"""

import argparse
from os.path import expanduser
import subprocess

class FluidSynth():
    def __init__(self, sound_font):
        if sound_font is not None:
            self.sound_font = sound_font
        else:
            try:
                with(open(expanduser('~/.fluidsynth/default_sound_font'))) as f:
                    self.sound_font = f.readline().strip()
            except Exception as ex:
                raise RuntimeError('Default sound font is not defined, you have to specify it explicitly') from ex

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
