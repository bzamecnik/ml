# Musical Instrument Classification

The goal of this experiment is to train a model to classify musical instrument from a sample of audio.

We first limit ourselves to fixed-length samples of single harmonic tone played by synthesized [MIDI instruments](https://www.midi.org/specifications/item/gm-level-1-sound-set).

## Datasets:

- `data/midi-instruments.csv`
  - "General MIDI Level 1 Instrument Patch Map"
  - source: [MIDI specification](https://www.midi.org/specifications/item/gm-level-1-sound-set)
  - 128 MIDI instruments, 16 families and tonal and harmonic boolean attributes
  - columns:
    - id - numeric instrument identifier (1-128)
    - name - instrument identifier
    - desc - human-readable instrument name
    - family_name - instrument family identifier
    - family_desc - human readable instrument family name
    - tonal - indicates a tonal instrument (True/False)
    - harmonic - indicates a harmonic instrument (True/False)

## MIDI to audio synthesis

We'll use [FluidSynth](http://www.fluidsynth.org) for synthesizing audio from MIDI and some sound fonts:

- [GeneralUser GS](http://www.schristiancollins.com/generaluser.php)

### Installation on OS X

`./install_fluidsynth_with_soundfonts_osx.sh`

### Usage

We can synthesize MIDI files using sound fonts to audio files in various formats (eg. WAV, FLAC, etc.). The output format is determined by the file extension.

Either use fluidsynth direcly:

```
fluidsynth -ni sound_font.sf2 input.mid -F output.wav
```

Or use a wrapper with a simpler interface:

```
# convert MIDI to WAV (with a default sound font)
python fluidsynth.py input.mid output.wav
# convert MIDI to FLAC
python fluidsynth.py input.mid output.flac

# playback MIDI
python fluidsynth.py input.mid

# convert MIDI to audio with a specific sound font
python fluidsynth.py -s sounf_font.sf2 input.mid output.flac
```

The default sound font for `fluidsynth.py` is stored in `~/.fluidsynth/default_sound_font`.

## Generating Datasets

### generate_chords.py

Just an example of how to generate a MIDI file with a sequence of chords using the `music21` library.

### generate_audio_samples.py

A script to generate a single-note audio sample to MIDI where several parameters (like pitch, volume, duration, instrument) can be specified.

Genrate a dataset:

```
# time spent: 2m24.281s:
time python generate_audio_samples.py -c 2000 -s 42 -o data/working/random-notes-2000 -f flac
```

Then load it (2000 samples of 2 seconds length at 44110 Hz sampling rate):

```
>>> dataset = SingleToneDataset('data/working/random-notes-2000')
>>> dataset.samples.shape
(2000, 88200)
```
