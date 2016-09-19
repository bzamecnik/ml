# Musical Instrument Classification

The goal of this experiment is to train a model to classify musical instrument from a sample of audio.

We first limit ourselves to fixed-length samples of single harmonic tone played by synthesized [MIDI instruments](https://www.midi.org/specifications/item/gm-level-1-sound-set).

## Datasets:

- `data/midi-instruments.csv`
  - "General MIDI Level 1 Instrument Patch Map"
  - source: https://www.midi.org/specifications/item/gm-level-1-sound-set
  - 128 MIDI instruments, 16 families and tonal and harmonic boolean attributes
  - columns:
    - id - numeric instrument identifier (1-128)
    - name - instrument identifier
    - desc - human-readable instrument name
    - family_name - instrument family identifier
    - family_desc - human readable instrument family name
    - tonal - indicates a tonal instrument (True/False)
    - harmonic - indicates a harmonic instrument (True/False)
