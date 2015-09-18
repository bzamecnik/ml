# Beatles annotations - musical dataset exploration

http://isophonics.net/content/reference-annotations-beatles

Manually annotates songs of Beatles
- song structure - segments + labels
- keys - segments + labels
- chords - segments + labels
- chroma features

## Time map visualization

https://districtdatalabs.silvrback.com/time-maps-visualizing-discrete-events-across-many-timescales

Multi-scale visualization of discrete events originating from physics of chaotic dynamic systems.
Basically scatter of time-to-previous vs. time-to-next event.

```
$ ipython notebook
```

Open `time_map_of_chord_events.ipynb` in the Jupyter (http://localhost:8888/).

## Chord labels to pitch class segments as features for further ML

`classification_based_on_chords`

- Chord labels parsed to binary pitch class sets via https://github.com/bzamecnik/chord-labels and added to the dataframe.
- It can be used to:
  - better analyses and visualizations
  - key classification based on chord contexts
  - chord classification based on chroma features
  - chord imputation and prediction
  - etc.

The classification in the notebook is really not complete...
