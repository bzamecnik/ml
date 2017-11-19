# Audio chord classification

## Motivation

Imagine you hear some interesting song and would like to learn to play it on guitar or piano. Typically you need text (lyrics) and melody to sing and chords (multiple tones at once) for accompaniment. Sometimes are're lucky since someone transcribed them on paper (or web page) before, sometimes not. Text and melody are usually easy to pick by ear, but chords can be sometimes tricky even if you're trained well. Thus it would be nice if a machine could help us recognize what chords are there in the audio.

## Task definition

More formally the task is to split the audio recording into time segments and mark each segment with a chord label. The label can be either symbolic (eg. `Eb:min7`) or a list of tones that are active in the particular segment.

Example:

```
start	end	label	C	Db	D	Eb	E	F	Gb	G	Ab	A	Bb	B
0.000000	1.053119	N	0	0	0	0	0	0	0	0	0	0	0	0
1.053119	3.593854	B:min	0	0	1	0	0	0	1	0	0	0	0	1
3.593854	6.090000	G	0	0	1	0	0	0	0	1	0	0	0	1
6.090000	8.655804	E	0	0	0	0	1	0	0	0	1	0	0	1
8.655804	11.14034	A	0	1	0	0	1	0	0	0	0	1	0	0
```

## Solution

In this project the solution can be described as follows.

### Pre-processing & feature extraction

We don't use the raw audio directly, instead we extract some more high-level frequency-domain features.

- we have to assure the file trees of audio files and labels are matching
- the raw audio is a series of time-domain digital samples
  - eg. 44100 Hz sampling rate, 16-bit resolution, stereo (2 channels)
- stereo channels are mixed down to mono
- reassigned chromagram is computed
  - the audio is split into overlapping frames
    - eg. frame size 4096, hop size 2048 (overlap 50%)
  - a DFT is computed for each frame
  - for each DFT bin a more precise time-frequency location is computed using the derivative of phase with respect to time and frequency (time-frequency reassignment)
  - the reassigned spectrum is requantized into log-frequency bins corresponding to musical tones
  - to reduce high dynamic range of the values they're converted to decibels
- the dataset is split into training/validation/test splits across songs and stored in a form suitable for further machine learning (eg. numpy arrays)
- no adaptive scaling is done in the pre-processing phase

### Training

The classification is formulated as multi-label classification problem. The output label is a binary vector of length 12 representing active pitch classes in a chord.

- 1D convolution + pooling layers
  - + dropout
- LSTM layers
- batch normalization
- fully connected layers
  - sigmoid activations at the end
- binary cross entropy loss

### Evaluation

- basic classification metrics during training - on frames
  - accuracy
    - ratio of completely correct binary vectors
    - quite strict, coarse-grained
    - not too robust to class imbalance
  - hamming score
    - ratio of correct bits in the binary vectors
    - more fine-grained
    - not too robust to class imbalance
  - AUC (area under ROC curve)
    - quite robust to class imbalance
- main MIREX metric
  - weighted chord symbol recall
  - on the final time segments

### Prediction

- just compute the results for a new audio through the whole pipeline

### Post-processing

- median filtering (?)
- joining together adjacent frames with the same predicted labels
- converting binary vector labels to symbolic chord labels

## Installation & Dependencies

```
conda create -n tensorflow python=3.4
source activate tensorflow

pip install Keras
# Theano or TensorFlow
# TensorFlow 0.7.1 on Linux 64 needs Python 3.4 (not 3.5)
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp34-none-linux_x86_64.whl
conda install pillow scipy scikit-learn pandas h5py matplotlib seaborn
pip install arrow

cat '{"floatx": "float32", "backend": "tensorflow", "epsilon": 1e-07}' > \
  ~/.keras/keras.json
```

Note (2017-11-19): It surprisingly works after 1.5 year even with Keras 2.0.9 + TensorFlow 1.4! Wow.

### Data

I have original cleaned data in FLAC available, but cannot publish due to copyright restrictions. You can drop me a message...

Preprocessed data (chromagram features + pitch class targets) - cca 332 MB:

- block size: 4096
- hop size: 2048
- range of pitch bins: -48,67
- bin subdivision: 1

```shell
wget https://data.neural.cz/music/isophonics/beatles/preproc/dataset_2016-05-15.npz
```

### Troubleshooting

- `import scipy.misc` gives Segmentation fault
  - install scipy via conda instead of pip
- `import scipy.misc` - no module found
  - PIL is missing
  - first install pillow, then scipy (possibly reinstall it)
- tensorflow 0.8.0 crashes saving PNG images from matplotlib
  - https://github.com/tensorflow/tensorflow/issues/1927
  - 0.7.1 works ok
- Theano worked but now waits on some lock indefinitely :/
