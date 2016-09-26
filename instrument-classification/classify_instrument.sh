# Classifies an audio clip with an musical instrument family using some fixed
# model.
#
# Example usage:
#
# $ ./classify_instrument.sh data/prepared/single-notes-2000/sample_000000.flac
# brass
#
# It takes around 3 sec.

AUDIO_FILE=$1
python classify_instrument.py \
  ${AUDIO_FILE} \
  -m data/working/single-notes-2000/model \
  -p data/working/single-notes-2000/ml-inputs/preproc_transformers.json \
  -c data/prepared/single-notes-2000/chromagram_transformer.json \
  2>/dev/null
