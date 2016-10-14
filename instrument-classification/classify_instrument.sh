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
python predict.py \
  ${AUDIO_FILE} \
  -m data/working/single-notes-2000/model \
  2>/dev/null
