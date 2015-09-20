# Extract binary pitch class sets + root and bass pitch classes from chord labels.

# $ tools/add_pitch_class_sets.sh data/beatles/chordlabs

export CHORD_LABELS=tools/chord-labels-1.0/bin/chord-labels
for INPUT in $(find $1 -name '*.lab'); do
  echo $INPUT
  OUTPUT="$INPUT.pcs.tsv"
  $CHORD_LABELS $INPUT > $OUTPUT
done
