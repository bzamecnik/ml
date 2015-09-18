# Extract binary pitch class sets + root and bass pitch classes from chord labels.

# https://github.com/bzamecnik/chord-labels

export CHORD_LABELS_DIR=~/Documents/dev/repos/chord-labels
for file in $(find $1 -name '*.lab'); do
  echo $file
  INPUT=$(greadlink -f $file)
  OUTPUT="$INPUT.pcs.tsv"
  $CHORD_LABELS_DIR/gradlew -q -p $CHORD_LABELS_DIR run -Pfile=$INPUT > $OUTPUT
done
