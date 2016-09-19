# Install FluidSynth MIDI synthesizer which is available for non-realtime batch
# synthesis and some sound fonts.
# This script works on OSX with homebrew.

if hash fluidsynth 2>/dev/null; then
  echo "Installing FluidSynth:"
  brew install fluid-synth --with-libsndfile
else
  echo "FluidSynth up-to-date"
fi

echo "Installing sound fonts:"

GS_TARGET_FILE="$HOME/Library/Audio/Sounds/Banks/generaluser_gs_v1.47.sf2"
if [ ! -f ${GS_TARGET_FILE} ]; then
  echo "Installing GeneralUser GS ..."
  if [ ! -f /tmp/GeneralUser_GS_1.47.zip ]; then
    wget https://dl.dropboxusercontent.com/u/8126161/GeneralUser_GS_1.47.zip -O /tmp/GeneralUser_GS_1.47.zip
  fi
  unzip -q /tmp/GeneralUser_GS_1.47.zip -d /tmp/
  mv '/tmp/GeneralUser GS 1.47/GeneralUser GS v1.47.sf2' ${GS_TARGET_FILE}
  rm -r '/tmp/GeneralUser GS 1.47'
else
  echo "GeneralUser GS is up-to-date."
fi

echo "Installed sound fonts:"
ls -lh ~/Library/Audio/Sounds/Banks/
