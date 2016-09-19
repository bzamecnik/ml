# Synthesizes an audio file from MIDI using some basic sound font.
#
# It's a wrapper over fluidsynth which provides a bit easier interface.
#
# Input: MIDI file
# Output: audio file (WAV, FLAC, ...)
SOUND_FONT="$HOME/Library/Audio/Sounds/Banks/generaluser_gs_v1.47.sf2"
fluidsynth -ni ${SOUND_FONT} $1 -F $2
