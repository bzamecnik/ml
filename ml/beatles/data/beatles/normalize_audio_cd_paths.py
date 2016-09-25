# Change the directory structure and file names of audio CD files so that they
# correspond to the naming in the isophonics annotation dataset.

import glob
import shutil

mkdir -p audio-cd/The_Beatles
echo mv audio-cd-raw/*/*/* audio-cd/The_Beatles

with open('album-mapping.csv', 'r') as file:
    album_mapping = [line.strip().split('\t') for line in file.readlines()]

base_dir = 'audio-cd/The_Beatles/'
for src, dest in album_mapping:
    shutil.move(base_dir + src, base_dir + dest)

with open('song-mapping.csv', 'r') as file:
    song_mapping = [line.strip().split('\t') for line in file.readlines()]
for src, dest in song_mapping:
    shutil.move(base_dir + src + '.wav', base_dir + dest + '.wav')
