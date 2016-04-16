import matplotlib.pyplot as plt
from matplotlib import animation

audio_file = '/Users/bzamecnik/Documents/harmoneye-labs/harmoneye/data/wav/c-scale-piano-mono.wav'

fig = plt.figure(figsize=(5,4))
# 350 frames = 25 fps * 14 sec
ch_resampled = resample(ch_from_file[:,:108], 350, axis=0)
vmin, vmax = ch_resampled.min(), ch_resampled.max(),
def animate(nframe):
    cla()
    imshow(ch_resampled[nframe].reshape(-1,12),
         cmap=plt.get_cmap('jet'), interpolation='nearest',
               vmin=vmin, vmax=vmax)

anim = animation.FuncAnimation(fig, animate, frames=ch_resampled.shape[0])
anim_file = 'c-scale-piano-chroma-grid.gif'
anim.save(anim_file, writer='imagemagick', fps=25);

#%%sh
#ffmpeg -i anim_file -i audio_file -c:v libx264 -c:a libmp3lame -b:a 128k c-scale-piano-chroma-grid.mp4
