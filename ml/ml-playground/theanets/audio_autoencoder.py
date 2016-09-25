import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
import theanets
import climate

climate.enable_default_logging()

X_orig = np.load('/Users/bzamecnik/Documents/music-processing/music-processing-experiments/c-scale-piano_spectrogram_2048_hamming.npy')
sample_count, feature_count = X_orig.shape
X = MinMaxScaler().fit_transform(X_orig)
X = X.astype(np.float32)

X_train, X_test = train_test_split(X, test_size=0.4, random_state=42)
X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=42)

# (np.maximum(0, 44100/512*np.arange(13)-2)).astype('int')
#blocks = [0, 84, 170, 256, 342, 428, 514, 600, 687, 773, 859, 945, 1031, 1205]
blocks = [0, 48, 98, 148, 198, 248, 298, 348, 398, 448, 498, 548, 598, 700]

def make_labels(blocks):
    label_count = len(blocks) - 1
    labels = np.zeros(blocks[-1])
    for i in range(label_count):
        labels[blocks[i]:blocks[i+1]] = i
    return labels

y = make_labels(blocks)

def score(exp, Xs):
    X_train, X_val, X_test = Xs
    def sc(exp, X):
        return r2_score(X, exp.network.predict(X))
    print("training:  ", sc(exp, X_train))
    # NOTE: only optimize to validation dataset's score!
    print("validation:", sc(exp, X_val))
    print("test:      ", sc(exp, X_test))

exp1 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    hidden_l1=0.1)

exp1.train(X_train, X_val, optimize='nag', learning_rate=1e-3, momentum=0.9)

exp2 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    hidden_l1=0.1)

exp2.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

# gives quite nice prediction, trains slow
exp3 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    hidden_l1=0.1, hidden_activation='relu')

exp3.train(X_train, X_val, optimize='nag', learning_rate=1e-3, momentum=0.9)

exp4 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    hidden_l1=0.1, input_dropout=0.3)

exp4.train(X_train, X_val, optimize='nag', learning_rate=1e-3, momentum=0.9)

# rmsprop - converges faster in this case than nag
exp5 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    hidden_l1=0.1)

exp5.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

# tied weighs - work good, much lower loss function values
# r2: 0.75037549551862703
exp6 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    hidden_l1=0.1, tied_weights=True)

exp6.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

# higher hidden L1 penalty - worse
exp7 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    hidden_l1=0.7, tied_weights=True)

exp7.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

# hidden L2 penalty - a bit worse
exp8 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    hidden_l1=0.1, hidden_l2=0.1, tied_weights=True)

exp8.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

# no regularization - in this case better
# r2: 0.82211329411744094
exp10 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    tied_weights=True)

exp10.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

# layerwise autoencoder training

exp11 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 500, feature_count), tied_weights=True)
exp11.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

# wow - this actually is able to to a 2D visualization
exp12 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 100, 10, 2, 10, 100, feature_count),
    tied_weights=True)
exp12.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

def compute_middle_layer(X, model):
    X_pred_ff = model.feed_forward(X)
    middle = int(len(X_pred_ff)/2)
    X_middle = X_pred_ff[middle]
    return X_middle

def visualize_2d(X, y=None):
    colors = y/max(y) if y is not None else np.linspace(0,1,len(X))
    scatter(X[:,0], X[:,1],
        c=colors, alpha=0.2, edgecolors='none', cmap='rainbow')

# same visualization, a little bit better r2
exp13 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 256, 64, 16, 2, 16, 64, 256, feature_count),
    tied_weights=True)
exp13.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

# contractive - better than without
# r2: 0.82820148664941162
exp14 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    tied_weights=True, contractive=0.8)
exp14.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

# tanh - bad
exp15 = theanets.Experiment( theanets.Autoencoder,
    layers=(feature_count, 500, feature_count),
    tied_weights=True, hidden_activation='tanh')
exp15.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

# relu, contractive
exp16 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 128, 16, 2, 16, 128, feature_count),
    tied_weights=True, hidden_activation='relu', contractive=0.5)
exp16.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

exp17 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 128, 16, 2, 16, 128, feature_count),
    tied_weights=True, contractive=0.8)
exp17.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

exp18 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, input_dropout=0.8)
exp18.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

# r2: 0.83371355062803953
exp19 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, input_dropout=0.8, hidden_dropout=0.8)
exp19.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

exp20 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, input_dropout=0.9, hidden_dropout=0.9)
exp20.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

# -----------------

# animate the 2D point movement

import matplotlib.animation as animation

def export_animation(X_2d, y, filename):
    fig = plt.figure()
    # 854x480 px (480p) in inches, note that 8.54 gives 853px width :/
    fig.set_size_inches(8.545, 4.80)
    plt.axis('equal')
    # plt.tight_layout()
    # plt.xlim(-0.1, 1.1)
    # plt.ylim(-0.1, 1.1)
    images = []
    im1 = scatter(X_2d[:, 0], X_2d[:, 1], c=y/max(y), cmap='rainbow', alpha=0.2)
    for i in range(len(X_2d)):
        im2 = scatter(X_2d[i, 0], X_2d[i, 1], c=y[i]/max(y), cmap='rainbow')
        images.append([im1, im2])

    ani = animation.ArtistAnimation(fig, images,
        interval=20, blit=False, repeat=False)
    writer = animation.writers['ffmpeg'](fps=50, bitrate=5000)
    ani.save(filename, writer=writer, dpi=100)

export_animation(X_tsne, y, 'piano-tsne.mp4')

#----------------------


exp21 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, input_dropout=0.3, hidden_dropout=0.5,
    batch_size=len(X_train))
exp21.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

exp22 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, input_dropout=0.3, hidden_dropout=0.5)
exp22.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

exp23 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, 256, 128, 64, 32, 16, 8, 4, 2,
        4, 8, 16, 32, 64, 128, 256, 512, feature_count),
    tied_weights=True, input_dropout=0.3, hidden_dropout=0.5)
exp23.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

exp24 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, input_dropout=0.3, hidden_dropout=0.5,
    hidden_activation='linear')
exp24.train(X_train, X_val, optimize='rmsprop', learning_rate=1e-3, momentum=0.9)

# r2: 0.833454635805
exp25 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True)
exp25.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.9)

# r2: 0.731835366439
exp26 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True)
exp26.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.1)

# r2: 0.854741515141 (*)
exp27 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True)
exp27.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

# r2: 0.84260338122
exp28 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True)
exp28.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.7)

exp29 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True)
exp29.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp30 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, input_dropout=0.9)
exp30.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp31 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 100, feature_count),
    tied_weights=True)
exp31.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp32 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 200, 20, 2, 20, 200, feature_count),
    tied_weights=True, input_dropout=0.5, hidden_dropout=0.5)
exp32.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

# bad - makes a single curve
exp33 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 200, 20, 2, 20, 200, feature_count),
    tied_weights=True, hidden_l1=0.1)
exp33.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

# bad - makes a non-discriminative curve
exp34 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 200, 20, 2, 20, 200, feature_count),
    tied_weights=True, input_dropout=0.5)
exp34.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp35 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 200, 20, 2, 20, 200, feature_count),
    tied_weights=True, hidden_dropout=0.5)
exp35.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp36 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 200, 20, 2, 20, 200, feature_count),
    tied_weights=True)
exp36.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)


exp33 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, 256, 128, 64, 32, 16, 8, 4, 2,
        4, 8, 16, 32, 64, 128, 256, 512, feature_count),
    tied_weights=True)
exp33.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

X_zca_train, X_zca_test = train_test_split(X_zca, test_size=0.4, random_state=42)
X_zca_val, X_zca_test = train_test_split(X_zca_test, test_size=0.5, random_state=42)


exp34 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True)
exp34.train(X_zca_train, X_zca_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp35 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, 256, 128, 64, 32, 16, 8, 4, 2,
        4, 8, 16, 32, 64, 128, 256, 512, feature_count),
    tied_weights=True, hidden_activation='relu')
exp35.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

# - try tanh and relu for deeper networks
# - try other normalization (mean-std instead od min-max)

X_ms = StandardScaler().fit_transform(X_orig).astype(np.float32)
X_ms_train, X_ms_test = train_test_split(X_ms, test_size=0.4, random_state=42)
X_ms_val, X_ms_test = train_test_split(X_ms_test, test_size=0.5, random_state=42)

exp36 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True)
exp36.train(X_ms_train, X_ms_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp37 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, hidden_activation='tanh')
exp37.train(X_ms_train, X_ms_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp38 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True)
exp38.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

X_orig_train, X_orig_test = train_test_split(X_orig.astype('float32'), test_size=0.4, random_state=42)
X_orig_val, X_orig_test = train_test_split(X_orig_test, test_size=0.5, random_state=42)

exp39 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True)
exp39.train(X_orig_train, X_orig_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp40 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, hidden_activation='linear', hidden_l1=0.5)
exp40.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp41 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, hidden_activation='relu', hidden_l1=0.5)
exp41.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp42 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, hidden_activation='relu', weight_l1=0.5)
exp42.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

# bad
exp43 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, hidden_activation='relu', contractive=0.9)
exp43.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

# not bad
exp44 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, hidden_activation='relu')
exp45.train(X_ms_train, X_ms_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp45 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, hidden_activation='relu', contractive=0.5)
exp45.train(X_ms_train, X_ms_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

# r2:  0.849283267068
exp46 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, hidden_activation='linear', contractive=0.5)
exp46.train(X_ms_train, X_ms_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)

exp47 = theanets.Experiment(theanets.Autoencoder,
    layers=(feature_count, 512, feature_count),
    tied_weights=True, hidden_activation='linear', contractive=0.5)
exp47.train(X_train, X_val, optimize='layerwise', learning_rate=1e-3, momentum=0.5)
