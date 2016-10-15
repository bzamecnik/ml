from flask import Flask, redirect, render_template, request
from gevent.wsgi import WSGIServer

from predict import InstrumentClassifier
from leaderboard import LeaderBoard

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 2**20

model_dir = 'data/working/single-notes-2000/models'
model_id = LeaderBoard(model_dir).best_model()
model = InstrumentClassifier(model_dir + '/' + model_id)

print('Using model:', model_id)

@app.route('/')
def hello():
    return render_template('home.html', model_id=model_id)

@app.route('/api/classify/instrument', methods=['POST'])
def classify():
    if 'audio_file' not in request.files:
        return redirect('/')

    # File-like object than can be directy passed to soundfile.read()
    # without saving to disk.
    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return redirect('/')

    class_probabilities = model.predict_probabilities(audio_file)
    class_probabilities = class_probabilities.round(5)
    label = model.class_label_from_probabilities(
        class_probabilities)

    return render_template('home.html',
        model_id=model_id,
        audio_file=audio_file.filename,
        predicted_label=label,
        class_probabilities=class_probabilities)

if __name__ == '__main__':
    # app.run(debug=True)
    app.debug = True

    # needed since Flask dev mode interacts badly with TensorFlow
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
