import os
import numpy as np
import librosa
import pickle
from flask import Flask, abort, render_template
from preprocessing import get_features
from model import get_model
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")


def run(filename):
    genres = {0: 'country', 1: 'hiphop', 2: 'classical', 3: 'metal', 4: 'jazz',
              5: 'blues', 6: 'pop', 7: 'rock', 8: 'reggae', 9: 'disco'}
    y, sr = librosa.load('./music/'+filename, mono=True, duration=30)
    x_data = []
    x_data.append(get_features(y, sr))
    with open('./models/Scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)
    x_data = scaler.transform(x_data)
    x_data = np.array(x_data[:, :-10])
    model = get_model(x_data.shape[1])
    model.load_weights('./models/best_model.hdf5')
    result = model.predict(x_data)
    result = genres[np.argmax(result, axis=1)[0]]
    return result


app = Flask(__name__)

app.config.update(dict(
                SECRET_KEY="sa2sd7gg4sa7a5as4d54fa78",
                WTF_CSRF_SECRET_KEY="k55h2l8o2n1n5g0"
            ))


@app.route('/badrequest400')
def bad_request():
    return abort(400)


class MyForm(FlaskForm):
    file = FileField('Ваша композиция: ',
                     validators=[FileRequired(), FileAllowed(['mp3', 'au', 'wav'], 'Загрузите аудио-файл')])


@app.route('/', methods=('GET', 'POST'))
def predict():
    form = MyForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join('./music', filename))
        output = run(filename)
        return render_template('submit.html', form=form, name=output)
    return render_template('submit.html', form=form, name=' ')


app.run()
