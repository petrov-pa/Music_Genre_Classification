import os
import re
import numpy as np
import librosa
import pickle
from preprocessing import get_features
from model import get_model


def run():
    files = [file for file in os.listdir('./music') if not file.startswith('.')]
    if files is False:
        return 'Отсутствуют файлы в папке music'
    genres = {0: 'country', 1: 'hiphop', 2: 'classical', 3: 'metal', 4: 'jazz',
              5: 'blues', 6: 'pop', 7: 'rock', 8: 'reggae', 9: 'disco'}
    x_data = []
    for audio in files:
        if re.search('[^.]*$', audio)[0] not in ['au', 'wav', 'mp3']:
            return 'Неизвестный формат файла'
        y, sr = librosa.load(os.path.join('./music/{}'.format(audio)), mono=True, duration=30)
        x_data.append(get_features(y, sr))
    with open('./models/Scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)
    x_data = scaler.transform(x_data)
    x_data = np.array(x_data[:, :-10])
    model = get_model(x_data.shape[1])
    model.load_weights('./models/best_model.hdf5')
    predict = model.predict(x_data)
    predict = [genres[ind] for ind in np.argmax(predict, axis=1)]
    with open('./result.txt', 'w') as res:
        for line in zip(files, predict):
            res.write('{} - {} \n'.format(line[0], line[1]))
    pass


run()
