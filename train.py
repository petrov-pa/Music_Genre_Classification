import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import librosa
import os
import pickle
from sklearn.preprocessing import StandardScaler
from preprocessing import get_features
from model import get_model

path = './data'
genres = os.listdir(path)

# Пройдем по всем файлам и извлечем признаки
x_data = []
y = []
for i, genre in enumerate(genres):
    for audio in os.listdir('./data/{}'.format(genre)):
        x, sr = librosa.load(os.path.join(path, '{}/{}'.format(genre, audio)), mono=True, duration=30)
        x_data.append(get_features(x, sr))
        y.append(i)
# нормируем данные
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
with open('./models/Scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)
# callback для изменения lr
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=0,
                                                   mode="auto", min_delta=0.0001, cooldown=0, min_lr=0)
# callback для сохранения лучшей модели
save_callback = tf.keras.callbacks.ModelCheckpoint('./models/best_model.hdf5', monitor='loss', verbose=1,
                                                   save_best_only=True, mode='auto')
callbacks = [lr_callback, save_callback]

x_data = np.array(x_data[:, :-10])
y = np.array(y)

# Обучаем модель
model = get_model(x_data.shape[1])
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_data, y, epochs=150, batch_size=16, callbacks=callbacks)
