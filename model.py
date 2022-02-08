from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU


def get_model(input_shape):
    model = Sequential()
    model.add(Dense(1000, input_dim=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    return model
