import numpy as np

from src.training import cnn_arch

x_fname = '../data/X_train.npy'
y_fname = '../data/y_train.npy'
x_train = np.load(x_fname)
y_train = np.load(y_fname)
print('Loading data...')

model = cnn_arch(x_train, y_train, batch_size=256, validation_split=0.2, epochs=3)

