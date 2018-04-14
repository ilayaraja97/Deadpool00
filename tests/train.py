import numpy as np

from src.training import cnn_arch

x_fname = '../data/x_train.npy'
y_fname = '../data/y_train.npy'
x_train = np.load(x_fname)
y_train = np.load(y_fname)
print('Loading data...')
print(x_train.shape, y_train.shape)
model = cnn_arch(x_train, y_train, batch_size=10, validation_split=0.2, epochs=3)
