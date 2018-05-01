import numpy as np

from src.training import cnn_arch, deep_cnn_arch

x_fname = '../data/x_train.npy'
y_fname = '../data/y_train.npy'
x_train = np.load(x_fname)
y_train = np.load(y_fname)
print('Loading data...')
print(x_train.shape, y_train.shape)
# model1 = cnn_arch(x_train, y_train, batch_size=10, validation_split=0.2, epochs=20)
x_fname = '../data/x_train2.npy'
y_fname = '../data/y_train2.npy'
x_train = np.load(x_fname)
y_train = np.load(y_fname)
print('Loading data...')
print(x_train.shape, y_train.shape)
model2 = deep_cnn_arch(x_train, y_train, batch_size=10, validation_split=0.2, epochs=20)
