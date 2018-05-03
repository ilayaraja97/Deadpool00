import cv2
import numpy as np

from src.training import cnn_arch, deep_cnn_arch, deep2_cnn_arch

x_fname = '../data/x_train1.npy'
y_fname = '../data/y_train1.npy'
x_train = np.load(x_fname)
y_train = np.load(y_fname)
print('Loading data...')
print(x_train.shape, y_train.shape)
model1 = cnn_arch(x_train, y_train, batch_size=10, validation_split=0.2, epochs=40)
x_fname = '../data/x_train2.npy'
y_fname = '../da6ta/y_train2.npy'
x_train = np.load(x_fname)
y_train = np.load(y_fname)

# print(y_train[1])
print('Loading data...')
print(x_train.shape, y_train.shape)
# cv2.imshow('Image', x_train[1])
# model2 = deep_cnn_arch(x_train, y_train, batch_size=10, validation_split=0.2, epochs=300)
# model3 = deep2_cnn_arch(x_train, y_train, batch_size=10, validation_split=0.2, epochs=200)
