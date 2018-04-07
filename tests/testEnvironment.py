import numpy as np
import cv2
import platform
import theano
import keras

print(platform.architecture())
print("numpy "+np.__version__)
print("opencv "+cv2.__version__)
print("Theano "+theano.__version__)
print("Keras "+keras.__version__)
