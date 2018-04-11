import numpy as np
import cv2
import platform
import tensorflow as tf
import keras
import math

print(platform.architecture())
print("numpy "+np.__version__)
print("opencv "+cv2.__version__)
print("Keras "+keras.__version__)
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
print("Keras "+keras.__version__)
print("cos "+str(math.cos(3)))
