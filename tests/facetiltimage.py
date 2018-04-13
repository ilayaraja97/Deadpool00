import cv2
import time
import numpy as np
import sys
from keras.models import model_from_json

from src.lbpDetectFace import crop_rot_images
from src.testing import predict_emotion, put_emoji
from src.lbpDetectFace import detect_faces, draw_faces, get_largest_face




emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
lbp_face_cascade = cv2.CascadeClassifier('../data/lbpcascade_frontalface.xml')

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,0,0)
lineType               = 2


cv2.namedWindow("exit on ESC")
image1 = cv2.imread("../data/test5.jpg")

cv2.destroyWindow("exit on ESC")
temp = crop_rot_images(image1, lbp_face_cascade)
print(temp)
h, w, c   = temp.shape
#print(x, y, w)
#print(temp)
#temp = cv2.resize(temp, (120, 120))

cv2.imshow("exit on ESC", temp)
#cv2.destroyWindow("exit on ESC")

angry, fear, happy, sad, surprise, neutral = predict_emotion(temp)

emoji = put_emoji(angry, fear, happy, sad, surprise, neutral)
emoji= cv2.resize(emoji, (120, 120), interpolation = cv2.INTER_CUBIC)
 # Get boolean for transparency
# trans = emoji.copy()
# trans[emoji == 0] = 1
# trans[emoji != 0] = 0

    # Delete all pixels in image where emoji is nonzero
# temp[10:10+h,5:5+w,:] *= trans
#
#     # Add emoji on those pixels
# temp[10:10+h,5:5+w,:] += emoji


print(temp.shape)
overlay = cv2.resize(emoji, temp.shape[:2], interpolation=cv2.INTER_AREA)
cv2.addWeighted(overlay, 0.5, temp, 0.5, 0, temp)
cv2.putText(temp,"something",bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
cv2.imshow("output", temp)

with open('emotion.txt', 'a') as fp:
    fp.write('{},{},{},{},{},{},{}\n'.format(time.time(), angry, fear, happy, sad, surprise, neutral))

#cv2.imshow("exit on ESC", temp)
cv2.waitKey(0)
cv2.destroyWindow("exit on ESC")
