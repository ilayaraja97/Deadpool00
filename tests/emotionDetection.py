import time
import matplotlib.pyplot as plt
import cv2

from src.detectFace import detect_faces, get_largest_face, crop_rot_images
from src.testing import predict_emotion, put_emoji, plot_emotion_matrix

# set emotions (6)
emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# get lbp classifier
lbp_face_cascade = cv2.CascadeClassifier('../data/lbpcascade_frontalface.xml')
# set font
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100, 30)
fontScale = 1
fontColor = (0, 0, 0)
lineType = 2

cv2.namedWindow("exit on ESC")
vc = cv2.VideoCapture(0)
frame = 0
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
f = cv2.flip(frame, 1)
while rval:
    cv2.imshow("exit on ESC", f)
    rval, frame = vc.read()

    # tilt optimization req
    # temp = crop_rot_images(frame, lbp_face_cascade,draw_face=True)
    temp = get_largest_face(frame, detect_faces(lbp_face_cascade, frame), draw_face=True)

    if temp.shape != (0, 0, 3):
        angry, fear, happy, sad, surprise, neutral = predict_emotion(temp)
        plot_emotion_matrix(angry, fear, happy, sad, surprise, neutral)
        overlay, status = put_emoji(angry, fear, happy, sad, surprise, neutral)
        frame = cv2.flip(frame, 1)
        cv2.addWeighted(overlay, 0.9, frame[0:80, 0:80], 0.1, 0, frame[0:80, 0:80])
        cv2.putText(frame, status, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        # with open('emotion.txt', 'a') as fp:
        #     fp.write('{},{},{},{},{},{},{}\n'.format(time.time(), angry, fear, happy, sad, surprise, neutral))

        f = frame
    else:
        f = cv2.flip(frame, 1)

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("exit on ESC")
cv2.VideoCapture.release(vc)
