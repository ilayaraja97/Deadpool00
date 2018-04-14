import cv2

from src.detectFace import detect_faces, draw_faces

lbp_face_cascade = cv2.CascadeClassifier('../data/lbpcascade_frontalface.xml')

cv2.namedWindow("exit on ESC")
vc = cv2.VideoCapture(0)
frame = 0
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

f = frame

while rval:
    cv2.imshow("exit on ESC", cv2.flip(f, 1))
    rval, frame = vc.read()

    f = draw_faces(frame, detect_faces(lbp_face_cascade, frame))

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("exit on ESC")
cv2.VideoCapture.release(vc)
