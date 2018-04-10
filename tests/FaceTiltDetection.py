import cv2

from src.lbpDetectFace import detect_faces, draw_faces, rotate_img, rotate_point, draw_tilt_faces

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

    f = draw_tilt_faces(frame, lbp_face_cascade)

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("exit on ESC")
cv2.VideoCapture.release(vc)
