import cv2
#import sys
#sys.path.insert(0, "/home/chandu/Desktop/Deadpool00/src")
#print(sys.path)
#from src import lbpDetectFace
from src.lbpDetectFace import detect_faces, draw_faces, draw_largest_face

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
# to write image as a new png file
#cv2.imwrite("cropped_test_image.png", f)
    rval, frame = vc.read()
    f = draw_largest_face(frame, detect_faces(lbp_face_cascade, frame))
    key = cv2.waitKey(50)
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("exit on ESC")
cv2.VideoCapture.release(vc)
