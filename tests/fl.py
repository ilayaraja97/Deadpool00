# import the OpenCV package
import cv2
import numpy as np

# load the image with imread()
imageSource = 'we.jpg'
img = cv2.imread(imageSource)

lbp_face_cascade = cv2.CascadeClassifier('../data/lbpcascade_frontalface.xml')

def detect_faces(f_cascade, colored_img, scale_factor=1.1):
    img_copy = np.copy(colored_img)
    # convert to gray
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    faces = f_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=6)

    return faces



def draw_faces(img, faces):
    # draw rectangles
    largest_face = 0
    length = 1
    a = b = c = d = 0
    for (x, y, w, h) in faces:
        if (w*h) >= largest_face:
            largest_face = w*h
            a = x
            b = y
            c = w
            d = h
    print(x, y, w, h)
#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in faces:
        if (x == a)and(y == b)and(w == c)and(h == d):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img


f = draw_faces(img, detect_faces(lbp_face_cascade, img))
cv2.imshow("exit on ESC", cv2.flip(f, 1))
cv2.waitKey(0)

# close the windows
cv2.destroyAllWindows()

