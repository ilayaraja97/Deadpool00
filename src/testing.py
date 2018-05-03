import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib
import matplotlib.pyplot as plt

shape = (48, 48)
shapec = (1, 48, 48, 1)

matplotlib.use("TkAgg")

emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# load json and create model arch
json_file = open('../data/modeldeep2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('../data/modeldeep2.h5')

# figure
# fig = plt.figure()


def predict_emotion(face_image):  # a single cropped face
    gray = face_image
    if shapec[3] == 1:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, shape, interpolation=cv2.INTER_AREA)
    image = resized_img.reshape(shapec)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]


def put_emoji(angry, fear, happy, sad, surprise, neutral):
    emotion = max(angry, fear, happy, sad, surprise, neutral)
    if emotion == angry:
        status = "You are angry"
        emoji = cv2.imread("../data/angry.png")
        print(" You are angry")
    elif emotion == fear:
        status = "You are fear"
        emoji = cv2.imread("../data/fearful.png")
        print(" You are fear")
    elif emotion == happy:
        status = "You are happy"
        emoji = cv2.imread("../data/happy.png")
        print(" You are happy")
    elif emotion == sad:
        status = "You are sad"
        emoji = cv2.imread("../data/sad.png")
        print(" You are sad")
    elif emotion == surprise:
        status = "You are surprise"
        emoji = cv2.imread("../data/surprised.png")
        print(" You are surprise")
    else:
        # emotion == neutral:
        status = "You are neutral"
        emoji = cv2.imread("../data/neutral.png")
        print(" You are neutral")
    # emoji = cv2.resize(emoji, (120, 120), interpolation=cv2.INTER_CUBIC)
    overlay = cv2.resize(emoji, (80, 80), interpolation=cv2.INTER_AREA)
    return overlay, status


def plot_emotion_matrix(angry, fear, happy, sad, surprise, neutral, frame):
    data = {'angry': angry, 'fear': fear, 'happy': happy, 'sad': sad, 'surprise': surprise, 'neutral': neutral}
    fig, a = plt.subplots()
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    a.bar(data.keys(), data.values(), 0.5, color='SkyBlue', alpha=1.0, )
    a.yaxis.set_visible(False)
    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # display image with opencv or any operation you like

    img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA)
    return img

    # cv2.imshow("plot", img)
    # plt.show()
