import cv2
from keras.models import model_from_json


emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# load json and create model arch
json_file = open('../data/model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('../data/model.h5')




def predict_emotion(face_image): # a single cropped face
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, (48, 48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]

def put_emoji(angry, fear, happy, sad, surprise, neutral,frame):
    emotion = max(angry, fear, happy, sad, surprise, neutral)
    if emotion == angry:
        status="You are angry"
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
    elif emotion == neutral:
        status = "You are neutral"
        emoji = cv2.imread("../data/neutral.png")
        print(" You are neutral")
    #emoji = cv2.resize(emoji, (120, 120), interpolation=cv2.INTER_CUBIC)
    overlay = cv2.resize(emoji, (48, 48), interpolation=cv2.INTER_AREA)


    return overlay, status

