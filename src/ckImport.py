import csv
import glob

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.detectFace import get_largest_face, detect_faces


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def import_ck_plus_dataset(directory, dim, rgb):
    includeNeutral = True
    # Contempt and Disgust went into the category of Angry and Neutral is added
    categories = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    categoriesCK = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    dirImages = directory + '/Images'
    dirLabels = directory + '/Labels'

    oneofseven = 0
    imageFiles = glob.glob(dirImages + '/*/*/*.png')
    labelFiles = glob.glob(dirLabels + '/*/*/*.txt')

    allLabeledImages = []

    for label in labelFiles:
        img = label.replace(dirLabels, dirImages)
        img = img.replace('_emotion.txt', '.png')
        allLabeledImages.append(img)

    labeledImages = []
    labels = []
    labelNames = []
    cascade = cv2.CascadeClassifier('../data/lbpcascade_frontalface.xml')
    for ind in range(len(labelFiles)):
        curLabel = labelFiles[ind]
        # print(allLabeledImages[ind])
        if rgb:
            image = cv2.imread(allLabeledImages[ind])
        else:
            image = cv2.imread(allLabeledImages[ind], 0)
        temp = get_largest_face(image, detect_faces(cascade, image, is_gray=not rgb))
        curImage = cv2.resize(temp,
                              dim,
                              interpolation=cv2.INTER_CUBIC)
        # if not rgb:
        #     curImage = rgb2gray(curImage)
        with open(curLabel, 'r') as csvfile:
            rd = csv.reader(csvfile)
            for row in rd:
                str = row
            numCK = int(float(str[0]))

            labelText = categoriesCK[numCK - 1]
            if labelText != 'Contempt' and labelText != 'Disgust':
                numEitW = categories.index(labelText)
                labeledImages.append(curImage)
                labels.append(numEitW)
                labelNames.append(labelText)
            else:
                # if Contempt or Disgust feature is noticed, it is classified under Angry
                numEitW = categories.index('Angry')
                labeledImages.append(curImage)
                labels.append(numEitW)
                labelNames.append(labelText)
    if includeNeutral:
        # The first image in every series is neutral
        neutralPattern = '_00000001.png'
        neutralInd = categories.index('Neutral')
        neutralImages = []
        neutralLabels = []
        neutralLabelNames = []

        for imgStr in imageFiles:
            if neutralPattern in imgStr:
                oneofseven += 1
                if oneofseven % 7 == 0:

                    if rgb:
                        image = cv2.imread(imgStr)
                    else:
                        image = cv2.imread(imgStr, 0)

                    temp = get_largest_face(image, detect_faces(cascade, image, is_gray=not rgb))
                    temp = cv2.resize(temp,
                                      dim,
                                      interpolation=cv2.INTER_CUBIC)
                    # if not rgb:
                    #     curImage = rgb2gray(curImage)
                    neutralImages.append(temp)
                    neutralLabels.append(neutralInd)
                    neutralLabelNames.append('Neutral')

        images = labeledImages + neutralImages
        labels = labels + neutralLabels

    else:
        images = labeledImages

    # # For testing only:
    # images = images[0:10]
    # labels = labels[0:10]
    print(np.copy(images).shape)

    return images, labels


def import_dataset(directory):
    imgList, labels = import_ck_plus_dataset(directory, (224, 224), rgb=True)
    if len(imgList) <= 0:
        print('Error - No images found in ' + str(directory))
        return None

    # Return list of filenames
    return imgList, labels


def translate_labels(labels):
    # categories = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    mod = []
    angry = 0
    fear = 0
    happy = 0
    sad = 0
    surp = 0
    neutral = 0
    for l in labels:
        if l == 0:
            # Angry
            angry += 1
            mod = np.append(mod, ([1, 0, 0, 0, 0, 0]))
        elif l == 1:
            # Fear
            fear += 1
            mod = np.append(mod, ([0, 1, 0, 0, 0, 0]))
        elif l == 2:
            # Happy
            happy += 1
            mod = np.append(mod, ([0, 0, 1, 0, 0, 0]))
        elif l == 3:
            # Sad
            sad += 1
            mod = np.append(mod, ([0, 0, 0, 1, 0, 0]))
        elif l == 4:
            # Surprise
            surp += 1
            mod = np.append(mod, ([0, 0, 0, 0, 1, 0]))
        elif l == 5:
            # Neutral
            neutral += 1
            mod = np.append(mod, ([0, 0, 0, 0, 0, 1]))
    print("\nangry " + str(angry))
    print("\nfear " + str(fear))
    print("\nhappy " + str(happy))
    print("\nsad " + str(sad))
    print("\nsurp " + str(surp))
    print("\nneutral " + str(neutral))
    return np.split(mod, labels.shape[0])
    pass


def save_numpy_array(path):
    input_list, labels = import_dataset(path)
    image = np.copy(input_list)
    print(image.shape)
    images = image.reshape(411, image.shape[2], image.shape[1], 3)
    labels = np.copy(labels)

    labels = np.copy(translate_labels(labels))
    print(images.shape, labels.shape)
    np.save('../data/x_train1', images)
    np.save('../data/y_train1', labels)
