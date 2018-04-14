import csv
import glob

import cv2
import numpy as np

from src.detectFace import get_largest_face, detect_faces


def importCKPlusDataset(dir='CK+', includeNeutral=False):
    # Contempt and Disgust went into the category of Angry and Neutral is added
    categories = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    categoriesCK = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    dirImages = dir + '/Images'
    dirLabels = dir + '/Labels'

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
        image = cv2.imread(allLabeledImages[ind])
        curImage = cv2.resize(get_largest_face(image, detect_faces(cascade, image)),
                              (224, 224),
                              interpolation=cv2.INTER_CUBIC)
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
                neutralImages.append(imgStr)
                neutralLabels.append(neutralInd)
                neutralLabelNames.append('Neutral')

        images = labeledImages + neutralImages
        labels = labels + neutralLabels

    else:
        images = labeledImages

    # # For testing only:
    # images = images[0:10]
    # labels = labels[0:10]
    return images, labels


def importDataset(dir):
    imgList, labels = importCKPlusDataset(dir, includeNeutral=True)
    if len(imgList) <= 0:
        print('Error - No images found in ' + str(dir))
        return None

    # Return list of filenames
    return imgList, labels


def translate_labels(labels):
    # categories = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    mod = []
    for l in labels:
        if l == 0:
            # Angry
            mod = np.append(mod, ([1, 0, 0, 0, 0, 0]))
        elif l == 1:
            # Fear
            mod = np.append(mod, ([0, 1, 0, 0, 0, 0]))
        elif l == 2:
            # Happy
            mod = np.append(mod, ([0, 0, 1, 0, 0, 0]))
        elif l == 3:
            # Sad
            mod = np.append(mod, ([0, 0, 0, 1, 0, 0]))
        elif l == 4:
            # Surprise
            mod = np.append(mod, ([0, 0, 0, 0, 1, 0]))
        elif l == 5:
            # Neutral
            mod = np.append(mod, ([0, 0, 0, 0, 0, 1]))
    return np.split(mod, labels.shape[0])
    pass


def saveNumpyArray(path):
    input_list, labels = importDataset(path)
    image = np.copy(input_list)
    labels = np.copy(labels)
    labels = np.copy(translate_labels(labels))
    np.save('../data/x_train', image)
    np.save('../data/y_train', labels)
