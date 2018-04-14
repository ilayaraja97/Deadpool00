import csv
import glob

import cv2
import numpy as np

from src.lbpDetectFace import get_largest_face, detect_faces


def importCKPlusDataset(dir='CK+', includeNeutral=False):
    # Note: "Neutral" is not labeled in the CK+ dataset
    categories = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    categoriesCK = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    # Root directories for images and labels. Should have no other .txt or .png files present
    dirImages = dir + '/Images'
    dirLabels = dir + '/Labels'

    # Get all possible label and image filenames
    imageFiles = glob.glob(dirImages + '/*/*/*.png')
    labelFiles = glob.glob(dirLabels + '/*/*/*.txt')

    # Get list of all labeled images:
    # Convert label filenames to image filenames
    # Label looks like: CK_Plus/CKPlus_Labels/S005/001/S005_001_00000011_emotion.txt
    # Image looks like: CK_Plus/CKPlus_Images/S005/001/S005_001_00000011.png
    allLabeledImages = []

    for label in labelFiles:
        img = label.replace(dirLabels, dirImages)
        img = img.replace('_emotion.txt', '.png')
        allLabeledImages.append(img)
    # Construct final set of labeled image file names and corresponding labels
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
        # Open the image as binary read-only
        with open(curLabel, 'r') as csvfile:

            # Convert filestream to csv-reading filestream
            rd = csv.reader(csvfile)
            for row in rd:
                str = row
                # print(int(float(str[0])))
            # Get integer label in CK+ format
            numCK = int(float(str[0]))

            # Get text label from CK+ number
            labelText = categoriesCK[numCK - 1]
            if labelText != 'Contempt' and labelText != 'Disgust':
                numEitW = categories.index(labelText)
                labeledImages.append(curImage)
                labels.append(numEitW)
                labelNames.append(labelText)
            else:
                # Lump "Contempt" in with another category
                numEitW = categories.index('Angry')
                labeledImages.append(curImage)
                labels.append(numEitW)
                labelNames.append(labelText)
    if includeNeutral:
        # Add all neutral images to our list too:
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

        # Combine lists of labeled and neutral images
        images = labeledImages + neutralImages
        labels = labels + neutralLabels
        labelNames = labelNames + neutralLabelNames

    else:
        images = labeledImages

    # # For testing only:
    images = images[0:10]
    labels = labels[0:10]
    print("HEY RAJA")
    # print(images)
    print("HEY HIMANI", np.copy(images))
    print(labels)
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
            mod = np.append(mod, ([1, 0, 0, 0, 0, 0]))
        elif l == 1:
            mod = np.append(mod, ([0, 1, 0, 0, 0, 0]))
        elif l == 2:
            mod = np.append(mod, ([0, 0, 1, 0, 0, 0]))
        elif l == 3:
            mod = np.append(mod, ([0, 0, 0, 1, 0, 0]))
        elif l == 4:
            mod = np.append(mod, ([0, 0, 0, 0, 1, 0]))
        elif l == 5:
            mod = np.append(mod, ([0, 0, 0, 0, 0, 1]))
    return np.split(mod, labels.shape[0])
    pass


def getNumpyArrayForTraining():
    input_list, labels = importDataset('../data/ck/CK')
    image = np.copy(input_list)
    labels = np.copy(labels)
    labels = np.copy(translate_labels(labels))
    print(image.shape, labels.shape)
    np.save('../data/x_train', image)
    np.save('../data/y_train', labels)