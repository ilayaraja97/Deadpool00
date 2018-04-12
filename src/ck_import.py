import csv
import glob

import cv2

from src.lbpDetectFace import get_largest_face, detect_faces


def importCKPlusDataset(dir = 'CK+', includeNeutral = False):
    ############################################################################
    # Function: importCKPlusDataset
    # Depending on preferences, this ranges from 309 - 920 images and labels
    #    - 309 labeled images
    #    - 18 more "Contempt" images (not in our vocabulary)
    #    - 593 neutral images
    #
    # For this to work, make sure your CKPlus dataset is formatted like this:
    # CKPlus = root (or whatever is in your 'dir' variable)
    # CKPlus/CKPlus_Images = Root for all image files (no other file types here)
    #    Example image path:
    #    CKPlus/CKPlus_Images/S005/001/S005_001_00000011.png
    #
    # CKPlus/CKPlus_Labels = Root for all image labels (no other file types)
    #    Example label path:
    #    CKPlus/CKPlus_Labels/S005/001/S005_001_00000011_emotion.png
    #
    # CKPlus/* - anything else in this directory is ignored, as long as it
    # is not in the _Images or _Labels subdirectories
    #
    # Optional inputs:
    # dir - Custom root directory for CKPlus dataset (if not 'CKPlus')
    #
    # includeNeutral - Boolean to include neutral pictures or not
    #    Note: Every sequence begins with neutral photos, so neutral photos
    #    greatly outnumber all other combined (approximately 593 to 327)
    #
    # contemptAs - Since it's not in our vocabulary, by default all pictures
    # labeled "Contempt" are discarded. But if you put a string here, e.g.
    # "Disgust", pictures labeled "Contempt" will be lumped in with "Disgust"
    # instead of being discarded.
    #
    #
    # RETURN VALUES:
    # images, labels = List of image file paths, list of numeric labels
    # according to EitW numbers
    ############################################################################

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
    # print(allLabeledImages)
    # Construct final set of labeled image file names and corresponding labels
    labeledImages = []
    labels = []
    labelNames = []
    contemptImages = []
    cascade = cv2.CascadeClassifier('../data/lbpcascade_frontalface.xml')
    # print(labelFiles)
    for ind in range(len(labelFiles)):
        curLabel = labelFiles[ind]
        # print(allLabeledImages[ind])
        image = cv2.imread(allLabeledImages[ind])
        curImage = get_largest_face(image, detect_faces(cascade, image))
        # print(curImage)
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
            labelText = categoriesCK[numCK-1]
            # print(labelText)
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

    # For testing only:
    images = images[0:10]
    labels = labels[0:10]
    print(images)
    print("HEY HIMANI")
    print(labels)
    return images, labels


def importDataset(dir):
    imgList, labels = importCKPlusDataset(dir, includeNeutral=True)
    if len(imgList) <= 0:
        print('Error - No images found in ' + str(dir))
        return None

    # Return list of filenames
    return imgList, labels


input_list, labels = importDataset('../data/ck/CK')

