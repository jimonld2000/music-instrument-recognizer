from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os
import pickle
import random
import operator
import math

# Function which determines 'distance' between two wav files MEL spectograms
def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1)
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


# Function which determines the k nearest neighbours
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(
            instance, trainingSet[x], k
        )
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# Function which determines the nearest neighbour class of an instance
def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1 

    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


# Function which determines the trained model's accuracy
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return 1.0 * correct / len(testSet)


directory = "D:\\Materii\\EA-3\\SEM1\\up\\proj\\music instrument recognition\\InstrumentSamples\\"
f = open("model.dat", "wb")
i = 0

# Looping the dataset
for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    # Extracting each soundwile and getting the MEL spectogram of it
    for file in os.listdir(directory + folder):
        (rate, sig) = wav.read(directory + folder + "\\" + file)
        mfcc_feat = mfcc(
            sig, rate, winlen=0.020, numcep=1000, nfft=1024, appendEnergy=False
        )
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        # Dumping the serialized data
        pickle.dump(feature, f)

f.close()
dataset = []

# Loading the data
def loadDataset(filename, split, trSet, teSet):
    with open("model.dat", "rb") as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])


trainingSet = []
testSet = []
# Loading the data with 0.80 of it being the trainingset, the rest the testSet
loadDataset("model.dat", 0.80, trainingSet, testSet)

leng = len(testSet)
predictions = []
# Getting the predictions
for x in range(leng):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))

# Determining and printing accuracy
accuracy1 = getAccuracy(testSet, predictions)
print("The accuracy of the recognition model is: " + str(accuracy1))
