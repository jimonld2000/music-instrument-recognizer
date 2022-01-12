from contextlib import nullcontext
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os
import pickle
import random
import operator
import sounddevice as sd
import soundfile as sf
from tkinter import *

# import filedialog module
from tkinter import filedialog
from PIL import Image, ImageTk
import math
from collections import defaultdict


# ------------Functional programming----------
dataset = []

# Loading the dataset of the learned files


def loadDataset(filename):
    with open("model.dat", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break


loadDataset("model.dat")

# Defining again the distance functions for instances


def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(),
                 np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

# Defining again the k nearest neighbour function


def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + \
            distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Defining again the nearest class function


def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(),
                    key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


results = defaultdict(int)

# Defining the result vector, its elements are named after the dataset's folder names
i = 1
for folder in os.listdir("./InstrumentSamples"):
    results[i] = folder
    i += 1


# -------------GUI programming--------------

# Global variable for opened file indication
openedfile = 0
# Function for opening the file explorer window

def browseFiles():
    global openedfile
    openedfile = filedialog.askopenfilename(initialdir="/",
                                            title="Select a File",
                                            filetypes=(("Sound files",
                                                        "*.wav*"),))

    # Change label contents
    label_file_explorer.configure(text="File Opened: "+openedfile)

# Function to record sound to be analyzed
def AudioRec():
    label_prediction.configure(text="Recording...")
    fs = 16000
    global openedfile
    # seconds
    duration = 5
    myrecording = sd.rec(int(duration * fs),
                         samplerate=fs, channels=2)
    sd.wait()
    # Save as wav file at correct sampling rate
    sf.write('Recording.wav', myrecording, fs)
    openedfile = "Recording.wav"
    label_prediction.configure(text="Recording finished")

# Function which executes the inference with the model
def Inference():
    global openedfile
    if (openedfile != 0):
        (rate, sig) = wav.read(openedfile)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, numcep=1000,
                         nfft=1024, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, 0)

        pred = nearestClass(getNeighbors(dataset, feature, 5))

        # Showing the prediction's result
        print(results[pred])
        label_prediction.configure(text="Prediction=" + results[pred])


# Create the root window
window = Tk()

# Set window title
window.title('Musical Intrument Recognizer')

# Set window size
window.geometry("700x500")

# Set window background color
window.config(background="white")
window.resizable(False, False)

# Creating a photoimage object to use image
image = Image.open("instruments_photo.png")
image_res = image.resize((400, 300), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(image_res)


# Create a File Explorer label
label_file_explorer = Label(window,
                            text="Choose a .wav file to be analyzed or click 'Record Audio' to record some audio!",
                            width=75, height=6,
                            fg="blue")

# Create a Prediction Result label
label_prediction = Label(window,
                         text="Prediction:",
                         width=75, height=6,
                         fg="blue")

# Button to Browse Files
button_explore = Button(window,
                        text="Browse Files",
                        command=browseFiles,
                        width=25, height=6)
# Button for Exit
button_exit = Button(window,
                     text="Exit",
                     command=exit,
                     width=25, height=6)
# Button for Prediction
button_test = Button(window,
                     text="Prediction",
                     command=Inference,
                     width=25, height=6)
# Button to Record Audio
button_record = Button(window,
                       text="Record Audio",
                       command=AudioRec,
                       width=25, height=6)
# Button with Image
label_image = Label(window,
                    image=photo, height=200)
label_image.image = photo

# Grid method to display widgets
label_file_explorer.grid(column=0, row=0, columnspan=3)
label_prediction.grid(column=0, row=1, columnspan=3)
label_image.grid(column=0, row=2, columnspan=3, rowspan=2, pady=(0.0))
button_explore.grid(column=3, row=0, padx=(0, 0))
button_test.grid(column=3, row=1, padx=(0, 0))
button_record.grid(column=3, row=2, padx=(0, 0))
button_exit.grid(column=3, row=3, padx=(0, 0))


# Let the window wait for any events
window.mainloop()
