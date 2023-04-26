#!/usr/bin/python
# -*- coding: utf-8 -*-
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from tkinter.ttk import *
import time
import threading
import pandas as pd
import mediapipe as mp
import numpy as np
import pickle
import time

word_table = pd.read_excel('word_table.xlsx')
words = word_table.to_dict('records')
#print(words)
global findHands
global trainCnt
trainCnt = 0
knownGestures = []
gestNames = []
total = 10
trainName = 'trained_pickle.pkl'
numGest = len(list(words[0].values()))
#print(numGest)
hands = mp.solutions.hands.Hands(static_image_mode=True,
                                 max_num_hands=1,
                                 min_detection_confidence=.5,
                                 min_tracking_confidence=.5)


def mpHands(frame):
    myHands = []
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #print("frameRGB",frameRGB)
    #cv2.imshow("rgb",frameRGB)
    results = hands.process(frameRGB)
    #print("results",results.multi_hand_landmarks)
    if results.multi_hand_landmarks != None:
        for handLandMarks in results.multi_hand_landmarks:
            myHand = []
            for landMark in handLandMarks.landmark:
                #print("dude",landMark)
                myHand.append((int(landMark.x * width),
                        int(landMark.y * height)))
            myHands.append(myHand)
    #print("mpHands",myHands)
    return myHands


def findDistances(handData):
    distMatrix = np.zeros([len(handData), len(handData)], dtype='float')
    #print("handdata",handData)
    palmSize = ((handData[0][0] - handData[9][0]) ** 2
                + (handData[0][1] - handData[9][1]) ** 2) ** (1. / 2.)
    for row in range(0, len(handData)):
        for column in range(0, len(handData)):
            distMatrix[row][column] = ((handData[row][0]
                    - handData[column][0]) ** 2 + (handData[row][1]
                    - handData[column][1]) ** 2) ** (1. / 2.) / palmSize
    return distMatrix


def findError(gestureMatrix, unknownMatrix, keyPoints):
    error = 0
    for row in keyPoints:
        for column in keyPoints:
            error = error + abs(gestureMatrix[row][column]- unknownMatrix[row][column])
    #print (error)
    return error


def findGesture(unknownGesture,knownGestures,keyPoints,gestNames,tol):
    errorArray = []
    for i in range(0, len(gestNames), 1):
        error = findError(knownGestures[i], unknownGesture, keyPoints)
        errorArray.append(error)
    errorMin = errorArray[0]
    minIndex = 0
    for i in range(0, len(errorArray), 1):
        if errorArray[i] < errorMin:
            errorMin = errorArray[i]
            minIndex = i
    if errorMin < tol:
        gesture = gestNames[minIndex]
    if errorMin >= tol:
        gesture = 'Unknown'
    return gesture


def toggleRecordText():
    if Record_button['text'] == 'Start Record':
        Record_button['text'] = 'Capture'
    elif Record_button['text'] == 'Capture':
        Record_button['text'] = 'Start Record'

    if Record_button['text'] == 'Capture':
        Record_button['state'] = 'normal'
        Train_button['state'] = 'normal'
        Predict_button['state'] = 'disabled'
    else:
        Record_button['state'] = 'normal'
        Train_button['state'] = 'normal'
        Predict_button['state'] = 'normal'
        

                
def togglePredictText():
    if Predict_button['text'] == 'Start Predict':
        Predict_button['text'] = 'Stop Predict'
    elif Predict_button['text'] == 'Stop Predict':
        Predict_button['text'] = 'Start Predict'
    if Predict_button['text'] == 'Stop Predict':
        Record_button['state'] = 'disabled'
        Train_button['state'] = 'disabled'
        Predict_button['state'] = 'normal'
    else:
        Record_button['state'] = 'normal'
        Predict_button['state'] = 'normal'
        Train_button['state'] = 'normal'


def Creator(frame, sign,temp):
    #print("temp",temp)
    if temp!=[]:
        frame = cv2.flip(frame, 1)
        #cv2.imshow("asdsd",frame)
        handData = mpHands(frame)
        if handData != []:
            #print("Handsss",handData)
            for hand in handData:
                for ind in keyPoints:
                    cv2.circle(frame, hand[ind], 10, (255, 0, 255), 3)
            #cv2.imshow('Sign', frame)
            gestNames.append(sign)
            knownGesture = findDistances(handData[0])
            knownGestures.append(knownGesture)
##            with open(trainName, 'wb') as f:
##                pickle.dump(gestNames, f)
##                pickle.dump(knownGestures, f)
    else:
        frame = cv2.flip(frame, 1)
    return frame


def Trainer():
    print("gest",gestNames)
    with open(trainName, 'wb') as f:
                pickle.dump(gestNames, f)
                pickle.dump(knownGestures, f)
    

def Predictor(frame, sign,temp):
    #cv2.imshow("frame",frame)
    #print("temp",temp)
    if temp!=[]:
        frame = cv2.flip(frame, 1)
        handData = mpHands(frame)
        if handData != []:
            #print("handData",handData)
            with open(trainName, 'rb') as f:
                gestNames = pickle.load(f)
                knownGestures = pickle.load(f)
            if frame != []:
                unknownGesture = findDistances(handData[0])
                myGesture = findGesture(unknownGesture, knownGestures,keyPoints, gestNames, total)
                #print("myGesture",myGesture)
                cv2.putText(frame,myGesture,(50, 100),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 0, 0),5)
    else:
        frame = cv2.flip(frame, 1)
    return frame


def Dummy(frame, sign, temp):
    if temp!=[]:
        #print("sds")
        #frame = cv2.flip(frame, 1)
         if Record_button['text'] == 'Capture':
             frame=Creator(frame, sign, temp)
         elif Predict_button['text'] == 'Stop Predict':
             frame=Predictor(frame, sign, temp)
    else:
        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            'Hand not detected',
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            5,
            )
    return frame


class MainWindow:

    sign = None

    def __init__(self,window,cap,action_class, action):
        self.window = window
        self.cap = cap
        self.action = action
        self.action_class = action_class
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 'idle'
        self.sign = sign_selection.get()
        if cap.isOpened() == False:
            self.cap = cv2.VideoCapture(0)

        if self.action == 'Record' and Record_button['text']  == 'Start Record':
            print ('1', Record_button['text'])
        elif self.action == 'Record' and Record_button['text']  == 'Capture':
            print ('2', Record_button['text'])
            self.update_image()
        if self.action == 'Predict' and Predict_button['text'] == 'Start Predict':
            print ('1', Predict_button['text'])
        elif self.action == 'Predict' and Predict_button['text'] == 'Stop Predict':
            print ('2', Predict_button['text'])
            self.update_image()

    def update_image(self):

            # print("3",Record_button['text'])

        findHands = mpHands(self.cap.read()[1])
        if findHands==[]:
            #print("No hands")
            self.action_class=Dummy
        else:
            self.action_class=self.action_class
        #print("class",self.action_class)
        if Record_button['text'] == 'Capture' or Predict_button['text'] == 'Stop Predict':
            self.frame = self.action_class(self.cap.read()[1],self.sign,findHands)
            if self.frame != []:

                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                self.window.configure(image=image)
                self.window.image = image
                self.window.after(self.interval, self.update_image)
            else:

                            # self.hand_data = self.action_class("None","None")

                self.cap.release()
                cv2.destroyAllWindows()
                placeholderImage(self.window)
                if self.action == 'Record':
                    toggleRecordText()
                elif self.action == 'Predict':
                    togglePredictText()
        else:

                    # self.hand_data = self.action_class("None","None")

            self.cap.release()
            cv2.destroyAllWindows()
            placeholderImage(self.window)

    def __del__(self):

            # print("4",Record_button['text'])

        print ('stopped')


def placeholderImage(panel):
    image = cv2.imread('test.jpg')
    image = cv2.resize(image, (500, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel.configure(image=image)
    panel.image = image


    # gp.gifplay(video_window,"test.gif",100)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Sign Detection')
    root.geometry('800x600')
    width = 650
    height = 470
    options = []
    keyPoints = [ 0,  4,  5,  9,13,  17,   8, 12, 16, 20]
    #keyPoints = [ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    # Dropdown menu options

    for vals in words[0].values():
        options.append(vals)

    sign_selection = tk.StringVar()

    # initial menu text

    sign_selection.set(list(words[0].values())[0])
    root.resizable(0, 0)

    # Create Dropdown menu

    drop_label = tk.Label(root, text='Select Sign', width=10)

    # drop_label.config(bg="white")

    drop_label.place(x=10, y=30)

    drop = tk.OptionMenu(root, sign_selection, *options)
    drop.config(width=5)
    drop.config(bg='Gray', fg='White')
    drop.place(x=10, y=30)

    video_window = tk.Label(root, width=650, height=450)
    video_window.config(bg='white')
    video_window.place(x=100, y=30)

    # button to start creating data

    cap = cv2.VideoCapture(0)

    time.sleep(2)

    Record_button = tk.Button(root,text='Start Record', bg='red',fg='white', width=25,height=2,command=lambda : [toggleRecordText() ,MainWindow(video_window,cap, Creator, 'Record')])
    #Record_button.place(x=125, y=500)
    
    Train_button = tk.Button(root, text='Save Model', bg='green', fg='white', width=25,height=2, command=lambda:[Trainer()])
    #Train_button.place(x=325, y=500)
    
    Predict_button = tk.Button(root,text='Start Predict',bg='blue',fg='white', width=25, height=2,command=lambda : [togglePredictText(), MainWindow(video_window,cap, Predictor, 'Predict')])
    Predict_button.place(x=325, y=500)
    #findHands = mpHands(cap.read()[1])
    root.mainloop()
