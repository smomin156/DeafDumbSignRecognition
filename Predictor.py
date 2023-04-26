import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#from gtts import gTTS
import os
import time
from playsound import playsound
import imutils as imutils
from PIL import Image, ImageTk
import pandas as pd
word_table = pd.read_excel('word_table.xlsx')
words = word_table.to_dict('records')
print(words)

model = keras.models.load_model(r"trained_model.h5")
global frame_copy
global background

class Predictor():
    num_frames =0
    background = None
    def __init__(self,frame,panel,sign):
        self.frame=frame
        self.panel=panel
        self.sign=sign
        self.frame_copy=None
        self.accumulated_weight = 0.5
        self.ROI_top = 100
        self.ROI_bottom = 300
        self.ROI_right = 200
        self.ROI_left = 400
        self.word_dict = words
        self.result = "B"
        if(frame=="None" and panel=="None" and sign=="None"):
            self.__class__.num_frames = 0
            self.__class__.element = 10
            self.__class__.num_imgs_taken = 0
            self.frame_copy = None
            self.result = None
            print("Destroyed")
        else:
            self.predicting()

    def cal_accum_avg(self, gray_frame):

       # global background
        
        if self.background is None:
            self.__class__.background = gray_frame.copy().astype("float")
            return None

        cv2.accumulateWeighted(gray_frame, self.background, self.accumulated_weight)



    def segment_hand(self, frame, threshold=25):
        #global background
        
        diff = cv2.absdiff(self.background.astype("uint8"), frame)

        
        _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
        contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If length of contours list = 0, means we didn't get any contours...
        if len(contours) == 0:
            return None
        else:
            # The largest external contour should be the hand 
            hand_segment_max_cont = max(contours, key=cv2.contourArea)
            
            # Returning the hand segment(max contour) and the thresholded image of hand...
            return (thresholded, hand_segment_max_cont)

    def predicting(self):
    #cam = cv2.VideoCapture(0)
    #ret, frame = cam.read()
        self.frame = cv2.flip(self.frame, 1)
        self.result = "some"
        self.frame_copy = self.frame.copy()

        # ROI from the frame
        roi = self.frame[self.ROI_top:self.ROI_bottom, self.ROI_right:self.ROI_left]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)


        if self.num_frames < 70:
            
            self.cal_accum_avg(gray_frame)
            
            cv2.putText(self.frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        else: 
            # segmenting the hand region
            hand = self.segment_hand(gray_frame)
            

            # Checking if we are able to detect the hand...
            if hand is not None:
                
                thresholded, hand_segment = hand

                # Drawing contours around hand segment
                cv2.drawContours(self.frame_copy, [hand_segment + (self.ROI_right, self.ROI_top)], -1, (255, 0, 0),1)
                
                cv2.imshow("Thesholded Hand Image", thresholded)
                
                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
                
                pred = model.predict(thresholded)
                print("prediction",np.argmax(pred))
                #myobj = gTTS(text=self.word_dict[np.argmax(pred)], lang="en", slow=False)
                #myobj.save("predict.mp3")
                #playsound(myobj)
                cv2.putText(self.frame_copy, list(self.word_dict[0].values())[np.argmax(pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
        # Draw ROI on frame_copy
        cv2.rectangle(self.frame_copy, (self.ROI_left, self.ROI_top), (self.ROI_right, self.ROI_bottom), (255,128,0), 3)

        # incrementing the number of frames for tracking
        self.__class__.num_frames += 1
        print("num frame" ,self.num_frames)

        # Display the frame with segmented hand
        #cv2.putText(frame_copy, "Your Text Here", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
        #cv2.imshow("Sign Detection", frame_copy)
        #image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(image)
        #image = ImageTk.PhotoImage(image)
        #self.panel.configure(image=image)
        #self.panel.image = image
        #self.result = None
        return self.frame_copy

    def __del__(self):
        print("released")

