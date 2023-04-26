import cv2
import numpy as np
import time
import imutils as imutils
from PIL import Image, ImageTk
import os

if not os.path.exists('SignData'):
   os.makedirs('SignData')
if not os.path.exists('SignData/train'):
   os.makedirs('SignData/train')
if not os.path.exists('SignData/test'):
   os.makedirs('SignData/test')

   
global frame_copy
global background

class Creator():
    num_frames = 0
    element = 10
    num_imgs_taken = 0
    background = None
    def __init__(self,frame,panel,sign):
        #self.background = None
        self.frame=frame
        self.frame_copy=None
        self.panel=panel
        self.sign=sign
        self.accumulated_weight = 0.5
        self.ROI_top = 100
        self.ROI_bottom = 300
        self.ROI_right = 200
        self.ROI_left = 400
        self.result = "B"
        #print("Creator values",frame,panel,sign)
        if(frame=="None" and panel=="None" and sign=="None"):
            self.__class__.num_frames = 0
            self.__class__.element = 10
            self.__class__.num_imgs_taken = 0
            self.frame_copy = None
            self.result = None
            print("Destroyed")
        else:
            self.creating()
        
        
        

    def cal_accum_avg(self, gray_frame):
        #global background
        if self.background is None:
            self.__class__.background = gray_frame.copy().astype("float")
            return None

        cv2.accumulateWeighted(gray_frame, self.background, self.accumulated_weight)
        print("background weight")


    def segment_hand(self,frame,threshold=25):
        #global background
        diff = cv2.absdiff(self.background.astype("uint8"), frame)
        _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Grab the external contours for the image
        contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None
        else:
            
            hand_segment_max_cont = max(contours, key=cv2.contourArea)
            print("hand segment")
            
            return (thresholded, hand_segment_max_cont)

    def creating(self):
    ##    cam = cv2.VideoCapture(0)
    ##    ret, frame = cam.read()
        self.frame = cv2.flip(self.frame, 1)
        self.result = "some"
        self.frame_copy = self.frame.copy()

        roi = self.frame[self.ROI_top:self.ROI_bottom,self.ROI_right:self.ROI_left]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        #cv2.imshow("gf",gray_frame)

        if self.num_frames < 60:
            self.cal_accum_avg(gray_frame)
            if self.num_frames <= 59:
                
                cv2.putText(self.frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                #cv2.imshow("Sign Detection",frame_copy)
             
        #Time to configure the hand specifically into the ROI...
        elif self.num_frames <= 100:
            #cv2.imshow("gah",gray_frame)

            hand = self.segment_hand(gray_frame)
            #print("here")
            
            cv2.putText(self.frame_copy, "Adjust hand please..", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # Checking if hand is actually detected by counting number of contours detected...
            if hand is not None:
                
                thresholded, hand_segment = hand

                # Draw contours around hand segment
                cv2.drawContours(self.frame_copy, [hand_segment + (self.ROI_right, self.ROI_top)], -1, (255, 0, 0),1)
                
                cv2.putText(self.frame_copy, str(self.num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # Also display the thresholded image
                #cv2.imshow("Thresholded Hand Image", thresholded)
        
        else: 
            
            # Segmenting the hand region...
            hand = self.segment_hand(gray_frame)
            
            # Checking if we are able to detect the hand...
            if hand is not None:
                
                # unpack the thresholded img and the max_contour...
                thresholded, hand_segment = hand

                # Drawing contours around hand segment
                cv2.drawContours(self.frame_copy, [hand_segment + (self.ROI_right, self.ROI_top)], -1, (255, 0, 0),1)
                
                cv2.putText(self.frame_copy, str(self.num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                #cv2.putText(frame_copy, str(num_frames)+"For" + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(self.frame_copy, str(self.num_imgs_taken) + ' images captured', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # Displaying the thresholded image
                cv2.imshow("Thresholded Hand Image", thresholded)
                print(self.sign,str(self.num_imgs_taken))
                if self.num_imgs_taken <= 300:
                    if not os.path.exists('SignData/test/'+self.sign):
                        os.makedirs('SignData/test/'+self.sign)
                    if not os.path.exists('SignData/train/'+self.sign):
                        os.makedirs('SignData/train/'+self.sign)
                    #cv2.imwrite(r"D:\\gesture\\train\\"+str(element)+"\\" + str(num_imgs_taken+300) + '.jpg', thresholded)
                    cv2.imwrite("SignData/train/"+self.sign+"/" + str(self.num_imgs_taken) + '.jpg', thresholded)
                    cv2.imwrite("SignData/test/"+self.sign+"/" + str(self.num_imgs_taken) + '.jpg', thresholded)
                else:
                    cv2.destroyAllWindows()
                    self.__class__.num_frames = 0
                    self.__class__.element = 10
                    self.__class__.num_imgs_taken = 0
                    self.result = None
                    return None
                self.__class__.num_imgs_taken +=1
                print(self.num_imgs_taken)
            else:
                cv2.putText(self.frame_copy, 'No hand detected...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Drawing ROI on frame copy
        cv2.rectangle(self.frame_copy, (self.ROI_left, self.ROI_top), (self.ROI_right, self.ROI_bottom), (255,128,0), 3)
        
        #cv2.putText(frame_copy, "Your Text Here", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
        
        # increment the number of frames for tracking
        self.__class__.num_frames += 1
        print("num frame" ,self.num_frames)

        # Display the frame with segmented hand
        #cv2.imshow("Sign Detection", frame_copy)
        #image = cv2.cvtColor(self.frame_copy, cv2.COLOR_BGR2RGB)
        #image = cv2.flip(image, 1)
##        image = Image.fromarray(image)
##        image = ImageTk.PhotoImage(image)
##        self.panel.configure(image=image)
##        self.panel.image = image
        return self.frame_copy

    def __del__(self):
        print("released")


        
    
# Releasing camera & destroying all the windows...

    
#cv2.destroyAllWindows()
#cam.release()
