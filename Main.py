import tkinter as tk
from PIL import Image, ImageTk
import cv2
import Creator as Cr
import Predictor as Pr
import Trainer as Tr
from tkinter.ttk import *
import time
import threading
import pandas as pd
word_table = pd.read_excel('word_table.xlsx')
words = word_table.to_dict('records')
print(words)

def toggleRecordText():
        if Record_button['text']=="Start Record":
                Record_button['text']="Stop Record"
        elif Record_button['text']=="Stop Record":
                Record_button['text']="Start Record"

        if  Record_button["text"]=="Stop Record":
                Record_button["state"] = "normal"
                Train_button["state"] = "disabled"
                Predict_button["state"] = "disabled"
        else:
                Record_button["state"] = "normal"
                Train_button["state"] = "normal"
                Predict_button["state"] = "normal"

def toggleTrainText():
        if Train_button['text']=="Start Train":
                Train_button['text']="Training"
        elif Train_button['text']=="Training":
                Train_button['text']="Start Train"
        if  Train_button["text"]=="Training":
                Record_button["state"] = "disabled"
                Train_button["state"] = "disabled"
                Predict_button["state"] = "disabled"
        else:
                Record_button["state"] = "normal"
                Train_button["state"] = "normal"
                Predict_button["state"] = "normal"
                
def togglePredictText():
        if Predict_button['text']=="Start Predict":
                Predict_button['text']="Stop Predict"
        elif Predict_button['text']=="Stop Predict":
                Predict_button['text']="Start Predict"
        if  Predict_button["text"]=="Stop Predict":
                Record_button["state"] = "disabled"
                Train_button["state"] = "disabled"
                Predict_button["state"] = "normal"
        else:
                Record_button["state"] = "normal"
                Train_button["state"] = "normal"
                Predict_button["state"] = "normal"
                
class MainWindow():
    sign = None
    def __init__(self,window, cap , action_class, action):
        self.window = window
        self.cap = cap
        self.action = action
        self.action_class = action_class
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = "idle"
        self.sign=sign_selection.get()
        if cap.isOpened()==False:
                self.cap=cv2.VideoCapture(0)
        
        if self.action=="Record" and Record_button['text']=="Start Record":
            print("1",Record_button['text'])
        elif self.action=="Record" and Record_button['text']=="Stop Record":
                print("2",Record_button['text'])
                self.update_image()
        if self.action=="Train" and Train_button['text']=="Start Train":
            print("1",Train_button['text'])
        elif self.action=="Train" and Train_button['text']=="Training":
                print("2",Train_button['text'])
                self.update_image()
        if self.action=="Predict" and Predict_button['text']=="Start Predict":
            print("1",Predict_button['text'])
        elif self.action=="Predict" and Predict_button['text']=="Stop Predict":
                print("2",Predict_button['text'])
                self.update_image()
        

    def update_image(self):
            #print("3",Record_button['text'])
            if(Record_button['text']=="Stop Record" or Train_button['text']=="Training" or Predict_button['text']=="Stop Predict"):
                    self.c_obj2 = self.action_class(self.cap.read()[1],self.window,self.sign)
                    if(self.c_obj2.result!=None):
                            if self.c_obj2.frame_copy=="None":
                                    placeholderImage(self.window)
                            else:
                                image = cv2.cvtColor(self.c_obj2.frame_copy, cv2.COLOR_BGR2RGB)
                                image = Image.fromarray(image)
                                image = ImageTk.PhotoImage(image)
                                self.window.configure(image=image)
                                self.window.image = image
                                self.window.after(self.interval, self.update_image)
                    else:
                            self.c_obj2 = self.action_class("None","None","None")
                            self.cap.release()
                            cv2.destroyAllWindows()
                            placeholderImage(self.window)
                            if self.action=="Record":
                                    toggleRecordText()
                            elif self.action=="Train":
                                    toggleTrainText()
                            elif self.action=="Predict":
                                    togglePredictText()
            else:
                    self.c_obj2 = self.action_class("None","None","None")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    placeholderImage(self.window)
                        
                

    def __del__(self):
            #print("4",Record_button['text'])
            print("stopped")
            
            
def placeholderImage(panel):
        if Train_button['text']=="Training":
           image =cv2.imread("relax.jpg")
        else:
            image =cv2.imread("test.jpg")
    
        image = cv2.resize(image,(500, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panel.configure(image=image)
        panel.image = image  
    #gp.gifplay(video_window,"test.gif",100)
    
if __name__ == "__main__":
    root = tk.Tk()
    root.title('Sign Detection')
    root.geometry( "800x600" )
    options=[]
    # Dropdown menu options
    for vals in words[0].values():
            options.append(vals)
            
    sign_selection = tk.StringVar()  
    # initial menu text
    sign_selection.set(list(words[0].values())[0])
    root.resizable(0, 0)         
    
    # Create Dropdown menu
    drop_label = tk.Label(root, text="Select Sign", width=10)
    #drop_label.config(bg="white")
    drop_label.place(x=10, y=30)
    
    drop = tk.OptionMenu( root , sign_selection , *options )
    drop.config(width=5)
    drop.config(bg="Gray", fg="White")
    drop.place(x=10, y=30)
    
    video_window = tk.Label(root, width=650, height=450)
    video_window.config(bg="white")
    video_window.place(x=100, y=30)
    
    #button to start creating data
    cap = cv2.VideoCapture(0)
    Record_button = tk.Button(root, text='Start Record', bg='red', fg='white', width=25,height=2, command=lambda:[toggleRecordText(),MainWindow(video_window, cap, Cr.Creator,"Record")])
    Record_button.place(x=125, y=500)
    
    Train_button = tk.Button(root, text='Start Train', bg='green', fg='white', width=25,height=2, command=lambda:[toggleTrainText(),MainWindow(video_window, cap, Tr.Trainer,"Train")])
    Train_button.place(x=325, y=500)

    #predict button start
    Predict_button = tk.Button(root, text='Start Predict', bg='blue', fg='white', width=25,height=2, command=lambda:[togglePredictText(),MainWindow(video_window, cap, Pr.Predictor,"Predict")])
    #Predict_button.place(x=525, y=500)
    #Predict button end

   
    #MainWindow(video_window, cv2.VideoCapture(0))
    root.mainloop()
