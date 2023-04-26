import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import itertools
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
import cv2
import keras.callbacks
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
warnings.simplefilter(action='ignore', category=FutureWarning)
import threading
import pandas as pd
import os
from threading import Thread
import sys
from threading import Event

word_table = pd.read_excel('word_table.xlsx')
words = word_table.to_dict('records')
print(words)
if not os.path.exists('SignData'):
   os.makedirs('SignData')
if not os.path.exists('SignData/train'):
   os.makedirs('SignData/train')
if not os.path.exists('SignData/test'):
   os.makedirs('SignData/test') 

global t1
global run_event

class Trainer():
    train_path = r'SignData\train'
    test_path = r'SignData\test'
    batch_size = len(list(os.walk(r'SignData\train')))-1
    print("batch_size",batch_size)
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(64,64), class_mode='categorical', batch_size=batch_size,shuffle=True)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(64,64), class_mode='categorical', batch_size=batch_size, shuffle=True)
    epochs = 10
    word_dict = words
    
    
    def __init__(self,frame,panel,sign):
        self.frame=frame
        self.frame_copy="None"
        self.panel=panel
        self.sign=sign
        self.result = "B"
        self.run_event = threading.Event()
        self.run_event.set()
        self.t1=Thread(target=self.training)
        #t1.daemon = True
        if(frame=="None" and panel=="None" and sign=="None"):
            self.__class__.num_frames = 0
            self.__class__.element = 10
            self.__class__.num_imgs_taken = 0
            self.frame_copy = None
            self.result = None
            print("Destroyed")
            sys.exit()
        else:
            try:
               self.t1.start()
            except KeyboardInterrupt as e:
               self.run_event.clear()
               #self.t1.join()
               print(f"Exception name:{e}")

    def training(self):        
        imgs, labels = next(self.train_batches)
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        model.add(Flatten())

        model.add(Dense(64,activation ="relu"))
        model.add(Dense(128,activation ="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(128,activation ="relu"))
        #model.add(Dropout(0.3))
        model.add(Dense(self.batch_size,activation ="softmax"))


        # In[23]:


        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')



        model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


        history2 = model.fit(self.train_batches, epochs = self.epochs, callbacks=[reduce_lr, early_stop],  validation_data =self.test_batches)
        imgs, labels = next(self.train_batches) # For getting next batch of imgs...

        imgs, labels = next(self.test_batches) # For getting next batch of imgs...
        scores = model.evaluate(imgs, labels, verbose=0)
        print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')


        model.save('trained_model.h5')

        #print(history2.history)

        imgs, labels = next(self.test_batches)

        model = keras.models.load_model(r"trained_model.h5")

        scores = model.evaluate(imgs, labels, verbose=0)
        print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

        model.summary()

        scores #[loss, accuracy] on test data...
        model.metrics_names


        

        predictions = model.predict(imgs, verbose=0)
        print("predictions on a small set of test data--")
        print("")
        for ind, i in enumerate(predictions):
            print(list(self.word_dict[0].values())[np.argmax(i)], end='   ')


        print('Actual labels')
        for i in labels:
            print(list(self.word_dict[0].values())[np.argmax(i)], end='   ')

        #print(imgs.shape)

        #history2.history
        self.result = None
        self.run_event.clear()
        #self.t1.join()
        #cv2.imshow("Sign Detection", frame_copy)
        return None

    def __del__(self):
        print("released")
