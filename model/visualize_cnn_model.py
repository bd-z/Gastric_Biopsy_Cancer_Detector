# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 21:04:12 2021

@author: zhang
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, BatchNormalization, MaxPool2D ,Activation, MaxPooling2D
from ann_visualizer.visualize import ann_viz
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras import backend as K
   
def build_cnn_model():    

    model = Sequential()
    
    
    # 1st Convolutional Layer
    model.add(Conv2D(filters=9, input_shape=(224,224,3), kernel_size=(11,11),#96
      strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
   # model.add(BatchNormalization())
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=25, kernel_size=(2,2), strides=(1,1), padding='valid'))#256#2x2
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
   # model.add(BatchNormalization())
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=38, kernel_size=(3,3), strides=(1,1), padding='valid'))#384
    model.add(Activation('relu'))
    # Batch Normalisation
 #   model.add(BatchNormalization())
    
    # 4th Convolutional Layer
    model.add(Conv2D(filters=38, kernel_size=(3,3), strides=(1,1), padding='valid'))#384
    model.add(Activation('relu'))
    # Batch Normalisation
   # model.add(BatchNormalization())
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters=25, kernel_size=(3,3), strides=(1,1), padding='valid'))#256
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
  #  model.add(BatchNormalization())
    
    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    
    model.add(Dense(40, input_shape=(224*224*1,)))#4096
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
  #  model.add(BatchNormalization())
    
    # 2nd Dense Layer
    model.add(Dense(40))#4096
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
 #   model.add(BatchNormalization())
    
    # 3rd Dense Layer
    model.add(Dense(10))#1000
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
  #  model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
    
model = build_cnn_model()

ann_viz(model, title="GTBR")

#only the first model can be visualized!

#clear sessions

K.clear_session()



input_shape = (512, 512, 3)
# transfer learning with ResNet50V2
resMod = ResNet50V2(include_top=False, weights='imagenet',
				  input_shape=input_shape)
#frozen the layers in ResNet50V2
for layer in resMod.layers:
    layer.trainable = False

def build_ResNet_transferlearning_model():    

    model = Sequential()

    model.add(resMod)
    
    
    model.add(tf.keras.layers.GlobalAveragePooling2D()) 
    
    #1st Dense: (None, 60) 
    model.add(keras.layers.Dense(60, activation='relu'))  
    #regularization with penalty term
    model.add(Dropout(0.2))
    
    # 2nd Dense: (None, 50)
    model.add(keras.layers.Dense(50, activation='relu'))
    #regularization
    model.add(keras.layers.BatchNormalization())  
    
    # 2nd Dense: (None, 50)
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.BatchNormalization()) 
    
    # Output Layer: (None, 1)  
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.summary()
    return model