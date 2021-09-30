# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 20:33:21 2021

@author: zhang
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, BatchNormalization, MaxPool2D ,Activation, MaxPooling2D


def data_table(folder):
    '''create a dataframe which has 'id' and 'label' columns. The id column is the path of each image
    and the label column contain 1 and 0 which indicate cancer cells exist or not 
    '''    
    p=os.walk(folder)
    list_empty=[]
    dict_empty={}
    for path, dir_list,file_list in p:
        for file_name in file_list:
            file_path=os.path.join(path,file_name)
            list_empty.append(file_path)            
    for file_path in list_empty:
        if 'non_cancer' in file_path:
            label=0
        else:
            label=1
        dict_empty['{}'.format(file_path)]=label
    df = pd.DataFrame.from_dict(dict_empty, orient='index',columns=['label'])
    df = df.reset_index().rename(columns={'index':'id'})   
    df = shuffle(df)    
    return df

#folder where the images data stored
f=r'G:\BaiduNetdiskDownload\train'


df_full=data_table(f)   

#define X and y
X=df_full['id']
y=df_full['label']

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100) # split into test and train sets


def slice_load(file_list):
    ''' load the images'''    
    images=[]    
    for filename in file_list:        
        im = image.load_img(filename,target_size=(512, 512, 3)) 
        b = image.img_to_array(im)
        images.append(b)
    return images

X_train_image=slice_load(X_train)
X_train_array=np.array(X_train_image)/255

X_test_image=slice_load(X_test)
X_test_array=np.array(X_test_image)/255

X_train_array.shape
type(y_train)


#clear sessions

K.clear_session()

input_shape = (512, 512, 3)


# transfer learning with ResNet50V2
resMod = ResNet50V2(include_top=False, weights='imagenet',
				  input_shape=input_shape)

#frozen the layers in ResNet50V2
for layer in resMod.layers:
    layer.trainable = False

# build model
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


# Compile 
model.compile(loss='categorical_crossentropy', optimizer='adam',\
 metrics=['accuracy'])

#add early stoping
callback = EarlyStopping(monitor='val_loss', patience=3)

#(5)Train

results=model.fit(X_train_array, y_train, batch_size=64, epochs=50, verbose=1, \
validation_split=0.2,callbacks=[callback], shuffle=True)


model.evaluate(X_test_array, y_test)

results.history['val_accuracy']
#save model
model.save(r'C:\Users\zhang\GitHub_projects\GTBR\Gastric_Biopsy_Cancer_Detector\model\resnet_gastric.h5')



















