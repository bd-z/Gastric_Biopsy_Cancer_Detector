# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 17:09:01 2021

@author: zhang
"""

from keras.preprocessing.image import ImageDataGenerator

def image_generator(from_folder, to_folder, n,m):
    '''
    Generate images
    from_folder is the folder where the original images saved
    to_folder is the folder where the generated images saved
    n is the times to perform generation
    m is the batch size
    '''
    
    datagen = ImageDataGenerator(
        zca_whitening=True,
        rotation_range=360,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        fill_mode='nearest')
    
    gener=datagen.flow_from_directory(from_folder,
                                         batch_size= m ,
                                         target_size=(2048,2048),
                                         shuffle=True,
                                         save_to_dir= to_folder,
                                         save_prefix='trans_',
                                         save_format='tiff')
    for i in range(n):
        gener.next()
        





#test1 cancer data augmentation
f3c=r'G:\BaiduNetdiskDownload\train\cancer'
f4cg=r'G:\BaiduNetdiskDownload\train\cancer_generated'
image_generator(f3c,f4cg,10,50) #469


#test2 non cancer augmentation
f5n=r'G:\BaiduNetdiskDownload\train\non_cancer'
f5ng=r'G:\BaiduNetdiskDownload\train\non_cancer_generated'
image_generator(f5n,f5ng,40,11)#110










