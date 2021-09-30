import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

res_gas_model = load_model(r'C:\Users\zhang\GitHub_projects\GTBR\Gastric_Biopsy_Cancer_Detector\model\resnet_gastric.h5')

def slice_load(file_list):    
    images=[]    
    for filename in file_list:
        im = image.load_img(filename,target_size=(224, 224, 3)) 
        b = image.img_to_array(im)
        images.append(b)
    return images

def get_prediction(path):
    X_inpute_image=slice_load(path)
    X_inpute_array=np.array(X_inpute_image)/255
    pred = res_gas_model.predict(X_inpute_array)
    print('Network prediction:', round(pred[0][0],4))
    return round(pred[0][0],4)

#test
#path=[r'G:\BaiduNetdiskDownload\testset\2017-06-10_20.52.36.ndpi.16.29083_14888.2048x2048.tiff']
#get_prediction(path)


