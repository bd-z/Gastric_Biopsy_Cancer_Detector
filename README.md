# Gastric_Biopsy_Cancer_Detector

Also named Gastric Tissue Biopsy Robot (GTBR), it uses convolutional neural networks to build a machine learning model to detect whether cancer cells exist in the pathological biopsy section of the stomach.

## Background
When society ages, cancer becomes more and more common. Early detection of cancer is the key to cancer treatment. Gastroscopy and gastric tissue biopsy are common methods to find gastric cancer. “The number of active pathologists in the U.S. plummeted between 2007 and 2017 by about 17.5% and puts the country at risk of a shortage in the future, though a smaller workforce is currently handling a bigger load of cases.” according to a study published on May 31 in JAMA Network Open. With machine learning technique to build a cancer cell identification system is a solution to make up for the lack of pathologists and to improve the accuracy of identification.

## Data
The data are images of stomach biopsy slices from the [2017 Data Science Competition in China](http://www.datadreams.org/#/newraceintro_detail?id=225). There are a total of 560 cancer samples and 140 non-cancer samples. With data augmentation cancer images are increased to 3361 pcs, and non-cancer images are increased to 2800pcs.

![image](https://github.com/bd-z/Gastric_Biopsy_Cancer_Detector/blob/main/static/asset/gtissue1.jpg)

## Model 1 -- simple CNN

A simple CNN model was built. With ann_visualizer the visualization of this CNN simplified as following:(Green button: activation layer with Relu, Purple button:  max pooling, Yellow button: drop out, Black triangle: flattening) The accuracy of the trained Model 1 on test data is 0.5475.

![image](https://github.com/bd-z/Gastric_Biopsy_Cancer_Detector/blob/main/static/asset/cnn_model.png)

## Model 2 With Transfer Learning
Based on ResNet50V2, model 2 has three hidden layers with drop out or BatchNormalization followed by GAP. The Output layer has 1 neuron with sigmoid activation. After training, the accuracy on test data is improved to 0.9446.

## Web Application
Inspired by [SkinCheck](https://github.com/leona-ha/Skin-Screening_Web-App) project of Leona, a Web App is developed with Model 2.  

![image](https://github.com/bd-z/Gastric_Biopsy_Cancer_Detector/blob/main/static/asset/web_app.png)



<video src="https://github.com/bd-z/Gastric_Biopsy_Cancer_Detector/blob/main/static/asset/GTBR_demo.mp4" data-canonical-src="https://github.com/bd-z/Gastric_Biopsy_Cancer_Detector/blob/main/static/asset/GTBR_demo.mp4" controls="controls" muted="muted" class="d-block rounded-bottom-2 width-fit" style="max-height:640px;">
  </video>

## License
[MIT](https://choosealicense.com/licenses/mit/)