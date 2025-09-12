# IP102

El objetivo de este proyecto sera alcanzar la mejor precision posible en la clasificacion de imagenes del [dataset IP102](https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset?select=val.txt).

Este dataset tiene la siguiente descripcion:

"*Context*

*Insect pest are one of the main factors affecting agricultural product. Accurate recognition and classification of insect pests can prevent huge economic losses. This dataset will play a great role in this regard.*

*Content*

*IP02 dataset has 75,222 images and average size of 737 samples per class. The dataset has a split of 6:1:3. There are 8 super classes. Rice, Corn, Wheat, Beet, Alfalfa belong to Field Crop(FC) and Vitis, Citrus, Mango belong to Economic Crop(EC).*"

En este proyecto se implementaran diferentes arquitecturas y tecnicas de deep learning basadas en CNNs, entre ellas:

* ResNet
* EfficientNet
* DenseNet
* ResNext
* VGG
* Separable Convolutions


Por otro lado, se implementaran tecnicas para favorecer el aprendizaje:

* LR Scheduling
* Weights Initialization
* Normalization
* Regularization : Dropout, Dropblock, Weight Decay
* Parallel Processing 
* DataAugmentation
* Automatic Mixed Precision
* Stochastic Depth regularization

# Carga del conjunto de datos

