#Breast Tumor Detection With U-Net

This projects aims to detect and classify tumors on mammography imagesThis projects aims to detect and classify tumors on mammography images with U-Net.

##Dataset 

Dataset can be reached by following link: 
https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#1db304a576184c7eb14f0bca91e5838e

The dataset includes original mammography images and their masks that were marked by radiologists.

##Preproceesing Data

Original images had low contrast. Pixel values are confined to some specific range of values only and images are not clear enough to detect tumors correctly. For this reason, original images need to become clearer. 

The following script equalizes hisgtogram:
[histogram.py](./preproceesing/histogram.py)

##Model Training 
The dataset includes limited number of images. Consequently, images are augmented by [ImageDataGenerator](https://keras.io/preprocessing/image/)

[U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) is implemented at [model.py](./model.py) and trained.