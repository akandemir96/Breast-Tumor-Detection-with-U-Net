import cv2
import os

full_image_path = "./X_Data"
text_list = list()
current = 0
image_list = list()

for root, dirs, files in os.walk(full_image_path, topdown=False):
    for name in files:
        image_list.append(os.path.join(root, name))

for image in image_list:
    img = cv2.imread(image, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)

    new_path = "./X_Processed_Data/"
    name = new_path+image.split("/")[-1]
    cv2.imwrite(name, cl1)