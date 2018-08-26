from joblib import Parallel, delayed
import imgaug as ia
from imgaug import augmenters as iaa
import multiprocessing
import os
import Augmentor
import pandas as pd
import cv2
import numpy as np

size = 350
if not os.path.exists("all/resize/"+str(size)):
    os.makedirs("all/resize/"+str(size))
    os.makedirs("all/resize/"+str(size)+"/ship")
    os.makedirs("all/resize/"+str(size)+"/noship")


df = pd.read_csv("train_ship_segmentations.csv", header=0)
df_unique = df.drop_duplicates("ImageId")


# Resize
sometimes1 = lambda aug: iaa.Sometimes(0.10, aug)
sometimes3 = lambda aug: iaa.Sometimes(0.30, aug)

def processInput(i):
    
    img = cv2.imread("all/train/"+i)
    img_resized = cv2.resize(img, (size,size))
    
    encoded = df.query('ImageId=="'+i+'"')['EncodedPixels'].tolist()
    if str(encoded[0]) == 'nan':
        name = "noship"
        seq = iaa.Sequential([
            sometimes1(iaa.CoarseSalt(p=0.07, size_percent=0.02)),
            sometimes1(iaa.GaussianBlur((0, 2.0)))
        ])
        images_aug = seq.augment_images([img_resized])
        img_resized = images_aug[0]
        cv2.imwrite("all/resize/"+str(size)+"/"+str(name)+"/"+i,img_resized)
        
    else:
        name = "ship"
        cv2.imwrite("all/resize/"+str(size)+"/"+str(name)+"/"+i,img_resized)
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            sometimes3(iaa.CoarseSalt(p=0.07, size_percent=0.02)),
            sometimes3(iaa.Affine(rotate=(-45, 45))),
            sometimes3(iaa.GaussianBlur((0, 2.0)))
        ])
        images_aug = seq.augment_images([img_resized])
        cv2.imwrite("all/resize/"+str(size)+"/"+str(name)+"/2-"+i,images_aug[0])
    
    
    
    
print("start ---- ")
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in list(reversed(df_unique["ImageId"].tolist())))