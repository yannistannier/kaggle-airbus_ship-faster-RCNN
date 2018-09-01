import os 
import sys
import random
import math
import numpy as np
import cv2
import random
import json
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

DATA_DIR = '/Users/yannis/Developpement/kaggle/airbus/all/train'
ROOT_DIR = '/Users/yannis/Developpement/kaggle/airbus'
ROOT_MODEL = '/Users/yannis/Developpement/kaggle/airbus/models/MaskRcnn/'

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def rle_decode(mask_rle, shape=(768, 768)):
#     if str(mask_rle) == 'nan':
#         return np.zeros(shape[0]*shape[1], dtype=np.uint8).reshape(shape).T 
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def mask_overlay(image, mask):
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.75, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def get_dicom_fps(dicom_dir):
    df = pd.read_csv(dicom_dir, header=0)
    return list(set(df["ImageId"]))[0:200], df

def parse_dataset(dicom_dir): 
    list_images, df = get_dicom_fps(dicom_dir)
    image_fps = [ "all/train/"+row for row in list_images]
    image_annotations = {fp: [] for fp in image_fps}
    i = 0 
    for row in list_images:
        rles = df.query('ImageId=="'+row+'"')['EncodedPixels'].tolist()
        if str(rles[0]) != 'nan':
            i += 1 
            if i % 3 == 0:
                image_fps.append("all/train/"+row)
            else:
                image_fps.append("all/train/"+row)
                image_fps.append("all/train/"+row)
                
        for rle in rles:
            mask = masks_as_image([rle])
            image_annotations["all/train/"+row].append(mask)
        
    return image_fps, image_annotations 


class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'Airbus'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8 
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768
    
    RPN_ANCHOR_SCALES = (32, 64)
    
    TRAIN_ROIS_PER_IMAGE = 16
    
    MAX_GT_INSTANCES = 3
    
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.1
    
    RPN_TRAIN_ANCHORS_PER_IMAGE = 16
    STEPS_PER_EPOCH = 100 
    TOP_DOWN_PYRAMID_SIZE = 32
    STEPS_PER_EPOCH = 100
    
    
config = DetectorConfig()
config.display()


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('airbus', 1, 'ship')
   
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('airbus', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        image = cv2.imread(fp)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
        class_ids = np.zeros((count,), dtype=np.int32)
        for i, a in enumerate(annotations):
            mask_instance = mask[:, :, i].copy()
            mask[:, :, i] = mask_instance
            class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)




image_fps, image_annotations = parse_dataset("train_ship_segmentations.csv")
ORIG_SIZE = 768
image_fps_train, image_fps_val = train_test_split(shuffle(image_fps), test_size=0.1, random_state=42)


# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()


# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()


#### MODEL
model = modellib.MaskRCNN(mode='training', 
                          config=config, 
                          model_dir=ROOT_MODEL)


# Image augmentation 
augmentation = iaa.Sequential([
    iaa.Sometimes(0.50, iaa.Fliplr(0.5)),
    iaa.Sometimes(0.50, iaa.Flipud(0.5)),
    iaa.Sometimes(0.30, iaa.CoarseSalt(p=0.10, size_percent=0.02)),
    iaa.Sometimes(0.30, iaa.Affine(rotate=(-25, 25))),
    iaa.Sometimes(0.30, iaa.GaussianBlur((0, 3.0)))
])

NUM_EPOCHS = 1
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=NUM_EPOCHS, 
            layers='all',
            augmentation=augmentation)
