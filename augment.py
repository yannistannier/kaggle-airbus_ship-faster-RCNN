import imgaug as ia
from imgaug import augmenters as iaa
import cv2

img = cv2.imread("all/train/resize/350/00021ddc3.jpg")

seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])

images_aug = seq.augment_images([img])