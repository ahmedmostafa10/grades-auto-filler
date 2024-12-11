import matplotlib.pyplot as plt
import imutils
from skimage.transform import resize
from skimage.feature import hog
import pytesseract
import numpy as np
import cv2
import os
from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
import random




def hog_fun(img):
    resized_img = resize(img, (128 * 4, 64 * 4))

    # Getting HOG features
    return hog(resized_img, orientations=12, pixels_per_cell=(
        12, 12), cells_per_block=(6, 6), visualize=True)

