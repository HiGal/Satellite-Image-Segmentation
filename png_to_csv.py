# Run-Length Encode and Decode

import os
import sys
import csv
import time

import imageio
import numpy as np
import pandas as pd

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def write_csv(path, filename, data):
    with open(os.path.join(path, filename.replace('.png', '.csv')), 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['ImageId', 'EncodedPixels'])
        writer.writerow([filename, data])
        file.close()

def convert_mask(path, filename):
    img = imageio.imread(os.path.join(path, filename))
    rle = rle_encode(img)
    write_csv(path, filename, rle)