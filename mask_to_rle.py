import os
import sys
import csv
import time

import imageio
import numpy as np
import pandas as pd


def rle_to_mask(data, shape):
    lines = []
    for line in data:
        line = list(map(int, line.split(',')[1].split(" ")))
        line = list(zip(line[::2], line[1::2]))
        lines.append(np.array(line))
    mask_rle = np.vstack(lines)
    
    mask = np.zeros(shape)
    mask_rle = np.hstack((np.array(divmod(mask_rle[:,0], shape[1])).T, mask_rle[:,1:]))
    for y, x, l in mask_rle:
        mask[y, x:x+l] += 1
    return mask


def mask_to_rle(mask, is_png=False):
    if is_png:
        mask = imageio.imread(mask)
    rle = []
    counter = 0
    index = 0
    for irow, row in enumerate(mask):
        for icol, p in enumerate(row):
            if p:
                if counter == 0:
                    index = irow * len(row) + icol
                counter += 1
            else:
                if counter:
                    rle.append(("{} {}".format(index, counter)))
                    counter = 0
    rle = " ".join(rle)
    return rle


if __name__ == "__main__":
    #for rle_to_mask
    PATH = 'DATA/TRAIN'
    masks = []
    for directory in os.listdir(PATH):
        path_to_dir = os.path.join(PATH, directory)
        path_to_data = [i for i in os.listdir(path_to_dir) if i.endswith(".csv")][0]
        shape = imageio.imread(os.path.join(path_to_dir, '{}_mask.png'.format(directory))).shape
        rle_data = open(os.path.join(path_to_dir, path_to_data)).read().splitlines()
        mask = rle_to_mask(rle_data, shape)
        masks.append(mask)
        