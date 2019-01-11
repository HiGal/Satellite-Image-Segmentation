import json
import os

import numpy as np
import scipy.misc
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D,Activation,Dropout
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from PIL import Image
import imageio
import matplotlib.pyplot as plt

from models import get_unet_resnet_dropout, set_best_unet_resnet_dropout_weights


tiles_dict = dict()
block_size_x = 256
block_size_y = 256
i = 0


def break_grid(img, x, y, shift_x, shift_y, output_folder):
    print('Original size:', (x, y))

    x_num = (x // block_size_x)
    y_num = (y // block_size_y)
    print('Num of images:', x_num * y_num)

    x_overflow = x % block_size_x
    y_iverflow = y % block_size_y
    print('Sides: x {}; y {}'.format(x_overflow, y_iverflow))

    break_grid_helper(img, x_num, y_num, x, y, shift_x, shift_y, output_folder)


def break_grid_helper(img, x_num, y_num, x, y, shift_x, shift_y, output_folder):
    global i

    for y_iter in range(y_num):
        for x_iter in range(x_num):
            i += 1
            curr_x = block_size_x * x_iter
            curr_y = block_size_y * y_iter
            end_curr_x = curr_x + block_size_x
            end_curr_y = curr_y + block_size_y

            test = img.crop((curr_x, curr_y, end_curr_x, end_curr_y))
            tile_name = '{}.png'.format(i)
            tiles_dict['images'][tile_name] = {'x1': curr_x + shift_x, 'y1': curr_y + shift_y,
                                               'x2': end_curr_x + shift_x, 'y2': end_curr_y + shift_y}
            test.save(os.path.join(output_folder, tile_name))

    if x_num * block_size_x < x:
        for y_iter in range(y_num):
            i += 1

            curr_y = block_size_y * y_iter
            end_curr_y = curr_y + block_size_y
            curr_x = x - block_size_x
            end_curr_x = x

            test = img.crop((curr_x, curr_y, end_curr_x, end_curr_y))
            tile_name = '{}.png'.format(i)
            tiles_dict['images'][tile_name] = {'x1': curr_x + shift_x, 'y1': curr_y + shift_y,
                                               'x2': end_curr_x + shift_x, 'y2': end_curr_y + shift_y}
            test.save(os.path.join(output_folder, tile_name))

    if y_num * block_size_y < y:
        for x_iter in range(x_num):
            i += 1

            curr_x = block_size_x * x_iter
            end_curr_x = curr_x + block_size_x
            curr_y = y - block_size_y
            end_curr_y = y

            test = img.crop((curr_x, curr_y, end_curr_x, end_curr_y))
            tile_name = '{}.png'.format(i)
            tiles_dict['images'][tile_name] = {'x1': curr_x + shift_x, 'y1': curr_y + shift_y,
                                               'x2': end_curr_x + shift_x, 'y2': end_curr_y + shift_y}
            test.save(os.path.join(output_folder, tile_name))

        if x_num * block_size_x < x:
            i += 1
            curr_x = x - block_size_x
            end_curr_x = x
            curr_y = y - block_size_y
            end_curr_y = y

            test = img.crop((curr_x, curr_y, end_curr_x, end_curr_y))
            tile_name = '{}.png'.format(i)
            tiles_dict['images'][tile_name] = {'x1': curr_x + shift_x, 'y1': curr_y + shift_y,
                                               'x2': end_curr_x + shift_x, 'y2': end_curr_y + shift_y}
            test.save(os.path.join(output_folder, tile_name))


def break_image(input_file_path, output_folder, fast_grid):
    tiles_dict['images'] = dict()

    original_img = Image.open(input_file_path)
    orig_x, orig_y = original_img.size
    shift_x = 0
    shift_y = 0
    break_grid(original_img, orig_x, orig_y, shift_x, shift_y, output_folder=output_folder)

    if not fast_grid:
        original_img_1 = original_img.copy()
        x_cropped_img = original_img_1.crop((block_size_x // 2, 0, orig_x - block_size_x // 2, orig_y))
        x, y = x_cropped_img.size
        shift_x = block_size_x // 2
        shift_y = 0
        break_grid(x_cropped_img, x, y, shift_x, shift_y, output_folder=output_folder)

        original_img_2 = original_img_1.copy()
        y_cropped_img = original_img_2.crop((0, block_size_y // 2, orig_x, orig_y - block_size_y // 2))
        x, y = y_cropped_img.size
        shift_x = 0
        shift_y = block_size_y // 2
        break_grid(y_cropped_img, x, y, shift_x, shift_y, output_folder=output_folder)

        original_img_3 = original_img.copy()
        x_y_cropped_img = original_img_3.crop(
            (block_size_x // 2, block_size_y // 2, orig_x - block_size_x // 2, orig_y - block_size_y // 2))
        x, y = x_y_cropped_img.size
        shift_x = block_size_x // 2
        shift_y = block_size_y // 2
        break_grid(x_y_cropped_img, x, y, shift_x, shift_y, output_folder=output_folder)

    tiles_dict['dimensions'] = [orig_x, orig_y]

    with open(os.path.join(output_folder, 'output_data.json'), 'w') as file:
        file.write(json.dumps(tiles_dict))


def combine_image(folder_path, model):
    j = 0
    
    with open(os.path.join(folder_path, 'output_data.json'), 'r') as json_file:
        parsed_json = json.loads(json_file.read())

        result = np.zeros((parsed_json['dimensions'][1], parsed_json['dimensions'][0]), dtype=np.uint8)

        for file_name, img_data in parsed_json['images'].items():
            j += 1
            path = os.path.join(folder_path, file_name)
            img = np.array(imageio.imread(path))
            img = np.array([img])
            img = np.true_divide(img, 255)
            print('Shape:', img.shape)
            img = model.predict(img).reshape(256,256)
            img = img > 0.2
          
            print('Predicted:', img.shape)


            temp = img.copy()*255//2
            print("File:", file_name, "Shape:", temp.shape, result.shape)
            for k in range(img_data['y2'] - img_data['y1']):
                for o in range(img_data['x2'] - img_data['x1']):
                    result[img_data['y1'] + k][img_data['x1'] + o] //= 2
                    result[img_data['y1'] + k][img_data['x1'] + o] += temp[k][o]
        
        plt.imshow(result.reshape((result.shape[0], result.shape[1])), cmap='gray')
        plt.show()
        imageio.imwrite(os.path.join(folder_path, 'Pre_Result.png'), result)
        return result


def clear_image(image, folder_path):
    result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 126:
                result[i][j] = 255

    imageio.imwrite(os.path.join(folder_path, 'Result.png'), result)
    



