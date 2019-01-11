import numpy as np
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D,Activation,Dropout
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization

def get_unet_resnet_dropout():
    
    base_model = ResNet50(weights='imagenet', input_shape=(256,256,3), include_top=False)
    base_out = base_model.output
    
    conv_prop1 = base_model.get_layer('activation_40').output # 16 16 1024
    conv_prop2 = base_model.get_layer('activation_22').output # 32 32 512
    conv_prop3 = base_model.get_layer('activation_4').output # 64 64 256
    conv_prop4 = base_model.get_layer('activation_2').output # 64 64 64
    conv_prop5 = base_model.get_layer('activation_1').output # 128 128 64
    conv_prop6 = base_model.get_layer('input_1').output # 256 256 3
    
    # ARCHITECTURE
    activation = 'elu'
    conv_size = 3
    drop_coef = 0.5

    up1 = UpSampling2D(2, interpolation='bilinear')(base_out) # 16 16
    up1 = concatenate([conv_prop1, up1])
    conv1 = Convolution2D(3072, (conv_size, conv_size), padding='same')(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)
    conv1 = Dropout(drop_coef)(conv1)
    conv1 = Convolution2D(3072, (conv_size, conv_size), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)
    conv1 = Dropout(drop_coef)(conv1)

    up2 = UpSampling2D(2, interpolation='bilinear')(conv1) # 32 32 
    up2 = concatenate([conv_prop2, up2])
    conv2 = Convolution2D(2048, (conv_size, conv_size), padding='same')(up2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)
    conv2 = Dropout(drop_coef)(conv2)
    conv2 = Convolution2D(2048, (conv_size, conv_size), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)
    conv2 = Dropout(drop_coef)(conv2)

    up3 = UpSampling2D(2, interpolation='bilinear')(conv2) # 64 64
    up3 = concatenate([conv_prop3, up3])
    conv3 = Convolution2D(1024, (conv_size, conv_size), padding='same')(up3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)
    conv3 = Dropout(drop_coef)(conv3)
    conv3 = concatenate([conv_prop4, conv3])
    conv3 = Convolution2D(1024, (conv_size, conv_size), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)
    conv3 = Dropout(drop_coef)(conv3)

    up4 = UpSampling2D(2, interpolation='bilinear')(conv3) # 128 128
    up4 = concatenate([conv_prop5, up4])
    conv4 = Convolution2D(512, (conv_size, conv_size), padding='same')(up4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)
    conv4 = Dropout(drop_coef)(conv4)
    conv4 = Convolution2D(512, (conv_size, conv_size), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)
    conv4 = Dropout(drop_coef)(conv4)

    up5 = UpSampling2D(2, interpolation='bilinear')(conv4) # 256 256
    up5 = concatenate([conv_prop6, up5])
    conv5 = Convolution2D(256, (conv_size, conv_size), padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)
    conv5 = Dropout(drop_coef)(conv5)
    conv5 = Convolution2D(128, (1, 1), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)
    conv5 = Dropout(drop_coef)(conv5)
    conv5 = Convolution2D(1, (1, 1))(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('sigmoid')(conv5)

    output = conv5
    model = Model(input=base_model.input, output=output)
    
    return model

def set_best_unet_resnet_dropout_weights(model):
    model.load_weights('weights/fcn_best_resnet_dropout.h5')
    
def set_last_unet_resnet_dropout_weights(model):
    model.load_weights('weights/fcn_last_resnet_dropout.h5')
