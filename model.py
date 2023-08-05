import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , MaxPooling2D
from keras.applications.vgg19 import VGG19
from keras.applications.resnet_v2 import ResNet101V2

def build_VGG19(IMG_SIZE):
    pre_trained_model = VGG19(input_shape= (IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")

    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False
        

    model = Sequential([
    pre_trained_model,
    MaxPool2D((2,2) , strides = 2),
    Flatten(),
    Dense(5 , activation='softmax')])
    
    return model

def build_ResNet101V2(IMG_SIZE):
    pre_train_model = ResNet101V2(input_shape = (IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet" )
    
    model = Sequential([
    pre_train_model,
    MaxPool2D((2,2) , strides = 2),
    Flatten(),
    Dense(5 , activation='softmax')])
    
    return model