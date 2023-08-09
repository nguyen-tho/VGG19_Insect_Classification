import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , Concatenate
from keras.applications.vgg19 import VGG19
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications import ResNet50


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

def build_RetinaNet(IMG_SIZE):
    input_tensor = keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    feature_extractor = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)


# Define the classification sub-network
    classification = Conv2D(filters=9, kernel_size=(3,3), activation='relu')(feature_extractor.output)
    #classification = Flatten()(classification)
    #classification = Dense(units=9, activation='sigmoid')(classification) # Remove the regression subnetwork


# Define the regression sub-network
    regression = Conv2D(filters=36, kernel_size=(3,3), activation='relu')(feature_extractor.output)
    #regression = Flatten()(regression) 
    #regression = Dense(units=4)(regression)  # Remove the classification subnetwork
    
    # Concatenate the outputs from the classification and regression sub-networks
    concat = Concatenate()([classification, regression])
   
    # retinanet for classification
     
    pooling = MaxPool2D((2,2) , strides = 2)(concat)
    flatten = Flatten()(pooling)
    output = Dense(5 , activation='softmax')(flatten) 
    
    # Define the final model
    model = Model(inputs=input_tensor, outputs=output)
    
    return model

def show_graphs(model_dir, log_dir):
    model = tf.keras.models.load_model(model_dir, compile = False)
    history = pd.read_csv(log_dir, sep = ',', engine = 'python')
    acc=hist['accuracy']
    val_acc=hist['val_accuracy']

    epoch=range(len(acc))

    loss=hist['loss']
    val_loss=hist['val_loss']

    f,ax=plt.subplots(2,1,figsize=(20,20))
    ax[0].set_title('Model Accuracy')
    ax[0].plot(epoch,acc,'b',label='Training Accuracy')
    ax[0].plot(epoch,val_acc,'r',label='Validation Accuracy')
    ax[0].legend()

    ax[1].set_title('Model Loss')
    ax[1].plot(epoch,loss,'b',label='Training Loss')
    ax[1].plot(epoch,val_loss,'r',label='Validation Loss')
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    
