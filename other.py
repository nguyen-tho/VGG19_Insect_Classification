import tensorflow as tf
from matplotlib import pyplot as plt
import preprocessing as pre
from tqdm import tqdm
import pandas as pd
import get_data
import synchronize as sync
from time import sleep
import os

def show_graphs(model_dir, log_dir):
    model = tf.keras.models.load_model(model_dir, compile = False)
    history = pd.read_csv(log_dir, sep = ',', engine = 'python')
    hist = history
    
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
    

def preprocess_progress(data_dir, saved_data_dir, labels, size):
    data_len = get_data.get_data_len(get_data.get_data(data_dir, labels))
    for i in tqdm(range(0, data_len), desc ="Preprocessed Image: "):
        pre.save_preprocessed_data(data_dir, saved_data_dir, labels, size)
        sleep(.1)    
        
    print('Preprocess and rotation data has been completed')
    

def synchronize_progress(data_dir, labels):
    data_len = get_data.get_data_len(get_data.get_data(data_dir, labels))
    for i in tqdm(range(0, data_len), desc ="Synchronized Image: "):
        sync.synchronize_data(data_dir, labels)
        sleep(.1)
        
    print('Data has been synchronized')
    
'''
def rotate_progress(data_dir, labels):
    data_len = get_data.get_data_len(get_data.get_data(data_dir, labels))
    for i in tqdm(range(0, data_len), desc ="Rotated Image: "):
        for label in labels:
            path = data_dir+'/'+label
            for img in os.listdir(path):
                pre.rotate_image(path+'/'+img, 90)
                pre.rotate_image(path+'/'+img, 180)
                pre.rotate_image(path+'/'+img, 270)
        sleep(.1)
        
    print('Rotate progress has been completed')
'''