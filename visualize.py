import matplotlib.pyplot as plt
import os
import get_data
import tensorflow as tf
import pandas as pd
import seaborn as sns
import get_data
import numpy as np

data_dir = 'dataset/input/insects'

def plot_graph(folder_names, file_counts):
    plt.bar(folder_names, file_counts, color='blue')
    plt.xlabel('Folders')
    plt.ylabel('Number of Files')
    plt.title('File Count in Folders')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
# show number of image of each class
def show_data_distribution(data_dir, labels):
    folder_path = data_dir
    folders = labels
    folder_names = []
    file_counts = []
    
    for folder in folders:
        full_path = os.path.join(folder_path, folder)
        if os.path.isdir(full_path):
            folder_names.append(folder)
            file_count = get_data.count_files_in_folder(full_path)
            file_counts.append(file_count)
  
    plot_graph(folder_names, file_counts)
    
# show number of image for train, valid and test    
def show_train_test_distribution_3_parts(image_train, image_val, image_test):
    plt.bar(["Train","Valid","Test"],
    [len(image_train), len(image_val), len(image_test)], align='center',color=[ 'green','red', 'blue'])
    plt.legend()

    plt.ylabel('Number of images')
    plt.title('Data distribution')


    plt.show()
    
def show_train_test_distribution_2_parts(image_train, image_test):
    plt.bar(["Train","Test"],
    [len(image_train), len(image_test)], align='center',color=['blue','red'])
    plt.legend()

    plt.ylabel('Number of images')
    plt.title('Data distribution')


    plt.show()
    
def show_train_result(model_dir, log_dir):
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
    
    
def plot_bar_x(classes, prob):
        # this is for plotting purpose
        index = np.arange(len(classes))
        plt.bar(index, prob)
        plt.xlabel('Labels', fontsize=8)
        plt.ylabel('Probability', fontsize=8)
        plt.xticks(index, classes, fontsize=8, rotation=20)
        plt.title('Probability for loaded image')
        plt.show()
        
def confusion_matrix(array):
    df_cm = pd.DataFrame(array, range(5), range(5))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    xticks = get_data.get_label(data_dir)
    yticks = get_data.get_label(data_dir)
    ax = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},
                   xticklabels=xticks,
                   yticklabels=yticks) # font size
    plt.xlabel('Predicted Classes')  # label title for x coord
    plt.ylabel('True Classes')  # label title for y coord
    plt.title('Confusion Matrix')

    plt.show()

    