import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(image, label, train_ratio, val_ratio, test_ratio):
    training_ratio = train_ratio
    valid_ratio = valid ratio
    testing_ratio = test_ratio

    image_train_val, image_test, label_train_val, label_test = train_test_split(image, label, test_size=test_ratio,  stratify = y , random_state = 0)
    remaining_ratio = valid_ratio / (train_ratio + valid_ratio)
    image_train, image_val, label_train, label_val = train_test_split(image_train_val, label_train_val, test_size=remaining_ratio, stratify=label_train_val, random_state=0)

    return image_train, image_val, image_test, label_train, label_val, label_test


def show_data_distribution(image_train, image_val, image_test):
    plt.bar(["Train","Valid","Test"],
    [len(image_train), len(image_val), len(image_test)], align='center',color=[ 'green','red', 'blue'])
    plt.legend()

    plt.ylabel('Number of images')
    plt.title('Data distribution')

    plt.show()
