import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import preprocessing as pre
IMG_SIZE = 224
class DataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=32, image_size=(IMG_SIZE, IMG_SIZE), num_classes=10, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.image_paths, self.labels = self._get_image_paths_and_labels()
        self.on_epoch_end()

    def _get_image_paths_and_labels(self):
        image_paths = []
        labels = []
        class_dirs = os.listdir(self.data_dir)
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.data_dir, class_dir)
            for image_file in os.listdir(class_path):
                image_paths.append(os.path.join(class_path, image_file))
                labels.append(class_idx)
        return np.array(image_paths), np.array(labels)

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size: (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size: (index + 1) * self.batch_size]
        images, labels = self._load_and_preprocess_images(batch_image_paths, batch_labels)
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = self.image_paths[indices]
            self.labels = self.labels[indices]

    def _load_and_preprocess_images(self, image_paths, labels):
        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            #image = cv2.resize(image, self.image_size)
            pre.preprocessing_img(image)
            image = image.astype(np.float32) / 255.0
            images.append(image)
        images = np.array(images)
        labels = to_categorical(labels, num_classes=self.num_classes)
        return images, labels
