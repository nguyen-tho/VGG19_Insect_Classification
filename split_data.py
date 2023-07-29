import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def pathListIntoId(path_list, class_mapping):
    """
    Convert a list of image paths into corresponding class IDs.

    Parameters:
        path_list (list): List of image paths.
        class_mapping (dict): A dictionary that maps class names to class IDs.

    Returns:
        list: A list of class IDs corresponding to the input image paths.
    """
    class_ids = []
    for image_path in path_list:
        class_name = os.path.basename(os.path.dirname(image_path))
        class_id = class_mapping.get(class_name)
        if class_id is not None:
            class_ids.append(class_id)
    return class_ids

  '''   

def split_data(train_ratio, val_ratio, test_ratio):
    
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))

    class_ids = pathListIntoId(image_paths, class_mapping)
    image_paths = np.array(image_paths)
    class_ids = np.array(class_ids)

# Split the data into train, validation, and test sets while maintaining class distribution
    train_ratio = train_ratio
    validation_ratio = val_ratio
    test_ratio = test_ratio

# First, split the data into train and test sets
    image_paths_train, image_paths_test, class_ids_train, class_ids_test = train_test_split(
    image_paths, class_ids, test_size=test_ratio, random_state=42, stratify=class_ids)

# Then, split the remaining data (train set) into train and validation sets
    remaining_ratio = validation_ratio / (train_ratio + validation_ratio)
    image_paths_train, image_paths_val, class_ids_train, class_ids_val = train_test_split(
    image_paths_train, class_ids_train, test_size=remaining_ratio, random_state=42, stratify=class_ids_train)

# Now you have four arrays:
# - image_paths_train: Array of paths for the training set images
# - image_paths_val: Array of paths for the validation set images
# - image_paths_test: Array of paths for the test set images
# - class_ids_train: Array of class IDs corresponding to the training set images
# - class_ids_val: Array of class IDs corresponding to the validation set images
# - class_ids_test: Array of class IDs corresponding to the test set images
    







    return unique_classes, class_counts

#for example
#train ratio 0.6, validation ratio = 0.2, test ratio = 0.2
#in this case, train_val_ratio = 0.2 while train_test_ratio = 0.25
#explaination:
#train+test/val = 0.8/0.2 => val = 0.2, train+test=0.8
#train/test=0.75/0.25 *(train+test) => train = 0.75*(train+test), test = 0.25*(train+test)
#=> train = 0.6, val = 0.2, test = 0.2

unique_classes, class_counts = split_data(0.6, 0.2, 0.2)

def showDataLayout(unique_classes, class_counts):
    plt.figure(figsize=(10, 6))
    plt.bar(unique_classes, class_counts)
    plt.xticks(unique_classes)
    plt.xlabel("Class ID")
    plt.ylabel("Number of Samples")
    plt.title("Data Distribution for Insect Classification")
    plt.show()
    
showDataLayout(unique_classes, class_counts)  
'''