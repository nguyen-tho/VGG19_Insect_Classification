import os
import random
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

torch_image_transform = transforms.ToTensor()


def get_data(dir, labels):
    data = []
    for label in labels:
        path = os.path.join(dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
          try:
            img = cv2.imread(os.path.join(path, img))
            torch_img = torch_image_transform(img)
            torch_img_numpy = torch_img.cpu().numpy().transpose([1, 2, 0])
            data.append([torch_img_numpy, class_num])
          except Exception as e:
            print(e)
    return data
  
  
def get_random_file(directory_path):
    # Get the list of files in the directory
    files = os.listdir(directory_path)

    # Optional: Filter out directories from the list (if needed)
    files = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]

    # Check if there are any files in the directory
    if not files:
        print("No files found in the directory.")
        return None

    # Generate a random index within the range of the files list
    random_index = random.randint(0, len(files) - 1)

    # Get the randomly chosen file from the list
    random_file = files[random_index]

    # Return the full path to the randomly chosen file
    return os.path.join(directory_path, random_file)

def get_data_len(data):
  return len(data)

def get_labels(data_path):
  #get children folders of root data directory
  labels = os.listdir(data_path)
  return labels

def count_files_in_folder(folder_path):
    file_count = 0
    for item in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, item)):
            file_count += 1
    return file_count

