import preprocessing as pre
from tqdm import tqdm
import get_data
import synchronize as sync
from time import sleep


    

def preprocess_progress(data_dir, saved_data_dir, labels):
    data_len = get_data.get_data(data_dir, labels)[1]
    for i in tqdm(range(0, data_len), desc ="Preprocessed Image: "):
        pre.save_preprocessed_data(data_dir, saved_data_dir, labels)
        sleep(.1)    
        
    print('Preprocess and rotation data has been completed')
    

def synchronize_progress(data_dir, labels):
    data_len = get_data.get_data(data_dir, labels)[1]
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