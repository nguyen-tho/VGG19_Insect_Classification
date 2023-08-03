import os
import sys
#to synchornize file name based on each labels of the dataset
    
def synchronize_data(root_dir, labels):
    for label in labels:
        counter = 1
        path = root_dir +'/'+str(label)
        print(label)
        for filename in os.listdir(path):
            file_ext = os.path.splitext(filename)[1]
            oldFileName = os.path.join(path, filename)
            newFileName = os.path.join(path, label+'_'+str(counter)+file_ext)
            os.rename(oldFileName, newFileName)
            counter = counter + 1
            print('Rename file: '+ newFileName) 

