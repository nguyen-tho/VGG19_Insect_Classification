import os
import sys
#to synchornize file name based on each labels of the dataset
    
def synchronize_data(root_dir, labels):
    for label in labels:
        path = root_dir
        counter = 1
        path = path +str(label)+'/'
        print(label)
        for filename in os.listdir(label):
            file_ext = os.path.splitext(filename)[1]
            oldFileName = os.path.join(label, filename)
            newFileName = os.path.join(label, label+'_'+str(counter)+file_ext)
            os.rename(oldFileName, newFileName)
            counter = counter + 1
            print('Rename file: '+ newFileName) 
