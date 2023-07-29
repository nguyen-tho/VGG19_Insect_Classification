import os
import sys
#to synchornize file name based on each labels of the dataset
root = os.chdir('../dataset/input/insects/')
labels = os.listdir(root)
print(root)

for label in labels:
    path = '../dataset/input/insects/'
    counter = 1
    path = path +str(label)+'/'
    print(label)
    for filename in os.listdir(label):
       
        file_ext = os.path.splitext(filename)[1]
        oldFileName = os.path.join(label, filename)
        newFileName = os.path.join(label, label+'_'+str(counter)+file_ext)
        os.rename(oldFileName, newFileName)
        counter = counter + 1
        print(newFileName +' is renaming....') 
    

