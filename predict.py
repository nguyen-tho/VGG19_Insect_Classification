from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import seaborn as sns
import pandas as pd

def predict(img_rel_path, model):
    # Import Image from the path with size of (300, 300)
    img = image.load_img(img_rel_path, target_size=(224, 224))

    # Convert Image to a numpy array
    img = image.img_to_array(img, dtype=np.uint8)

    # Scaling the Image Array values between 0 and 1
    img = np.array(img)/255.0

    # Plotting the Loaded Image
    plt.title("Loaded Image")
    plt.axis('off')
    plt.imshow(img.squeeze())
    plt.show()

    # Get the Predicted Label for the loaded Image
    p = model.predict(img[np.newaxis, ...])

    # Label array
    labels = {0: 'Butterfly', 1: 'Dragonfly', 2: 'Grasshopper', 3: 'Ladybird', 4: 'Mosquito'}

    print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
    predicted_class = labels[np.argmax(p[0], axis=-1)]
    print("Classified:", predicted_class, "\n\n")

    classes=[]
    prob=[]
    print("\n-------------------Individual Probability--------------------------------\n")

    for i,j in enumerate (p[0],0):
        print(labels[i].upper(),':',round(j*100,2),'%')
        classes.append(labels[i])
        prob.append(round(j*100,2))

    def plot_bar_x():
        # this is for plotting purpose
        index = np.arange(len(classes))
        plt.bar(index, prob)
        plt.xlabel('Labels', fontsize=8)
        plt.ylabel('Probability', fontsize=8)
        plt.xticks(index, classes, fontsize=8, rotation=20)
        plt.title('Probability for loaded image')
        plt.show()
    plot_bar_x()
    return p

def confusion_matrix(array):

    df_cm = pd.DataFrame(array, range(5), range(5))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    xticks = ['Butterfly','Dragonfly','Grasshopper', 'Ladybird', 'Mosquito']
    yticks = ['Butterfly','Dragonfly','Grasshopper', 'Ladybird', 'Mosquito']
    ax = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},
                   xticklabels=xticks,
                   yticklabels=yticks) # font size
    plt.xlabel('Predicted Classes')  # label title for x coord
    plt.ylabel('True Classes')  # label title for y coord
    plt.title('Confusion Matrix')

    plt.show()

