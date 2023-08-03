import os
import cv2
from PIL import Image
import numpy as np
# Tranform OpenCV to PIL
def OpenCV_to_PIL_img(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image color chanel for transforming to PIL image
  pil_img = Image.fromarray(img) # PIL transform
  return pil_img

def crop_img(img, n):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image color channel for transforming to PIL image
  pil_img = Image.fromarray(img) # PIL transform
  cropped_img = pil_img.crop((0, 0, n, n))
  return cropped_img

def get_environment_color(pil_img, num_of_pixels):
    img = pil_img.copy() #image temp

    width, height = img.size

    color_list = []
    # Get left colors
    for w in range(0, num_of_pixels):
      for h in range(0, height):
        pixel_color = img.getpixel((w, h))
        color_list.append(pixel_color)

    # Get below colors
    for h in range(0, num_of_pixels):
      for w in range(0, width):
        pixel_color = img.getpixel((w, h))
        color_list.append(pixel_color)

    # Get right colors
    for w in range((width-num_of_pixels), width):
      for h in range(0, height):
        pixel_color = img.getpixel((w, h))
        color_list.append(pixel_color)

    # Get top colors
    for h in range((height-num_of_pixels), height):
      for w in range(0, width):
        pixel_color = img.getpixel((w, h))
        color_list.append(pixel_color)

    environment_color = max(color_list, key=color_list.count)

    # Checking
    #print("environment color: ", environment_color)

    return environment_color

def coloring_img(pil_img, width, height, n, color):
  if(n > width and n > height):
    for w in range(width, n):
      for h in range(n):
        pil_img.putpixel((w, h), color)
    for w in range(n):
      for h in range(height, n):
        pil_img.putpixel((w, h), color)

  elif(n > width and n == height):
    for w in range(width, n):
      for h in range(n):
        pil_img.putpixel((w, h), color)

  elif(n > height and n == width):
    for w in range(n):
      for h in range(height, n):
        pil_img.putpixel((w, h), color)
  return pil_img

def get_upscaled_img(img):
  y, x = img.shape[:2]
  sr = cv2.dnn_superres.DnnSuperResImpl_create()
  path = "EDSR_x4.pb"
  sr.readModel(path)
  sr.setModel("edsr",4)
  return sr.upsample(img)

def resize_img(img, width, height, n):
  scaleX = n/width
  scaleY = n/height
  return cv2.resize(img,dsize=None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_CUBIC)

def preprocessing_img(img):
  # Get largest edge of the image
  height, width = img.shape[:2]

  if width >= height:
    n = width
    min = height
  else:
    n = height
    min = width

  # Crop to n x n
  cropped_img = crop_img(img, n)

  # Coloring expanded part of the image
  dc = get_environment_color(cropped_img, int(min*1/100)) # Get colors at 1% pixels-of-min-length border of the image
  colored_img = coloring_img(cropped_img, width, height, n, dc)

  # Change into OpenCV image
  colored_img = np.asarray(colored_img) # The original image chanel is RGB now
  enhanced_img = get_upscaled_img(colored_img)

  return enhanced_img

def save_preprocessed_data(inputs, outputs, labels):
   # data = []
    for label in labels:
        input_path = inputs + '/' + label
        output_path = outputs + '/' + label
        #class_num = labels.index(label)
        for img in os.listdir(input_path):
          filename, file_ext = os.path.splitext(img)
          try:
            img = cv2.imread(input_path+'/'+img)
            #if the image file name is exist it will be passed
            saved_path = output_path+'/'+filename+file_ext
            isExist = os.path.exists(saved_path)
            if isExist:
              continue
            else:
              saved_img = preprocessing_img(img)
              saved_img = saved_img.astype(np.uint8)
              cv2.imwrite(saved_path, saved_img)
          except Exception as e:
            print(e)
            
def rotate_image(image_path, degree):
  #rotate an image in 3 cases
  #case 1: 90 degree clockwise
  #case 2: 180 degree
  #case 3: 270 degree clockwise or 90 degree counter clockwise
  src = cv2.imread(image_path)
  if degree == 90: 
    image = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
  elif degree==180:
    image = cv2.rotate(src, cv2.ROTATE_180)
  else: 
    image = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
  #save image
  cv2.imwrite(image_path+'_'+str(degree), image)   
  
  
