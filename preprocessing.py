import os
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import io
import random

# Tranform OpenCV to PIL
def OpenCV_to_PIL_img(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image color chanel for transforming to PIL image
  pil_img = Image.fromarray(img) # PIL transform
  return pil_img

'''
def crop_img(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image color channel for transforming to PIL image
  pil_img = Image.fromarray(img) # PIL transform
  min_edge = min(pil_img.size)
  cropped_img = pil_img.crop((0, 0, min_edge, min_edge))
  return cropped_img
  '''
'''
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
'''
def get_upscaled_img(img):
  y, x = img.shape[:2]
  sr = cv2.dnn_superres.DnnSuperResImpl.create()
  path = "EDSR_x4.pb"
  sr.readModel(path)
  sr.setModel("edsr",1)
  return sr.upsample(img)
'''
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
  cropped_img = crop_img(img)

  # Coloring expanded part of the image
  dc = get_environment_color(cropped_img, int(min*1/100)) # Get colors at 1% pixels-of-min-length border of the image
  colored_img = coloring_img(cropped_img, width, height, n, dc)

  # Change into OpenCV image
  colored_img = np.asarray(colored_img) # The original image chanel is RGB now
  enhanced_img = get_upscaled_img(colored_img)

  return enhanced_img
'''
'''
def crop_img(input_image):
  
  # Get the dimensions of the input image
  height, width = input_image.shape[:2]

# Determine the size of the square you want
   # Get the dimensions of the image
   
    # Calculate the aspect ratio
  aspect_ratio = width / height
  average = (width+height)/2
    
    # Determine the cropping dimensions
  if aspect_ratio >= 1:
      new_width = width
      new_height = int(new_width / aspect_ratio)
  else:
      new_height = height
      new_width = int(new_height * aspect_ratio)
    
    # Calculate the cropping coordinates
  crop_x = (width - new_width) // 2
  crop_y = (height - new_height) // 2
    
    # Crop the image
  cropped_image = input_image[crop_y:crop_y + new_height, crop_x:crop_x + new_width]
    
  return cropped_image

'''
def resize_to_average(original_image):
    
    # Get the dimensions of the original image
    original_height, original_width = original_image.shape[:2]
    
    # Calculate the average size
    average_size = (original_width + original_height) // 2
    
    # Create a new square image with the average size
    square_image = cv2.resize(original_image, (average_size, average_size))
    
    # Save the resulting square image
    return square_image
  
def resize_image(image, size):
  new_size = (size, size)
  return cv2.resize(image, new_size, interpolation= cv2.INTER_CUBIC)

def enhance_image(input_image):
  upscaled_img = get_upscaled_img(input_image)
  sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
  sharpened_image = cv2.filter2D(upscaled_img, -1, sharpening_kernel)
  return sharpened_image

def preprocessing_img(img, size):
  #change image into square (n x n) image
  image = resize_to_average(img) 
  # sharpening image
  enhanced_image = enhance_image(image)
  # resize image into standard size
  image_resized = resize_image(enhanced_image, size)
  return image_resized
'''
def preprocessing_image(image, IMG_SIZE): #return CV2 image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Change to PIL format
    pil_img = Image.fromarray(image)

    #Get the shortest edge length
    min_edge_length = min(pil_img.size[0], pil_img.size[1])
    max_edge_length = max(pil_img.size[0], pil_img.size[1])

    if min_edge_length == pil_img.size[0]:
        shortest_edge = "width"
    else:
        shortest_edge = "height"

    #Crop the image to n x n pixels with n = max_edge_length
    cropped_img = pil_img.crop((0, 0, max_edge_length, max_edge_length))

    #Transform the image by translating the image to center
    translate_length = int((max_edge_length - min_edge_length)/2)

    if shortest_edge == "width":
        translated_img = cropped_img.transform(cropped_img.size, Image.AFFINE, (1, 0, -translate_length, 0, 1, 0))
    else:
        translated_img = cropped_img.transform(cropped_img.size, Image.AFFINE, (1, 0, 0, 0, 1, -translate_length))

    #Grayscale image
    width, height = translated_img.size
    gray_img = Image.new('1', (width, height)) #blank grayscale image with same width and height

    #Reducing RBG values for grayscale image
    divide_number = random.randint(75000, 100000)

    for x in range(width):
        for y in range(height):
            r, g, b = translated_img.getpixel((x, y))
            multiply_g = 587.0/divide_number
            multiply_r = 299.0/divide_number
            multiply_b = 114.0/divide_number

            value = r * multiply_r + g * multiply_g + b * multiply_b
            value = int(value)
            gray_img.putpixel((x, y), value)

    RBG_img = Image.new("RGB", (width, height))
    RBG_img.paste(gray_img)
    gray_3chanels_img = RBG_img

    #Resize the image to 224 x 224 pixels fit to the model
    gray_3chanels_img.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    preprocessed_image = np.asarray(gray_3chanels_img)
    return preprocessed_image
'''  
def preprocessing_image(image):
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the shortest and longest edge lengths
    min_edge_length = min(image.shape[0], image.shape[1])
    max_edge_length = max(image.shape[0], image.shape[1])

    # Crop to a square
    cropped_img = gray_img[:max_edge_length, :max_edge_length]
    
    # Translate the image to center
    translate_length = abs(max_edge_length - min_edge_length) // 2
    if min_edge_length == image.shape[0]:
        translated_img = np.pad(cropped_img, ((translate_length, translate_length), (0, 0)), mode='constant')
    else:
        translated_img = np.pad(cropped_img, ((0, 0), (translate_length, translate_length)), mode='constant')

    # Resize the image to 224 x 224 pixels
    resized_img = cv2.resize(translated_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    return resized_img

def save_preprocessed_data(inputs, outputs, labels):
   # data = []
    for label in labels:
        input_path = inputs + '/' + label
        output_path = outputs + '/' + label
        #class_num = labels.index(label)
        for img in os.listdir(input_path):
          filename, file_ext = os.path.splitext(img)
          try:
            
            #if the image file name is exist it will be passed
            saved_path = output_path+'/'+filename+file_ext
            isExist = os.path.exists(saved_path)
            if isExist:
              continue
            else:
              img = cv2.imread(input_path+'/'+img)
              saved_img = preprocessing_image(img)
              saved_img = saved_img.astype(np.uint8)
              cv2.imwrite(saved_path, saved_img)
              rotate_image(saved_path, 90)
              rotate_image(saved_path, 180)
              rotate_image(saved_path, 270)
          except Exception as e:
            print(e)
          
            
def rotate_image(image_path, degree):
  #rotate an image in 3 cases
  #case 1: 90 degree clockwise
  #case 2: 180 degree
  #case 3: 270 degree clockwise or 90 degree counter clockwise
  src = cv2.imread(image_path)
  file_name, file_ext = os.path.splitext(image_path)
  if degree == 90: 
    image = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
  elif degree==180:
    image = cv2.rotate(src, cv2.ROTATE_180)
  else: 
    image = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
  #save image
  saved_path = file_name+'_'+str(degree)+file_ext
  cv2.imwrite(saved_path, image)   


