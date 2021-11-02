from ctypes import Array
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import easyocr
import array as arr

# function takes image as input and returns the license number of the vehicle
# Automatic Number Plate Recognization
def ANPR(image): 
    real_image = cv2.imread(image)  #Importing the image as it is
    img = cv2.imread(image,0) #Importing the image in GRAY format
    # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    # bFilter = cv2.bilateralFilter(img,11,17,17) #Noise removal
    bFilter = cv2.bilateralFilter(img, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bFilter,30,300) #Edge Detection
    # plt.imshow(cv2.cvtColor(edged,cv2.COLOR_BGR2RGB))

    keypoints = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Finding all Contours in image
    contours = imutils.grab_contours(keypoints) # Storing it in contours variable
    contours = sorted(contours,key=cv2.contourArea, reverse=True)[:10] # Taking first 10 contours from sorted contours in the form of area

    location = None # Declaring a variable to store the locaton of number plate
    # Iterating over the finded 10 contours and finding a bounded rectangle and storing in location variable
    for contour in contours:  
       approx=cv2.approxPolyDP(contour,10,True) # To approximate the shape of contour
       if len(approx)==4:
          location=approx
          break

    # print(location)

    mask = np.zeros(img.shape, np.uint8) # Creating a array for making a blank mask
    new_image = cv2.drawContours(mask, [location],0,255,-1) # Drawing the finded countour on the image
    new_image = cv2.bitwise_and(real_image,real_image,mask=mask) # Overlay mask on real_image
    # plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))

    (x,y) = np.where(mask==255) # (x,y) will store all the coordinates of boundary of number plate
    (x1, y1) = (np.min(x), np.min(y)) # Determined the top left corner of number plate
    (x2, y2) = (np.max(x), np.max(y)) # Determined the bottom right corner of number plate
    cropped_image = img[x1:x2+1, y1:y2+1] # Stored the cropped number plate image
    # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    reader = easyocr.Reader(['en']) # Creating instance for easyocr reader method
    result = reader.readtext(cropped_image) # Reading the text from the image

    text = result[0][-2] # Retriving the Vehicle number from result
    # print(text)
    return text # Returning the license number of vehicle
  
# Defining the array of images location
img_arr = np.array(['image2.jpg','image3.jpg','image4.jpg','image5.JPG'])
# An empty array to store result of ANPR function
res_arr = np.array([]) 

# Iterating thorugh all images and storing result of ANPR function
for i in img_arr:
    number_plate = ANPR(i)
    state=number_plate[0:2]
    res_arr = np.append(res_arr,state)

# print(res_arr)

# Array for data of each state vehicles
all_state = {
    'MH':0,
    'RJ':0,
    'GJ':0,
    'other':0
    } 

# Iterating through res_arr and updating all_state dictonary
for state in res_arr:
    if(state!='MH' and state!='RJ' and state!='GJ'):
        all_state['other'] = all_state['other']+1
    else:
        all_state[state] = all_state[state]+1

# print(all_state)

# Retriving the state_names from dictonary
state_names = list(all_state.keys())
# Retriving the values for each state from dictonary
values = list(all_state.values())

# Plotting results using Matplotlib
plt.bar(state_names,values)
plt.show()