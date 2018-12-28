#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:42:48 2018

@author: parkershankin-clarke
"""
#load packages
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
from LRGUI import *
from numpy import linalg
from list_maker import *


def GUI_data():
    '''Takes RGB values from pixels in rectangles chosen by user(less than or equal to four).  This function returns labled pixel as training ddata for the model  '''

    #opens image by importing it from GUI.
    #takes an argument "image" that will be processed by GUI as the reference image for the training set.
    im = Image.open(args["image"])
    rgb_im = im.convert('RGB')

    #Intialize counter variables
    count =0
    countx=0
    county=0

    #Intialize ash/nonash  arrays
    nonash_pixel_array = np.array([])
    nonash_pixel_array2 = np.array([])
    nonash_pixel_array3 = np.array([])
    ash_pixel_array = np.array([])

    ##iterate through and collect RGB values for ash
    #if the user uses two rectanges:
    if len(pixpos) == 2:
        #The user should use the first rectangle to identify the ash in the photograph. All subsquent rectangles are considered nonash
        ash = pixpos[0]
        
        #define the starting and ending points.
        ashs = ash[0]
        ashe = ash[1]
        #break points into their x-components.
        ashxs = ashs[0]
        ashxe = ashe[0]
        #break points into their y-components.
        ashys = ashs[1]
        ashye = ashe[1]

       #puts pixel data into an array
        for x in range(ashxs,ashxe):
            for y in range(ashys,ashye):
                r, g, b = rgb_im.getpixel((x, y))
                ash_pixel_array = np.append(ash_pixel_array,[r,g,b])

       #iterate through and RGB values for non-ash
        nonash = pixpos[1]

        nonashs = nonash[0]
        nonashe = nonash[1]

        nonashxs = nonashs[0]
        nonashxe = nonashe[0]

        nonashys = nonashs[1]
        nonashye = nonashe[1]

       #puts pixel data into an array
        for x in range(nonashxs,nonashxe):
            for y in range(nonashys,nonashye):
                r, g, b = rgb_im.getpixel((x, y))
                nonash_pixel_array = np.append(nonash_pixel_array,[r,g,b])
       
        #Reformat ash pixels GB
        ash_array  = np.delete(ash_pixel_array, slice(None,None,3))
        ash_array = np.delete(ash_pixel_array,np.arange(0,ash_pixel_array.size,3))
        ash_array_GB  = ash_array.reshape(-1, 2)
        #Reformat ash pixels RG
        ash_array  = np.delete(ash_pixel_array, np.arange(1,ash_pixel_array.size,3))
        ash_array_RG = ash_array.reshape(-1, 2)
        #Reformat ash pixels RB
        ash_array  = np.delete(ash_pixel_array, np.arange(2,ash_pixel_array.size,3))
        ash_array_RB = ash_array.reshape(-1, 2)
        #Reformat non-ash pixels for GB
        nonash_array = np.delete(nonash_pixel_array, slice(None,None,3))
        nonash_array = np.delete(nonash_pixel_array,np.arange(0,nonash_pixel_array.size,3))
        nonash_array_GB =  nonash_array.reshape(-1, 2)
        #Reformat non-ash pixels for RG
        nonash_array_RG = np.delete(nonash_pixel_array, np.arange(1,nonash_pixel_array.size,3))
        nonash_array_RG =  nonash_array_RG.reshape(-1, 2)
        #Reformat non-ash pixels for RB
        nonash_array_RB = np.delete(nonash_pixel_array, np.arange(2,nonash_pixel_array.size,3))
        nonash_array_RB =  nonash_array_RB.reshape(-1, 2)

        ##Create Labels :
        
        #Create labels for GB
        nonash_labels = [1] * len(nonash_array_GB)
        nonash_labels_GB = np.array(nonash_labels)
        ash_labels = [0] * len(ash_array_GB)
        ash_labels_GB = np.array(ash_labels)

        #Create labels for RG
        nonash_labels = [1] * len(nonash_array_RG)
        nonash_labels_RG = np.array(nonash_labels)
        ash_labels = [0] * len(ash_array_RG)
        ash_labels_RG = np.array(ash_labels)

        #Create labels for RB
        nonash_labels = [1] * len(nonash_array_RB)
        nonash_labels_RB = np.array(nonash_labels)
        ash_labels = [0] * len(ash_array_RB)
        ash_labels_RB = np.array(ash_labels)

        #Checks to see whether the dimensions agree. If the dimensions agree then the pixel values for the ash and non-ash pixel values are concatenated into a single list
        if len(ash_array_GB) == len(ash_array_RG) and len(ash_array_GB) == len(ash_array_RB):
            X_before_GB = np.concatenate((ash_array_GB, nonash_array_GB), axis=0)
            X_before_RG = np.concatenate((ash_array_RG, nonash_array_RG), axis=0)
            X_before_RB = np.concatenate((ash_array_RB, nonash_array_RB), axis=0)
        elif len(ash_array_GB) != len(ash_array_RG):
            print('The dimensions of ash_array_GB and the dimensions of ash_array_RG are not equal')
            print('The dimension of ash_array_GB is {}'.format(len(ash_array_GB)))
            print('The dimension of ash_array_RB is {}'.format(len(ash_array_RB)))
        elif len(ash_array_GB) != len(ash_array_RB):
            print('The dimensions of ash_array_GB and the dimensions of ash_array_RB are not equal')
            print('The dimension of ash_array_GB is {}'.format(len(ash_array_GB)))
            print('The dimension of ash_array_RB is {}'.format(len(ash_array_RB)))
        else:
            print('An unknown error exists')

        #Scale down RGB values to values that are less than 1 and greater than 0
        X_GB = X_before_GB/255
        X_RG = X_before_RG/255
        X_RB = X_before_RB/255
        
        totallabelGB = np.array([])
        totallabelGB = np.concatenate((totallabelGB,nonash_labels_GB),axis=0)
      #  print('totallabelGB')
       # print(totallabelGB)
        totallabelRG= np.array([])
        totallabelRG = np.concatenate((totallabelRG,nonash_labels_RG),axis=0)
        totallabelRB = np.array([])
        totallabelRB = np.concatenate((totallabelRB,nonash_labels_RB),axis=0)

        #Concatenate non-ash and ash pixel labels. Make sure the index values. Make sure that the index labels are in the same order as the pixel values.
        y_GB = np.concatenate((ash_labels_GB, totallabelGB),axis=0)
        y_RG = np.concatenate((ash_labels_RG,totallabelRG),axis=0)
        y_RB = np.concatenate((ash_labels_RB,totallabelRB),axis=0)




    elif len(pixpos) == 3:
        print(pixpos)
        print('-----')
        print(rgb_im)
        ash = pixpos[0]

        ash0 = ash[0]
        ash1 = ash[1]

        ashxs = ash0[0]
        ashxe = ash1[0]

        ashys = ash0[1]
        ashye = ash1[1]

       #puts pixel data into an array
        for x in range(ashxs,ashxe):
            for y in range(ashys,ashye):
                r, g, b = rgb_im.getpixel((x, y))
                ash_pixel_array = np.append(ash_pixel_array,[r,g,b])
       #iterate through and RGB values for non-ash
        nonash = pixpos[1]

        nonash0 = nonash[0]
        nonash1 = nonash[1]

        nonashxs = nonash0[0]
        nonashxe = nonash1[0]

        nonashys = nonash0[1]
        nonashye = nonash1[1]

       #puts pixel data into an array
        for x in range(nonashxs,nonashxe):
            for y in range(nonashys,nonashye):
                r, g, b = rgb_im.getpixel((x, y))
                nonash_pixel_array = np.append(nonash_pixel_array,[r,g,b])
        #iterate through and RGB values for non-ash
        nonash = pixpos[2]

        nonash0 = nonash[0]
        nonash1 = nonash[1]
    
        nonashxs = nonash0[0]
        nonashxe = nonash1[0]

        nonashys = nonash0[1]
        nonashye = nonash1[1]
       
        #Reformat ash pixels GB
        ash_array  = np.delete(ash_pixel_array, slice(None,None,3))
        ash_array = np.delete(ash_pixel_array,np.arange(0,ash_pixel_array.size,3))
        ash_array_GB  = ash_array.reshape(-1, 2)
        #Reformat ash pixels RG
        ash_array  = np.delete(ash_pixel_array, np.arange(1,ash_pixel_array.size,3))
        ash_array_RG = ash_array.reshape(-1, 2)
        #Reformat ash pixels RB
        ash_array  = np.delete(ash_pixel_array, np.arange(2,ash_pixel_array.size,3))
        ash_array_RB = ash_array.reshape(-1, 2)
       
        #Reformat non-ash pixels for GB
        nonash_array = np.delete(nonash_pixel_array, slice(None,None,3))
        nonash_array = np.delete(nonash_pixel_array,np.arange(0,nonash_pixel_array.size,3))
        nonash_array_GB =  nonash_array.reshape(-1, 2)
        #Reformat non-ash pixels for RG
        nonash_array_RG = np.delete(nonash_pixel_array, np.arange(1,nonash_pixel_array.size,3))
        nonash_array_RG =  nonash_array_RG.reshape(-1, 2)
        #Reformat non-ash pixels for RB
        nonash_array_RB = np.delete(nonash_pixel_array, np.arange(2,nonash_pixel_array.size,3))
        nonash_array_RB =  nonash_array_RB.reshape(-1, 2)
        
        #Reformat non-ash2 pixels for GB
        nonash_array2 = np.delete(nonash_pixel_array2, slice(None,None,3))
        nonash_array2 = np.delete(nonash_pixel_array2,np.arange(0,nonash_pixel_array2.size,3))
        nonash_array_GB2 =  nonash_array2.reshape(-1, 2)
        #Reformat non-ash2 pixels for RG
        nonash_array_RG2 = np.delete(nonash_pixel_array2, np.arange(1,nonash_pixel_array2.size,3))
        nonash_array_RG2 =  nonash_array_RG2.reshape(-1, 2)
        #Reformat non-ash2 pixels for RB
        nonash_array_RB2 = np.delete(nonash_pixel_array2, np.arange(2,nonash_pixel_array2.size,3))
        nonash_array_RB2 =  nonash_array_RB2.reshape(-1, 2)
       
        totalGB = np.concatenate((nonash_array_GB,nonash_array_GB2), axis=0)
        totalRG = np.concatenate((nonash_array_RG,nonash_array_RG2), axis=0)
        totalRB = np.concatenate((nonash_array_RB,nonash_array_RB2), axis=0)
       #Checks to see whether the dimensions agree. If the dimensions agree then the pixel values for the ash and non-ash pixel values are concatenated into a single list
        if len(ash_array_GB) == len(ash_array_RG) and len(ash_array_GB) == len(ash_array_RB): 
            X_before_GB = np.concatenate((ash_array_GB,totalGB), axis=0)
            X_before_RG = np.concatenate((ash_array_RG, totalRG), axis=0)
            X_before_RB = np.concatenate((ash_array_RB, totalRB), axis=0)
        elif len(ash_array_GB) != len(ash_array_RG):
            print('The dimensions of ash_array_GB and the dimensions of ash_array_RG are not equal')
            print('The dimension of ash_array_GB is {}'.format(len(ash_array_GB)))
            print('The dimension of ash_array_RB is {}'.format(len(ash_array_RB)))
        elif len(ash_array_GB) != len(ash_array_RB):
            print('The dimensions of ash_array_GB and the dimensions of ash_array_RB are not equal')
            print('The dimension of ash_array_GB is {}'.format(len(ash_array_GB)))
            print('The dimension of ash_array_RB is {}'.format(len(ash_array_RB)))
        else:
            print('unknown error')
     


        ##Create Labels :

        #Create labels for GB
        nonash_labels = [1] * len(nonash_array_GB)
        nonash_labels2 = [1] * len(nonash_array_GB2) 
        nonash_labels_GB = np.array(nonash_labels)
        nonash_labels_GB2 = np.array(nonash_labels2)
        ash_labels = [0] * len(ash_array_GB)
        ash_labels_GB = np.array(ash_labels)

        #Create labels for RG
        nonash_labels = [1] * len(nonash_array_RG)
        nonash_labels2 = [1] * len(nonash_array_RG2)
        nonash_labels_RG = np.array(nonash_labels)
        nonash_labels_RG2 = np.array(nonash_labels2)
        ash_labels = [0] * len(ash_array_RG)
        ash_labels_RG = np.array(ash_labels)

        #Create labels for RB
        nonash_labels = [1] * len(nonash_array_RB)
        nonash_labels2 = [1] * len(nonash_array_RB2)
        nonash_labels_RB = np.array(nonash_labels)
        nonash_labels_RB2 = np.array(nonash_labels2)
        ash_labels = [0] * len(ash_array_RB)
        ash_labels_RB = np.array(ash_labels)
        #Scale down RGB values to values that are less than 1 and greater than 0
        X_GB = X_before_GB/255
        X_RG = X_before_RG/255
        X_RB = X_before_RB/255
        totallabelGB = np.concatenate((nonash_labels_GB,nonash_labels_GB2),axis = 0)
        totallabelRG = np.concatenate((nonash_labels_RG,nonash_labels_RG2),axis = 0)
        totallabelRB = np.concatenate((nonash_labels_RB,nonash_labels_RB2),axis = 0)


        #Concatenate non-ash and ash pixel labels. Make sure the index values. Make sure that the index labels are in the same order as the pixel values.
        y_GB = np.concatenate((ash_labels_GB, totallabelGB),axis=0)
        y_RG = np.concatenate((ash_labels_RG,totallabelRG),axis=0)
        y_RB = np.concatenate((ash_labels_RB,totallabelRB),axis=0)


    elif len(pixpos) == 4:
       # print('pixpos is:')
       # print(pixpos)
        ash = pixpos[0]

        ash0 = ash[0]
        ash1 = ash[1]

        ashxs = ash0[1]
        ashxe = ash1[1]

        ashys = ash0[1]
        ashye = ash1[1]

       #puts pixel data into an array
        for x in range(ashxs,ashxe):
            for y in range(ashys,ashye):
                r, g, b = rgb_im.getpixel((x, y))
                ash_pixel_array = np.append(ash_pixel_array,[r,g,b])

       #iterate through and RGB values for non-ash
        nonash = pixpos[1]

        nonash0 = nonash[0]
        nonash1 = nonash[1]

        nonashxs = nonash0[0]
        nonashxe = nonash1[0]

        nonashys = nonash0[1]
        nonashye = nonash1[1]

       #puts pixel data into an array
        for x in range(nonashxs,nonashxe):
            for y in range(nonashys,nonashye):
                r, g, b = rgb_im.getpixel((x, y))
                nonash_pixel_array = np.append(nonash_pixel_array,[r,g,b])

        #iterate through and RGB values for non-ash
        nonash = pixpos[2]

        nonash0 = nonash[0]
        nonash1 = nonash[1]
    
        nonashxs = nonash0[0]
        nonashxe = nonash1[0]

        nonashys = nonash0[1]
        nonashye = nonash1[1]

       #puts pixel data into an array
        for x in range(nonashxs,nonashxe):
            for y in range(nonashys,nonashye):
                r, g, b = rgb_im.getpixel((x, y))
                nonash_pixel_array2 = np.append(nonash_pixel_array2,[r,g,b])
    
        #iterate through and RGB values for non-ash
        nonash = pixpos[3]

        nonash0 = nonash[0]
        nonash1 = nonash[1]

        nonashxs = nonash0[0]
        nonashxe = nonash1[0]

        nonashys = nonash0[1]
        nonashye = nonash1[1]

       #puts pixel data into an array
        for x in range(nonashxs,nonashxe):
            for y in range(nonashys,nonashye):
                r, g, b = rgb_im.getpixel((x, y))
                nonash_pixel_array3 = np.append(nonash_pixel_array3,[r,g,b])
       # print('nonash pixel array 3 is:')
       # print(nonash_pixel_array3)
        
        ##Create Labels :
        ash_array  = np.delete(ash_pixel_array, slice(None,None,3))
        ash_array = np.delete(ash_pixel_array,np.arange(0,ash_pixel_array.size,3))
        ash_array_GB  = ash_array.reshape(-1, 2)
        #Reformat ash pixels RG
        ash_array  = np.delete(ash_pixel_array, np.arange(1,ash_pixel_array.size,3))
        ash_array_RG = ash_array.reshape(-1, 2)
        #Reformat ash pixels RB
        ash_array  = np.delete(ash_pixel_array, np.arange(2,ash_pixel_array.size,3))
        ash_array_RB = ash_array.reshape(-1, 2)

        #Reformat non-ash pixels for GB
        nonash_array = np.delete(nonash_pixel_array, slice(None,None,3))
        nonash_array = np.delete(nonash_pixel_array,np.arange(0,nonash_pixel_array.size,3))
        nonash_array_GB =  nonash_array.reshape(-1, 2)
        #Reformat non-ash pixels for RG
        nonash_array_RG = np.delete(nonash_pixel_array, np.arange(1,nonash_pixel_array.size,3))
        nonash_array_RG =  nonash_array_RG.reshape(-1, 2)
        #Reformat non-ash pixels for RB
        nonash_array_RB = np.delete(nonash_pixel_array, np.arange(2,nonash_pixel_array.size,3))
        nonash_array_RB =  nonash_array_RB.reshape(-1, 2)

        #Reformat non-ash2 pixels for GB
        nonash_array2 = np.delete(nonash_pixel_array2, slice(None,None,3))
        nonash_array2 = np.delete(nonash_pixel_array2,np.arange(0,nonash_pixel_array2.size,3))
        nonash_array_GB2 =  nonash_array2.reshape(-1, 2)
        #Reformat non-ash2 pixels for RG
        nonash_array_RG2 = np.delete(nonash_pixel_array2, np.arange(1,nonash_pixel_array2.size,3))
        nonash_array_RG2 =  nonash_array_RG2.reshape(-1, 2)
        #Reformat non-ash2 pixels for RB
        nonash_array_RB2 = np.delete(nonash_pixel_array2, np.arange(2,nonash_pixel_array2.size,3))
        nonash_array_RB2 =  nonash_array_RB2.reshape(-1, 2)

        #Reformat non-ash3 pixels for GB
        nonash_array3 = np.delete(nonash_pixel_array3, slice(None,None,3))
        nonash_array3 = np.delete(nonash_pixel_array3,np.arange(0,nonash_pixel_array3.size,3))
        nonash_array_GB3 =  nonash_array3.reshape(-1, 2)
        #Reformat non-ash3 pixels for RG
        nonash_array_RG3 = np.delete(nonash_pixel_array3, np.arange(1,nonash_pixel_array3.size,3))
        nonash_array_RG3 =  nonash_array_RG3.reshape(-1, 2)
        #Reformat non-ash3 pixels for RB
        nonash_array_RB3 = np.delete(nonash_pixel_array3, np.arange(2,nonash_pixel_array3.size,3))
        nonash_array_RB3 =  nonash_array_RB3.reshape(-1, 2)


        #Create labels for GB
        nonash_labels = [1] * len(nonash_array_GB)
        nonash_labels2 = [1] * len(nonash_array_GB2)
        nonash_labels3 = [1] * len(nonash_array_GB3)
        nonash_labels_GB = np.array(nonash_labels)
        nonash_labels_GB2 = np.array(nonash_labels2)
        nonash_labels_GB3 = np.array(nonash_labels3)
        ash_labels = [0] * len(ash_array_GB)
        ash_labels_GB = np.array(ash_labels)

        #Create labels for RG
        nonash_labels = [1] * len(nonash_array_RG)
        nonash_labels2 = [1] * len(nonash_array_RG2)
        nonash_labels3 = [1] * len(nonash_array_RG3)
        nonash_labels_RG = np.array(nonash_labels)
        nonash_labels_RG2 = np.array(nonash_labels2)
        nonash_labels_RG3 = np.array(nonash_labels3)
        ash_labels = [0] * len(ash_array_RG)
        ash_labels_RG = np.array(ash_labels)

        #Create labels for RB
        nonash_labels = [1] * len(nonash_array_RB)
        nonash_labels2 = [1] * len(nonash_array_RB2)
        nonash_labels3 = [1] * len(nonash_array_RB3)
        nonash_labels_RB = np.array(nonash_labels)
        nonash_labels_RB2 = np.array(nonash_labels2)
        nonash_labels_RB3 = np.array(nonash_labels3)
        ash_labels = [0] * len(ash_array_RB)
        ash_labels_RB = np.array(ash_labels)

        totalGB = np.concatenate((nonash_array_GB,nonash_array_GB2,nonash_array_GB3), axis=0)
        totalRB = np.concatenate((nonash_array_RB,nonash_array_RB2,nonash_array_RB3), axis=0)
        totalRG = np.concatenate((nonash_array_RG,nonash_array_RG2,nonash_array_RG3), axis=0)
   
       #Checks to see whether the dimensions agree. If the dimensions agree then the pixel values for the ash and non-ash pixel values are concatenated into a single list
        if len(ash_array_GB) == len(ash_array_RG) and len(ash_array_GB) == len(ash_array_RB):
            X_before_GB = np.concatenate((ash_array_GB,totalGB), axis=0)
            X_before_RG = np.concatenate((ash_array_RG, totalRG), axis=0)
            X_before_RB = np.concatenate((ash_array_RB, totalRB), axis=0)
        elif len(ash_array_GB) != len(ash_array_RG):
            print('The dimensions of ash_array_GB and the dimensions of ash_array_RG are not equal')
            print('The dimension of ash_array_GB is {}'.format(len(ash_array_GB)))
            print('The dimension of ash_array_RB is {}'.format(len(ash_array_RB)))
        elif len(ash_array_GB) != len(ash_array_RB):
            print('The dimensions of ash_array_GB and the dimensions of ash_array_RB are not equal')
            print('The dimension of ash_array_GB is {}'.format(len(ash_array_GB)))
            print('The dimension of ash_array_RB is {}'.format(len(ash_array_RB)))
        else:
            print('There is an unknown error')

        #Scale down RGB values to values that are less than 1 and greater than 0
        X_GB = X_before_GB/255
        X_RG = X_before_RG/255
        X_RB = X_before_RB/255

        totallabelGB = np.concatenate((nonash_labels_GB,nonash_labels_GB2,nonash_labels_GB3), axis = 0)
        totallabelRG = np.concatenate((nonash_labels_RG,nonash_labels_RG2,nonash_labels_RG3), axis = 0)
        totallabelRB = np.concatenate((nonash_labels_RB,nonash_labels_RB2,nonash_labels_RB3), axis = 0)
        #Concatenate non-ash and ash pixel labels. Make sure the index values. Make sure that the index labels are in the same order as the pixel values.
        y_GB = np.concatenate((ash_labels_GB, totallabelGB),axis=0)
        y_RG = np.concatenate((ash_labels_RG,totallabelRG),axis=0)
        y_RB = np.concatenate((ash_labels_RB,totallabelRB),axis=0)
    
    return X_GB, y_GB, X_RG, y_RG, X_RB, y_RB

def get_area():
    '''This function returns the area of the reference image'''
   
   #open image
    imgg = Image.open(args["image"])
    width, height = imgg.size

    #Get area
    area = width * height
    return area

def training_set():
    '''This function takes a series of unknown images,then loads and formats them ''' 
    
   #list of photos to be identified
   # unkPic = ["U1.jpg"] 
   # calPic= imlist
   # unkPic = imlist
   # print('unkPic')
   # print(unkPic)
   # trainPic = calPic
    trainPic = ['smallTest.png']
    print('training pic')
    print(trainPic)
    unknown_array=list([])
   
   #for each photo collect the pixel values and reformat them accordingly
    for elem in trainPic:
        im = Image.open(elem)
        rgb_im = im.convert('RGB')
        train_pixel_array = np.array([])
        dim_array = []
        dim1 = im.height
        dim2 = im.width
        for x in range(dim2):
            for y in range(dim1):
                dim_array = dim_array +[(x,y)] 
                r, g, b = rgb_im.getpixel((x, y))
                train_pixel_array = np.append(train_pixel_array,[r,g,b])
     #   print('dim_array is and length of dim array is:')
      #  print(dim_array)
      #  print(len(dim_array))

       # print('train pixel array is and length')
        #print(train_pixel_array)
        #print(len(train_pixel_array))

        
        #Reformat train pixels GB
        train_array  = np.delete(train_pixel_array, slice(None,None,3))
        train_array = np.delete(train_pixel_array,np.arange(0,train_pixel_array.size,3))
        train_array_GB  = train_array.reshape(-1, 2)
        #Reformat train pixels RG
        train_array  = np.delete(train_pixel_array, np.arange(1,train_pixel_array.size,3))
        train_array_RG = train_array.reshape(-1, 2)
        #Reformat train pixels RB
        train_array  = np.delete(train_pixel_array, np.arange(2,train_pixel_array.size,3))
        train_array_RB = train_array.reshape(-1, 2)
       # print('train_array_RB')
       # print(len(train_array_RB))

        #Rename
        T_before_GB = train_array_GB
        T_before_RB = train_array_RB
        T_before_RG = train_array_RG

        #Make values between 0 and 1
        T_GB =  T_before_GB/255
        T_RB =  T_before_RB/255
        T_RG =  T_before_RG/255

        #collect values into a list
        unknown_array = unknown_array + [T_GB,T_RB,T_RG]
        unknownRGB = train_pixel_array/255  
    return unknown_array,unknownRGB
def sequencethru():
   # calPic= imlist
   # unkPic = imlist
    #print('unkPic')
   # print(unkPic)
    trainPic = ['smallTest.png']
    unknown_array=list([])
    
    Tarray = []

   #for each photo collect the pixel values and reformat them accordingly
    for elem in trainPic:
        dim_array = []
        im = Image.open(elem)
        rgb_im = im.convert('RGB')
        dim1 = im.height
        dim2 = im.width         
        for x in range(dim2):
            for y in range(dim1):
                dim_array = dim_array +[(x,y)]
        Tarray = Tarray + [dim_array] 
           # print('dim_array is and length of dim array is:')
           # print(len(dim_array))
           # return dim_array
   # print('return Tarray is:')
   # print(Tarray)
   # print(len(Tarray[0]))
   # print(len(Tarray[1]))
   # print(len(Tarray))
    return Tarray

def visualize(X, y, clf):
    '''This function allows the user to visualize the data as a scatter plot '''
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
   # plt.savefig('figure'+str(i)+".jpg")
    


def plot_decision_boundary(pred_func, X, y):
    '''plots decision boundry '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    plt.show()


def classify(X, y):
    #Use a logistic regression model to classify data
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    return clf

def predict(X,y,T):
    ''' This function returns a predicted value for a given data point where [0] corresponds to red and [1] corresponds to blue'''
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    prediction_array = np.array([])
    #Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
    data = T
   # print('data')
   # print(data)
    for i in range(len(T)):
        elem_data = data[i]
        elem_data = elem_data.reshape(1, -1)

        prediction = clf.predict(elem_data)
        prediction_array = np.append(prediction_array,prediction)
       # print('prediction_array inside of the loop is:')
       # print(prediction_array)
    s = sum(prediction_array)
    l = len(prediction_array)
    t = s/l
    return prediction_array


def predictRGB(X,y,T):
    ''' This function returns a predicted value for a given data point where [0] corresponds to red and [1] corresponds to blue'''
    clf = linear_model.LogisticRegressionCV()    
    clf.fit(X, y)

def delete(labels,positions):
    reformatted_labels = [] 
    reformatted_positions = []
    
    for i in range(len(labels)):
        label = labels[i]
        poistion = positions[i]
        if label == 0.:
            reformatted_labels =  reformatted_labels + [label]
            reformatted_positions = reformatted_positions + [poistion]
    return reformatted_labels,reformatted_positions

def draw(positions,image,color,name,number):
    "this function draws on labeled pixels in order to differentiate them from each other at given positions"
    imlist = ['smallTest.png']
    for img in imlist:
        counter = 1
        im = Image.open(img)
        draw = ImageDraw.Draw(im)
    
        draw.point(positions, fill= color )
        del draw
        im.show()
        im.save(str(color)+str(number)+".png")
        counter = counter + 1





def main():
    #intialize arrayrs
    predGB = []
    predRG = []
    predRB = []
    prediction_listGB = []
    prediction_listRB = []
    prediction_listRG = []
    meanpred = []
    comblists=  []
    #call known data
    X_GB, y_GB ,X_RG, y_RG, X_RB, y_RB  = GUI_data()
    
   #load training data
    training_data = np.array([X_GB,X_RG,X_RB]) 
    
   #concantonated list of unknown images
    conlist,unknownRGB =  training_set()
    
    dim_arrays = sequencethru()
    
    #seperated list of unknown pixel images
    sep_list = [conlist[x:x+3] for x in range(0, len(conlist), 3)]
  #  print('sep list')
   # print(len(sep_list))
    #call known data
    X_GB, y_GB ,X_RG, y_RG, X_RB, y_RB  = GUI_data()
    images = imlist
    #create labels for teh known data
    labels = np.array([y_GB,y_RG,y_RB])
   # print('dim arrays')
   # print(len(dim_arrays))
   # print(len(dim_arrays[0]))
   # print(len(dim_arrays[1]))
#    for i in range(len(training_data)):
 #       clf = classify(training_data[i],labels[i])
  #      visualize(training_data[i],labels[i], clf)
       # plot_decision_boundary(clf, training_data[i], labels[i])
       
    for i in range(len(sep_list)):
        dim_array = dim_arrays[i]
        unknown_data = sep_list[i]
        image = images[i]
        area = get_area()
        prediction_list = []
        final_list = []
        predictionGB = predict(training_data[0],labels[0],unknown_data[0])
        predictionRG = predict(training_data[1],labels[1],unknown_data[1])
        predictionRB = predict(training_data[2],labels[2],unknown_data[2])
        prediction_listGB = prediction_listGB + [list(predictionGB)]
        # print('prediction list GB is:')
       # print(prediction_listGB)
        #print(len(prediction_listGB))i
       # print('predictionGB is:')
       # print(predictionGB)
       # print(len(predictionGB))
       # print(len(predictionGB))
       # print(type(predictionGB))
       # print(' dimension array')
       # print(dim_array)
       # print(len(dim_array))
        predGB= predGB + [1 - sum(predictionGB)/len(predictionGB)] 
        predRG= predRG + [1 - sum(predictionRG)/len(predictionRG)]
        predRB = predRB + [1 - sum(predictionRB)/len(predictionRB)]
        avgpred =1- (sum(predictionGB)/len(predictionGB) + sum(predictionRG)/len(predictionRG) + sum(predictionRB)/len(predictionRB))/(3)
        meanpred = meanpred + [avgpred]  
       # print('length of dim_array')
       # print(len(dim_array))
       # print('lenpredictionGB')
       # print(len(predictionGB))
       # print('The argeement for GB for photo {} is {}'.format(i,sum(predictionGB)/len(predictionGB)))
      # print('The argeement for RG for photo {} is {}'.format(i,sum(predictionRG)/len(predictionRG)))
      # print('The argeement for RB  for photo {} is {}'.format(i,sum(predictionRB)/len(predictionRB)))
       # print('Therefore the average surface area covered is {}'.format(avgpred))
        
        lGB,pGB = delete(predictionGB,dim_array)
        lRB,pRB = delete(predictionRB,dim_array)
        lRG,pRG = delete(predictionRG,dim_array)
        draw(pGB,image,(0,255,255),'GB',int(i))
        draw(pRG,image,(255,255,0),'RG',int(i))
        draw(pRB,image,(255,0,255),'RB',int(i))
        print('predGB')
        print(predGB)
        print('predRG')
        print(predRG)
        print('predRB')
        print(predRB)
        print('meanpred')
        print(meanpred)
        
       # if meanpred[0] != 1.0 or  meanpred[1] != 0.0 or meanpred[2] != 0.5 :
        #    print('Calibration Incorrect')
       # else:
        #    print('Calibration Correct')
   # comblists = [predRB,predRG,predGB,meanpred]        
       
if __name__ == "__main__":
    main()

