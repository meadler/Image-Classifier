#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:35:05 2018

@author: parkershankin-clarke
"""

# import the necessary packages
import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt3 = []
refPt2 = []
refPt = []
cropping = False
pixpos = []
class2 = []
class3 = []

def click_and_crop(event, x, y, flags, param):
    global refPt3,refPt2,refPt, cropping,pixpos, class2,class3
    m = cv2.waitKey(0)
    if m == 27:
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            pixpos.append(refPt)
            cropping = False
    
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
        #    cv2.destroyAllWindows()
                
    if m == ord("m"):
        cv2.destroyAllWindows()

        '''
        if 'q' == chr(c & 255):
            if event == cv2.EVENT_LBUTTONDOWN:
                    refPt = [(x, y)]
                    cropping = True
          
            #check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                    # record the ending (x, y) coordinates and indicate that
                    # the cropping operation is finished
                
                    refPt.append((x, y))
                    pixpos.append(refPt)
                    cropping = False

                    # draw a rectangle around the region of interest
                    cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
                    cv2.imshow("image", image)

        if 'g' == chr(c & 255):
            if event == cv2.EVENT_LBUTTONDOWN:
                    refPt2 = [(x, y)]
                    cropping = True

            #check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                    # record the ending (x, y) coordinates and indicate that
                    # the cropping operation is finished
                
                    refPt2.append((x, y))
                    class2.append(refPt2)
                    cropping = False

                    # draw a rectangle around the region of interest
                    cv2.rectangle(image, refPt2[0], refPt2[1], ( 255,0, 0), 2)
                    cv2.imshow("image", image)

        if 'u' == chr(c & 255):
            if event == cv2.EVENT_LBUTTONDOWN:
                    refPt3 = [(x, y)]
                    cropping = True

            #check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                    # record the ending (x, y) coordinates and indicate that
                    # the cropping operation is finished

                    refPt3.append((x, y))
                    class3.append(refPt3)
                    cropping = False

                    # draw a rectangle around the region of interest
                    cv2.rectangle(image, refPt3[0], refPt3[1], (0, 0,255), 2)
                    cv2.imshow("image", image)

'''
print("im outside the function")
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(0)  

