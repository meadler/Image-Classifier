import PIL
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import shutil
i=0
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
#ap.add_argument("-a", "--ashoutput", required=True,
#        help="path to output ash pics")
#ap.add_argument("-n", "--nonAsh", required=True,
#        help="path to output non ash pics")
args = vars(ap.parse_args())

model = load_model(args["model"])
path = args["dataset"]
dirs = os.listdir( path )
for item in dirs:
        image = cv2.imread(path+"/"+item)
        print(path+"/"+item)
        if image is not None:
            image = cv2.resize(image, (28, 28))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            (notSanta, santa) = model.predict(image)[0]
            if (santa>notSanta):
              shutil.move(path+"/"+item, "/Users/Max/ashSort")
              print(santa)
              print(notSanta)
            if (notSanta>santa):
             shutil.move(path+"/"+item, "/Users/Max/ashNon")
             print(santa)
             print(notSanta)		        
        else:
             print("null baybe")
         
