from keras.models import Sequential, Model
from keras.layers import Lambda,Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import os
import csv
import math
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


samples = []
with open('./mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

images = []
angles=[]
s=0.2
count=0;
steering_factor = [0.0, s,-s];
for sample in samples[1:]:
    print(count);
    count=count+1;
    for i in range(3):
        name = './mydata/IMG/'+sample[i].split('/')[-1]
        image = cv2.imread(name)
        angle = float(sample[3])
        images.append(image)
        angles.append(angle+steering_factor[i])
        images.append(cv2.flip(image,1))
        angles.append((angle+steering_factor[i])*-1.0)
        

X_train = np.array(images)
y_train = np.array(angles)
 

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu")) 
model.add(Conv2D(64, (3, 3), activation="relu")) 
model.add(Conv2D(64, (3, 3), activation="relu")) 
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(X_train, y_train, nb_epoch=3,validation_split=0.2,shuffle=True)
model.save('model.h5')
