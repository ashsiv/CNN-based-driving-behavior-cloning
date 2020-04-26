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

## Import data path locations && do training/validation split
samples = []
with open('./mydata/mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

## Define generator function to load and preprocess data on the fly in batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            s=0.2
            count=0;
            steering_factor = [0.0, s,-s];
            for batch_sample in batch_samples:
                #print(count);
                count=count+1;
                for i in range(3):
                    name = './mydata/mydata/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    images.append(image)
                    angles.append(angle+steering_factor[i])
                    images.append(cv2.flip(image,1))
                    angles.append((angle+steering_factor[i])*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)        
        
# Set our batch size
batch_size=100       

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
                
## Define the Model Architecture
model = Sequential()
model.add(Lambda(lambda x: (x/255)-0.5, input_shape=(160,320,3))) # trim image to only see section with road
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
model.fit_generator(train_generator,steps_per_epoch=math.ceil(len(train_samples)/batch_size),validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size),epochs=3, verbose=1)

## Save the model
model.save('model.h5')
model.summary()