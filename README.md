# CNN based driving behavior cloning

## OVerview
In this project, a CNN Neural Network architecture is trained with user driving behavior data on a track and then the car is attempted to be driven autonomously around the track. [NVIDIA's End to End Deep neural network architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) is implemented in this project.

* The input to the neural network architecture is a set of camera images obtained from three sources of camera located in the left, center and right sections of the hood of the car. This is to map recovery paths from each camera. 
* The predicted output variable is the steering angle command.

![Network Architecture](https://github.com/ashsiv/CNN-based-driving-behavior-cloning/blob/master/images/architecture.JPG)

---
## Data Preprocessing
### 2D Cropping
The incoming data from three cameras (left, right and center) are first cropped to appropriate size before subjecting them to training. This helps to keep the region of focus only within the lane of interest. As you can see, cropping helps to remove unnecessary background information such as hood of the car, sky, mountains etc. Plus computatinally it is effective to work with cropped image sizes.

#### Original Image:
![Image from camera](https://github.com/ashsiv/CNN-based-driving-behavior-cloning/blob/master/images/original.jpg)
#### Cropped Image:
![Image cropped to region of interest](https://github.com/ashsiv/CNN-based-driving-behavior-cloning/blob/master/images/cropped.JPG)

### Normalization
In keras implementation, a lambda layer is used to parallelize image normalization
```
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
```
### Steering offset 
Multiple camera images aid data augmentation. For example, if the model is trained to associate a given image from the center camera with a left turn, then the model can also be trained to associate the corresponding image from the left camera with a somewhat softer left turn and the  image from the right camera with an even harder left turn. 
During training, the left and right camera images are used to train the model as if they were coming from the center camera. For this purpose, a steering offset factor of + 0.2 deg is used for left image and a steering offset of -0.2 deg is used for the right image.
---
## Model summary
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
```
---
## Training and Validation loss
For chosen three epochs, the training and validation loss are found to be monotonically decreasing.
```
Epoch 1/3
53/53 [==============================] - 45s 842ms/step - loss: 0.0316 - val_loss: 0.0271
Epoch 2/3
53/53 [==============================] - 41s 774ms/step - loss: 0.0254 - val_loss: 0.0244
Epoch 3/3
53/53 [==============================] - 41s 773ms/step - loss: 0.0241 - val_loss: 0.0232
```
---
## Results

[Output Video.mp4](https://github.com/ashsiv/CNN-based-driving-behavior-cloning/blob/master/output_video.mp4)

1. The car was found to safely manuever around the track.
2. Sufficient training data - 2 to 3 laps of driving, additional training data around curves of the road track, augmenting the data with flipped images & offset corrected steering angles helped to keep the lane cross track error as low as possible.


