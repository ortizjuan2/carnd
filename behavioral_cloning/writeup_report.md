#**Behavioral Cloning**

##Writeup Report

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
*   Use the simulator to collect data of good driving behavior
*   Build, a convolution neural network in Keras that predicts steering angles from images
*   Train and validate the model with a training and validation set
*   Test that the model successfully drives around track one without leaving the road
*   Summarize the results with a written report


[//]: # (Image References)


[image2]: ./examples/img2.jpg "Center lane driving"
[image3]: ./examples/img3.jpg "Recovery Image"
[image4]: ./examples/img4.jpg "Recovery Image"
[image5]: ./examples/img5.jpg "Recovery Image"
[image6]: ./examples/img6.jpg "Normal Image"
[image7]: ./examples/img7.jpg "Flipped Image"
[image8]: ./examples/img8.jpg "Motion blur Image"
[image9]: ./examples/img9.jpg "Motion blur Flipped Image"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
*   model.py containing the script to create and train the model, it also has a generator function to pass batches of images and labels to the keras function fit_generator
*   drive.py for driving the car in autonomous mode, the only modifications includes the preprocessing of the image before it is passed to the predictor, and a basic PID controller for the throttling.
*   model.h5 containing the model and the weights after training the convolutional neural network
*   writeup_report.md summarizing the results
*   Below there are additional scripts used to preprocessing the data, although they were not submitted, they can be found in my [github repo.](https://github.com/ortizjuan2/carnd/tree/master/behavioral_cloning)

*   syncdata.py used to sync images and log driving file in order to remove images with sharp angles, images with zero angle and to delete references to images no longer available
*   balancedata.py this script first classify each image in and bin, then it selects randomly the same amount of sample images from each angle bin thus balancing the data
*   augment_data.py takes the balanced data and perform augmentation of each image using flip, motion blur and motion blur flipped.
*   get_data_v02.py takes the augmented data and makes a h5 dataset in order to facilitate passing samples to keras during training.

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
*NOTE:* the drive.py file was modified to include the same image preprocessing as in the model training, which was just to normalize the image between 0 and 1.0. It also includes a basic PID controller for the throttling in order to maintain a fix speed of 10.0 MPH
####3. Submssion code is usable and readable
The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. In summary this file creates a model using keras, fits the model with an Adam optimizer and then starts training using keras fit_generator.
This file also includes the generator function to feed images in batches instead of the whole set of images, which due to memory constrains it was not possible in my computer.

###Model Architecture and Training Strategy

####1. Model
My model starts with an input layer that receives a batch of images of size (120, 320, 3). Then, five (5) consecutive convolutional layers with 24, 36, 48, 64 and 64 filters. Inside each convolutional layer, keras is adding an ELU activation which is not shown below. (See function get_model() inside model.py file).
After the last convolutional layer I began to add a Batch Normalization layer and then an ELU Activation layer before the first fully connected layer. Then, follow 5 combinations of Dense, Batch Normalization and ELU Activation layers. Last layer, the output layer is a Dense layer of size 1, without an activation.

| Layer (type)       | Output Shape         | Param #    |
| ------------------ | -------------------- | ----------:|
| input              | (?, 120, 320, 3)  | 0          |
| Convolution2D      | (?, 58, 158, 24)  | 1824       |
| Convolution2D      | (?, 27, 77, 36)   | 21636      |
| Convolution2D      | (?, 12, 37, 48)   | 43248      |
| Convolution2D      | (?, 5, 18, 64)    | 27712      |
| Convolution2D      | (?, 3, 16, 64)    | 36928      |
| BatchNormalization | (?, 3, 16, 64)     | 256        |
| Activation         | (?, 3, 16, 64)    | 0          |
| Flatten            | (?, 3072)         | 0          |
| Dense              | (?, 256)          | 786688     |
| BatchNormalization | (?, 256)          | 1024       |
| Activation         | (?, 256)          | 0          |
| Dropout            | (?, 256)          | 0          |
| Dense              | (?, 128)          | 32896      |
| BatchNormalization | (?, 128)          | 512        |
| Activation         | (?, 128)          | 0          |
| Dropout            | (?, 128)          | 0          |
| Dense              | (?, 64)           | 8256       |
| BatchNormalization | (?, 64)           | 256        |
| Activation         | (?, 64)           | 0          |
| Dense              | (?, 16)           | 1040       |
| BatchNormalization | (?, 16)           | 64         |
| Activation         | (?, 16)           | 0          |
| Dense              | (?, 1)            | 17         |


*   **Total params:** 962,357
*   **Trainable params:** 961,301
*   **Non-trainable params:** 1,056

####2. Attempts to reduce overfitting in the model

The model contains two (2) dropout layers in order to reduce overfitting (model.py lines 85 and 89).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 119 and 128). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. See video in my [github repo.](https://github.com/ortizjuan2/carnd/tree/master/behavioral_cloning)

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 112). Also the batch size was fixed at 128 images due to memory constrains in my desktop computer.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a similar model to the one used by Nvidia paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). This model gave excellent result, so I decide to use a similar architecture modifying the number of neurons to fit the resources of my desktop computer and to allow the parameters to fit my GPU which only have 2 GB of memory.

In order to gauge how well the model was working, I split my image and steering angle data into a training and test set. I found that my model had a low mean squared error on the training set and a low mean squared error on the test set when trained by about 40 epochs using 5.000 images. This implied that the model was working well but adding more images will definitively help improve the performance of the prediction reducing the training and test errors.

Then I starting to add more images taking them at places like right and left turns, placing the car near the left border of the lane and turning towards the center and placing the car near the right border of the lane and turning towards the center. This helped to maintain the car in the center of the lane.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track specially in the curves after the bridge, so to improve the driving behavior in these cases, I took more sample images and added to the training dataset, this helped to drive passing those turns very well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is the one depicted in section 1 of section Model Architecture and Training Strategy.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from the left side, then the right side and back to the center of the lane :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Also, I alter each original image with a motion blur kernel and added to the data set, normal and flipped, this is an example of altered image.

![alt text][image8]
![alt text][image8]

After the collection process, I had more o less 12.000 images and angles. I put them in a h5 dataset (training 80% and test 20%). The only preprocessing at this stage was to crop the image fro **160, 320, 3** to **120, 320, 3**, this to remove the sky and the front of the car.


Each time I generate a train and test dataset, the images are randomly shuffled.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 40. If I run more that 40 epochs the car starts shaking on the simulator. I used an adam optimizer so that manually training the learning rate wasn't necessary.
