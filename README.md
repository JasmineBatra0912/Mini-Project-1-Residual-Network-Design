**Mini Project 1**

Subject: Deep Learning
Project: Residual Network Design
Language used: Python

Department of Computer Engineering
New York University
New York, NY 11201	

Project Members:
Jasmine Batra - jb7854@nyu.edu
Sachin Karthik Kumar - sk9622@nyu.edu
Rudrasinh Nimeshkumar Ravalji - rnr8237@nyu.edu


**Abstract** 

Deep Learning algorithms are created to replicate the function of the human cerebral cortex. These algorithms are deep neural network representations, or neural networks with numerous hidden layers. Convolutional neural networks are deep learning algorithms that can train enormous datasets with millions of parameters using 2D images as input and filtering to get the required outputs. To improve the performance of neural networks, the proposal of residual neural networks has a great contribution to the advancement of the bottleneck of traditional networks. Our goal was to maximize the accuracy of CIFAR-10 by reducing the number of model parameters under limit. On the CIFAR-10 dataset, we reduced the number of parameters and performance is evaluated using different augmentation techniques different regularization techniques on ResNet architectures.
  
**Introduction**

Deep Learning has been developing since the introduction of AlexNet. As with the emergence of GoogleNet, VGG, and Inception, the deeper the number of layers in a network is, the more complex the network is. The deeper the network is, more features can be extracted and the performance will be strong[4]. Residual Networks are one of the most used for image classification. Detecting an object or recognizing an image from a digital image or video is a difficult challenge.[1]. Deep neural networks have layers that extract many features and hence provide multiple degrees of abstraction. This cannot extract or work on numerous data sets unlike shallow networks. By inputting a 2D image and convolving it with filters/kernels and producing output volumes, convolutional neural networks is a sophisticated deep learning algorithm capable of coping with millions of parameters and saving computing cost.


**Dataset**

CIFAR- 10 dataset consists of 60,000 images divided into 10 categories with 32x32 pixel color images as illustrated in In Fig 2 which includes airplanes, cars, birds, cats, deer, dogs, frogs, horse boats and trucks. There are 50000 pictures in the training set and 10000 pictures in the testing set. It has natural photographs in it and aids in the implementation of image detection systems. On the CIFAR-10 dataset, the model is trained with real-time data augmentation. CIFAR-10 is a series of images for teaching a computer to recognize items. Because the images in CIFAR-10 are low-resolution, researchers can quickly test different algorithms to discover which one’s work.


**How are the goals of the problem statement achieved?**
**I. Methodology**

  1. For finding the right model hyperparameter, we tried multiple combinations of hyperparameters by keeping into consideration the trainable parameter less than 5M and accuracy greater than 80. 
  2. We even tried the model with combinations present in Table 2, but the count of parameters exceeded more than 5 million which violated the condition specified. So, these models were not taken into consideration further. 
  3. Out of the combinations above we figured out Channel size- 40, Layer- 4, Block size- 2, kernel size- 3 and padding- 1 are the best hyperparameters by training the model with Adam optimizer and Learning rate 0.001 until 90 and from 90 to 110 it is 0.01 and after 110 it is 0.1. 
  4. We observed that Adam and RMSprop performed well with respect to our model. 
  5. We tried these different optimizers with different Learning Rates. We increased the learning rate, did fine tuning for some epochs, and stop our training when we saw that there is no further change in the accuracy. We have done manual tuning of learning rate.
  6. we conclude that Average pool and Batch size shows no effect on parameters. Layers, Block size , channel size ,Kernel and padding have major impact on parameters. 
  7. Out of the combinations above we figured out Channel size is 40, Layer is 4, Block size is 2 , kernel size is 3 and padding is 1 are the Best hyperparameters. So using these Best Hyperparameters we trained our model with different optimizers and kept the learning rate as 0.001.


**II. Data augmentation:**

Data augmentation techniques can be used to artificially expand by carefully choosing transforms. Data augmentation techniques involve data image distortion and alteration for processing to obtain more data. The following are some of the techniques we have used: 
  1. Random cropping: 
  Cropping some areas of the image and dividing it into subsets from the main image. Output size of the randomCrop will be 32. Padding parameters will add white spaces to the edge of the image before cropping such as 0 padding to left , padding of size 2 to top, padding of size 3 to top and padding of size 4 to bottom.
  2. RandomRotation: 
  This entails rotating the images in any way and generating new images. Rotating the image through 10 degrees. The feature of object in image remains the same this way the model can learn about the feature of same object.
  3. Random Horizontal flipping:
  When we perform a horizontal flip the object inside the image remains the same but the change is in the angle of the object present in the image. It will randomly flip the image horizontally with probability of 0.5.

**III. Optimizers**

There is a wide range of optimizers available when training a resnet model. The mainly used optimizers are:

  1. RMSprop:
  RMSprop is also known as root mean square prop, is a gradient optimizer that works with the root mean square value of the gradient change. With the help of the rms value, the gradient parameters are determined by changes in weights and bias. The algorithm's learning rate defines how many steps it will take to reach the global minimum.
  2. Stochastic gradient descent :
  It updates all the parameters for each training example x(i) and label y(i) individually.Since of its unpredictability in descent, SGD is generally noisier than standard Gradient Descent because it takes a longer number of iterations to reach the minima.
  3. Adadelta:
  Adadelta is an Adgrad modification that aims to slow down the program's aggressive, monotonically decreasing learning rate. Adadelta restricts the window of accumulated past gradients to a specified size rather than gathering all past squared gradients.
  4. Adam:
  Adam is one of the most common optimizers, also known as adaptive Moment Estimation, it combines the best features of the Adadelta and RMSprop optimizers into one and hence performs better for the majority of tasks. Adam preserves an exponentially decaying average of past gradients mt, comparable to Adadelta and RMSprop, in addition to an exponentially decaying average of past squared gradients vt.
  5. Adagrad 
  Adagrad adapts the learning rate to the parameter that is performing low learning rates for parameters  associated with frequently occurring features and for higher learning rates for parameters associated with infrequents features.

**Results**
After completing the preprocessing stage, the ResNet model is used to learn. Each architecture layer on ResNet has its own set of conditions for its development. The convolution layer is commonly used in feature learning models. The ResNet architecture includes a batch normalization, activation and pooling layer in addition to the convolution layer. Following the convolution layer, batch normalization is used, and the activation layer uses ReLu activation function. There are residual blocks in each ResNet architecture that divide the convolution layer into four layers.

Count of parameters	4368690
Test Accuracy Achieved	91.23%

|		Optimizer	  | Learning rate |		Accuracy	 |
| ------------- | ------------- |	------------ |			
| 		Adam 			|		 	0.001 		|			91.23		 |
| 		RMSprop	  | 		0.001 		|			90.07		 |
| 		Adagrad		|		 	0.001 		|			81.44		 |
| 		Adadelta  | 		0.001 		|			66.68		 |
| 		SGD				|		 	0.001 		|			79.79		 |

|		Optimizer	  | Learning rate |		Accuracy	 |
| ------------- | ------------- |	------------ |			
| 		RMSprop		|		 	0.0001 		|			87.62		 |
| 		RMSprop	  | 		0.001 		|			90.07		 |
| 		RMSprop		|		 	0.01 			|			88.62		 |
| 		Adam		  | 		0.0001 		|			89.44		 |
| 		Adam			|		 	0.001 		|			91.23		 |
| 		Adam			|		 	0.01 			|			89.73		 |


Adam achieved the highest Test accuracy.

**Steps for running the program**

  1. Copy the repository link from GitHub Repository.
  2. Open terminal and Git Clone respository link. You will see the downloaded project1_model.py in the folder.
  3. Run project1_model.py in the terminal.
  4. .out file will be generated which will have the test and train accuracy in the same directory 


