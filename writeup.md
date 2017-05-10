# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plot.jpg "Visualization"
[image2]: ./gntest.jpg "Grayed and normalized"
[image21]: ./gntrain.jpg "Grayed and normalized"
[image22]: ./gnvalid.jpg "Grayed and normalized"
[image3]: ./randomoriginal.jpg "Random Noise"
[image31]: ./train_original.jpg "Train original image"
[image32]: ./valid_original.jpg "Validation original image"
[image33]: ./test_original.jpg "Test original image"
[image4]: ./testimages/1.png "Traffic Sign 1"
[image5]: ./testimages/2.png "Traffic Sign 2"
[image6]: ./testimages/3.png "Traffic Sign 3"
[image7]: ./testimages/4.png "Traffic Sign 4"
[image8]: ./testimages/5.png "Traffic Sign 5"

## [Rubric Points](https://review.udacity.com/#!/rubrics/481/view)
---
Here is a link to my [project code](https://github.com/ahubi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset

The code for this step is contained in the third code cell of the IPython notebook.  
The table below shows the number of train labels for the first 9 classes

|Class ID     | Name                      | Number of labels|
|-------------|:--------------------------|----------------:|
|0            |Speed limit (20km/h)       |180              |
|1            |Speed limit (30km/h)       |1980             |
|2            |Speed limit (50km/h)       |2010             |
|3            |Speed limit (60km/h)       |1260             |
|4            |Speed limit (70km/h)       |1770              |
|5            |Speed limit (80km/h)       |1650              |
|6            |End of speed limit (80km/h)|360              |
|7            |Speed limit (100km/h)      |1290              |
|8            |Speed limit (120km/h)      |1260              |
|9            |No passing                 |1320              |
|...          | | |

Here is an exploratory visualization of the data set. It is a bar chart showing number of train labels for each Class ID from 0 to 42.

![alt text][image1]

### Model Architecture

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because according to this [article](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740) gray-scaling "simplifies the algorithm and reduces computational requirements" Additionally color images introduces more information to the pipeline which would require more training information to achieve good training result. There are many ways to convert the image from RGB to gray-scale, according to the the article at least thirteen. I used the simple average method (R+G+B)/3. Additionally gray-scaling was explained by the trainer in the video session as a method to use.

Here is an example of a traffic sign image before and after gray-scaling.

Original images:

![alt text][image31] ![alt text][image32] ![alt text][image33]

And here are the preprocessed images:

![alt text][image21] ![alt text][image22] ![alt text][image2]

As a last step, I normalized the image data because this was the advice from the trainer in the videos on udacity in which he explained the numerical stability. Running algorithms with zero centered numbers makes the calculations less complex and it's easier for the optimizer. As suggested by the trainer I used (Pixelvalue - 128) / 128 as a normalization method. According to this [article](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html) augmentation is another method to increase performance of the nework. In my solution I didn't use any augmentation methods.

#### 2. Training and validaiton split

There is no code for splitting in training, validation and testing data. The data is already separated at loading in the first IPython code cell. Please see statistics at the beginning of the writeup for numbers about training, validation and testing set.


#### 3. Final model architecture

The code for my final model is located in the fifth cell of the Ipython notebook.
My final model consisted of the following layers:

| Layer         		|     Description	        					                |
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 gray-scaled normalized image   						|
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x6 	      |
| RELU					    |												                            |
| Max pooling	      | 2x2 stride, outputs 14x14x6 				              |
| Convolution 5x5	  | 1x1 stride, valid padding, outputs 10x10x16      									                                        |
| RELU		          |    									                              |
| Max pooling				| 2x2 stride, Outputs 5x5x16        									                                        |
|Flatten						| Input 5x5x16 output 400				                    |
|Fully connected		|	Input = 400, Output = 120											  |
|RELU		            |												                          |
|Fully connected		|	Input = 120, Output = 84											  |
|RELU		            |											                            |
|Fully connected		|	Input = 84, Output = 43											  |



#### 4. Training Model

The code for training the model is located in the tenth cell of the ipython notebook.
To train the model, I used 50 Epochs, Batch size of 128, learning rate of 0.001, mean of 0 and standard deviation of 0.1. AdamOptimizer was used.

#### 5. Final solution approach
The code for calculating the test accuracy of the model is located in the eleventh cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.944
* test set accuracy of 0.914

If a well known architecture was chosen:
* Lenet architecture as used during udacity teaching was deployed.
* Lenet architecture's main purpose was originaly recognition of hand written letters.
I think it suits well for recognition of traffic signs.
* The final results show that training accuracy (1), validation acurracy (0.944) and test accuracy (0.914) are close to each other and therefore I would assume that the model is working proper for sign recognition.

### Test a Model on New Images

#### 1. Test on 5 traffic signs

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

During my experiments on the test images I observed that the first 4 images were always recognized correctly. Tha last image was not always predicted correctly. I think this is due to angle of the image was taken.

#### 2. Prediction discussion

The code for making predictions on my final model is located in the thirteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right-of-way at the next intersection | Right-of-way at the next intersection   									|
| Speed limit (30km/h)     			| Speed limit (30km/h) 										|
| Priority road				| Priority road											|
| Turn left ahead	      		| Turn left ahead					 				|
| Road work			| Road work      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. From time to time I observed the accuracy of 80% in this case only the first four images were predicted properly. This compares favorably to the accuracy on the test set of 0.914.

#### 3. The top 5 softmax probabilities

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a "Right-of-way at the next intersection" (probability of 1, actually I would expect the probabilities to sum up to 1 in total, but already the first probability equals to 1...?), and the image does contain a "Right-of-way at the next intersection". The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Right-of-way at the next intersection   									|
| 9.85556080e-16     				| Double curve 										|
| 3.63415469e-19					| Beware of ice/snow											|
| 4.85972404e-31	      			| Speed limit (80km/h)					 				|
| 2.33963319e-31				    | Traffic signals      							|
The prediction can be considered as very solid.

For the second image "Speed limit (30km/h)":

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Speed limit (30km/h)   									|
| 8.91795915e-10     				| Speed limit (50km/h) 										|
| 4.15478307e-13					| Speed limit (80km/h)											|
| 2.41675864e-15	      			| Speed limit (60km/h)					 				|
| 5.72162316e-20				    | Stop      							|
The prediction can be considered as very solid.

For the third image "Priority road":

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Priority road   									|
| 6.50071143e-12     				| Roundabout mandatory 										|
| 3.24874979e-19					| Children crossing											|
| 2.97306600e-19	      			| No passing					 				|
| 5.54823189e-20				    | Turn right ahead      							|
The prediction can be considered as very solid.

For the fourth image "Turn left ahead":

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Turn left ahead   									|
| 1.36934507e-12     				| End of no passing 										|
| 8.06996505e-13					| Beware of ice/snow											|
| 5.52545917e-14	      	| End of all speed and passing limits					 				|
| 4.61378423e-16				    | Speed limit (120km/h)      							|
The prediction can be considered as very solid.

For the fourth image "Road work":

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99         			| Road work   									|
| 0.011     				| Bicycles crossing 										|
| 1.02209094e-12					| Dangerous curve to the right											|
| 7.40086398e-15	      	| Slippery road					 				|
| 3.78407438e-15				    | Road narrows on the right      							|
At the last traffic sign it can be seen that the model isn't that solid on predicting this image. The first softmax output < 1. This correlates with my observations for this image. In some cases the image couldn't be recognized properly.
