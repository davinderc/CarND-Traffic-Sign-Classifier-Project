#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/visualization.png "Visualization"
[image2]: ./writeup/unnormalized.png "Original Image"
[image3]: ./writeup/normalized.png "Normalized Image"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/davinderc/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_ClassifierGPUTrainedFinal.ipynb)
This project was undertaken as part of a three-term course in artificial intelligence and computer vision for self-driving cars, offered by Udacity through their online platform.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the fifth code cell of the IPython notebook. I used numpy to obtain the information, since I had some knowledge of it from the previous project:

* The size of training set is 34799 samples.
* The size of test set is 12630 samples.
* The shape of a traffic sign image is 32x32x3 (3 color channels, RGB).
* The number of unique classes/labels in the data set is 43, although the output of the cell shows 42, as I forgot to add 1 for the class represented by the number 0.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the sixth code cell of the IPython notebook.

Here is an exploratory visualization of the data set. I used a bar graph to show the number of samples in each class of the dataset, with the x-axis showing the class number and the y-axis showing sample count. The axes labels were not included in the image as I didn't focus as much on the details of matplotlib, but rather much more on using TensorFlow.

![alt text][image1]

It is very clear that the dataset is not balanced and most of the images are in the first 15 or so classes, with significantly fewer images in the later classes. There were about three of the higher classes that had higher sample counts.

In addition, I wanted to know what each of the signs looked like, since I don't live in Germany and know there are probably differences from the signs I'm used to. I randomly picked signs of each class and plotted them below the histogram to have an idea of what kinds of signs and image qualities I would be using for classification.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the seventh code cell of the IPython notebook.

I decided not to convert my dataset to grayscale, because I felt that I would be losing information in doing so, and since I would be training on powerful GPU cloud services anyways, it really didn't matter that it would take more computational power. However, I did dedice to normalize all of my images, so that the optimization algorithm would have a shorter path to a local minimum, and training would be faster. I decided to normalize from -1.0 to 1.0, although initially I wanted to use 0 to 1.0.

Here is an example of a traffic sign image before and after normalizing. Of course, the normalized image is hardly recognizable, even though some features, such as the edges of the traffic signs are visible in some cases. This is just to test that the normalization function I used (from matplotlib.colors) actually works. The code for showing these is in the eighth code cell.

![alt text][image2]
![alt text][image3]

Finally I shuffled the data in the ninth code cell.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

There was no code for splitting the data into training and validation sets, since at this point the datasets were provided as three files: training, validation, and test sets.  

My final training set had 34799 images. My validation set and test set had 4410 and 12630 images, respectively.

I decided not to augment my dataset, mostly due to not having time to spend on learning how to augment it and how to make use of the available libraries to do this. However, towards the end of the course, I intend to go back and improve these implementations, including augmenting the dataset. This would include mirroring, rotating, scaling, and otherwise altering images (adding noise, changing lighting, etc.) in ways that allow them to still be recognizable as their original classes, or even as other classes. This would make the classifier network better able to handle images taken from different situations.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eleventh cell of the Ipython notebook. The model was based closely on the LeNet architecture used in a previous neural network exercise in the course. The weights and bias values were set using the TensorFlow truncated normal function, with a mean of 0 and standard deviation of 0.05. These values seemed to result in the best and fastest training accuracy results.

Some changes were made to the LeNet architecture. The output size was changed to 43, to accommodate the 43 classes of traffic signs. In addition, dropout layers were introduced just before the last two fully connected layers, to ensure that the model would not overfit the training set, and be able to provide accurate predictions in generalized situations. All fully connected layers' weights were used in L2 regularization, to also decrease overfitting, by adding to the loss function when weights are large. Dropout probability was set at 0.5 for both layers and L2 regularization had a rate of 0.002.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input           | 32x32x3 RGB image   	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					  |												|
| Max pooling	    | 2x2 stride, same padding, outputs 14x14x6 				|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16      	  |
| RELU            |                       |
| Max pooling     | 2x2 stride, same padding, outputs 5x5x16          |
| Fully connected	| 400 inputs            |
| RELU            |                       |
| Dropout         | Keep probability 0.5  |
| Fully connected | 120 inputs            |
| RELU            |                       |
| Dropout         | Keep probability 0.5  |
| Fully connected | 84 inputs and final output of 43 logits           |
| Softmax				  |              					|   |



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the fifteenth cell of the Ipython notebook, with some additional relevant code in the previous three code cells, including an evaluation function for measuring accuracy, placeholders and constants, and optimization settings.

The model was trained with the Adam optimizer due to its ability to use momentum and decay the learning rate dynamically. This makes sure that the training operation does not get stuck in local minima, nor does it overshoot local minima if the learning rate is too high.

The batch size chosen was 256, after playing around with how high it could go and discovering that too high a batch size would result in poor training performance (low training accuracy, with little change). Initially, the network was trained for 100 epochs and later for 150 epochs, until some changes resulted in a performance increase and the number of epochs was dropped to 60 for a reasonable accuracy result.

A low learning rate of 0.0003 was found to be the optimal choice in fast training, without waiting too long for the Adam optimizer to reduce the rate as accuracy went up and loss went down. L2 regularization was set at 0.002, and dropout probability at 0.5.

To observe training, batch accuracy was used on the training set and for the validation accuracy, the full validation set was used. The accuracies were plotted against the epoch numbers to show progress over time.

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...
