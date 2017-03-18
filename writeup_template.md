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
[image4]: ./writeup/01speed30_2.jpg "Traffic Sign 1"
[image5]: ./writeup/13yield.jpg "Traffic Sign 2"
[image6]: ./writeup/25roadWork.jpg "Traffic Sign 3"
[image7]: ./writeup/28children.jpg "Traffic Sign 4"
[image8]: ./writeup/01speed30.jpg "Traffic Sign 5"
[image9]: ./writeup/16weightLimit.jpg "Traffic Sign 6"
[image10]: ./writeup/27pedestrian2.jpg "Traffic Sign 7"
[image11]: ./writeup/29bicycles.jpg "Traffic Sign 8"
[image12]: ./writeup/13yield2.jpg "Traffic Sign 9"
[image13]: ./writeup/17noEntry.jpg "Traffic Sign 10"
[image14]: ./writeup/27pedestrian.jpg "Traffic Sign 11"
[image15]: ./writeup/36aheadRight.jpg "Traffic Sign 12"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

This is the writeup/README and here is a link to my [project code](https://github.com/davinderc/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_ClassifierGPUTrainedFinal.ipynb)
This project was undertaken as part of a three-term course in artificial intelligence and computer vision for self-driving cars, offered by Udacity through their online platform.

###Files for submission
The files for submission included in the github link include the Traffic_Sign_Classifier.ipynb notebook (with all questions answered and all code cells executed and displaying output), an html export of the project with the name report.html, additional images used in the ./writeup/ folder, and this writeup markdown file.

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

The code for training the model is located in the fifteenth cell of the Ipython notebook, with some additional relevant code in the previous three code cells, including an evaluation function for measuring accuracy, placeholders and constants to be used during training, and optimization settings to enhance training.

The model was trained with the Adam optimizer due to its ability to use momentum and decay the learning rate dynamically. This makes sure that the training operation does not get stuck in local minima, nor does it overshoot local minima if the learning rate is too high.

The batch size chosen was 256, after playing around with how high it could go and discovering that too high a batch size would result in poor training performance (low training accuracy, with little change). This is due to how the gradient is calculated, since a larger batch size does not necessarily result in a better gradient direction. In fact, a smaller batch size means that the a smaller step will be taken, but that step will likely be more accurate, since the weights will be updated more frequently than with a larger batch size. Initially, the network was trained for 100 epochs and later for 150 epochs, until some tweaking of parameters resulted in a performance increase and the number of epochs was dropped to 60 for a reasonable accuracy result.

A low learning rate of 0.0003 was found to be the optimal choice in fast training, without waiting too long for the Adam optimizer to reduce the rate as accuracy went up and loss went down. L2 regularization was set at 0.002, and dropout probability at 0.5. These were chosen as the best values that increased training and validation set accuracies.

To observe training, batch accuracy was used on the training set and for the validation accuracy, the full validation set was used. The accuracies were plotted against the epoch numbers to show progress over time. The training accuracy was initially calculated on the whole set, until my mentor noted that using batch accuracy made it easier to see if my model was improving. However, I did not change this back to the full set before my final run and ended up leaving it this way since I had good results, and had noted before that the full set accuracy seemed to be very close to the batch accuracy in previous runs.



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 14th code cell of the Ipython notebook. The evaluate function simply takes a dataset, splits it into batches, calculates accuracies on those batches, and then returns a total accuracy based on the batches used. In order to calculate test set accuracy, the evaluate function's code was repeated in code cell 17, along with additional code in cells 16 and 18, due to some difficulties I had evaluating the test set. The evaluation on the test set was run more than once, but not with the intent to improve accuracy by modifying the model. This was done only due to some difficulties in debugging the saving and loading of the model and its weights. Saving was only done once after a training run was completed, but I did not understand too well how to restore the model and its weights, so I had to look up documentation and do a little trial and error to find out the proper way to do this, before I was consistently showing the original test set accuracy, confirming that I had managed to reload the original weights.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96%
* test set accuracy of 95.4%

* What was the first architecture that was tried and why was it chosen?
The first architecture that was tried was the LeNet architecture with some simple modifications to accommodate the different dataset. It was chosen because I understood how it worked from the previous lab exercise and expected to have some difficulties using TensorFlow due to the unfamiliar programming paradigm as well as new libraries and functions. I thought that once I had everything working I could modify things a bit and see if I could improve on what I had.

* What were some problems with the initial architecture?
The initial architecture had low accuracy that didn't seem to improve initially. Other than that, since it had worked for the MNIST dataset, it didn't present any difficult problems.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
One of the most important adjustments I made was to add dropout on the first two fully connected layers. This improved accuracy, but I decided to also add L2 regularization on all three fully connected layers in an attempt to increase these further. Eventually I found that I was evaluating with the same dropout factor I used during training, and this was artificially pulling down my accuracy. Once I adjusted this (after having tuned parameters extensively), I saw my accuracy bump up to a number that left me satisfied enough to evaluate the test set.

* Which parameters were tuned? How were they adjusted and why?
The following parameters were tuned: learning rate, batch size, epochs, L2 regularization rate, and dropout. All of them were tuned individually, increasing and decreasing them in accordance with accuracy increase. That is, if increasing a parameter caused an increase in steady-state accuracy after a given number of epochs, the parameter was increased more. Otherwise, the parameter was decreased slightly until I was satisfied that accuracy was not improving any more. Each parameter was usually only tuned once, even if there was a possibility that the tuning of others might make it possible to retune (and thus increase accuracy) a previously tuned parameter.

Learning rate was tuned because it was clear from watching how accuracy oscillated towards high epoch numbers (70). It seemed that once accuracy was that high, it didn't increase any more, so I decided to explicitly decay the learning rate, which seemed to improve accuracy, even if not significantly. However, I later found out that the Adam optimizer did this on its own and removed the explicit decay. I nearly left the learning rate at 0.0001, until someone suggested it was slow, and I tried 0.0003 with success, and I left it there.

Initially, as I was training on a GPU cloud service, I increased the batch size all the way up to 4096, mostly thinking that it would speed up training, since I was processing more data at a time, and the instance I was running could handle all that data. However, I later found out that this was capping my training accuracy at around 90%, and I ended up altering it back down to 256, before trying out batch sizes of 512 and 128, only to find out that neither of those offered significant improvements over 256. I chose to use multiples of two simply out of ease of choice.

The epoch number was adjusted as a function of accuracy. If I saw that validation accuracy was still increasing towards the end of my training, I increased the epoch number until I saw that it no longer changed much. This was helpful when I was explicitly decaying the learning rate in accordance with epoch number. However, at one point, one of the other parameters increased training speed so that I was hitting 90% validation accuracy within 10 epochs, and I decided that going much further beyond 80 epochs was unnecessary. I ended up choosing to train up to 60 epochs.

L2 regularization rate was adjusted very little. I had managed to set it very close to the optimal value, and found that accuracy was only dropping when I moved it, so I ended up leaving it at 0.002.

Dropout played a big role in this model as it was one of the first things that helped my model move further. I initially thought I would maintain the keep probability high, somewhere around 75%, but decided against it when I couldn't see my model's validation and training accuracy move much more than a small oscillation. I figured it couldn't hurt to constantly kill some weights and that if anything, I might move out of local minima by doing this. I left the keep probability at 50%.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
A convolution layer works well with this problem because it is capable of picking up edges, and eventually connect them into shapes, which is a good strategy when trying to detect traffic signs, which are often differentiated in categories by their shapes and then further by the shapes of the symbols on them.

Dropout layers make sure that the model does not overfit to a training dataset. That is, if a training set is skewed in some manner, as was the case in this problem, with the unbalanced sample numbers for the different classes. Since an unbalanced dataset might cause a model to predict the classes with higher numbers of samples, dropout might influence this by preventing the same class from being detected by too many pathways, and increasing the chances that an underrepresented class can be detected, since it probably has much fewer pathways through the layers that will detect it.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
I chose the LeNet architecture because I thought that if it was able to detect hand-drawn digits, it would probably not be difficult to adjust it to work on standardized signs, even if the signs might suffer from light noise or something else obscuring their view.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The final model's training accuracy was somewhat surprising, as it showed 100% accuracy, which would seem to suggest that it was perfectly fit to the training dataset. However, since the validation accuracy (96.0%) did not decrease, other than a slight oscillation, I took this to mean that the model was just really well trained for this application. The test set accuracy of 95.4% was only 0.6% below the validation accuracy, and this dataset was never used while the network was being modified, I felt that this meant the model had generalized very well for the dataset I had used for training.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Below are the German traffic signs that I found on the web. Since the images I found had multiple signs, I decided to crop out and scale 12 signs, in order to have a larger personal test set. These images might be difficult to classify in general, because I realized afterwards that the cropping and scaling might be different from that done for the training set.

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]

![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]

![alt text][image12] ![alt text][image13] ![alt text][image14] ![alt text][image15]

The first image might be difficult to classify because it was taken from slightly off-center of the sign, causing a skew and there are parts of other signs appearing in the image.

The second image should be easily classified, since it's fairly clear what it shows, although the different lighting might cause a slight difficulty.

The third image could be difficult to classify due to the sign's small size in the image and part of another sign obscuring part of it.

The fourth image should not be very difficult to classify, despite part of another sign showing up beneath it.

The fifth image might be difficult to classify due to the stickers that are covering parts of the sign.

The sixth image would be impossible to classify, since it wasn't in the dataset. I only discovered later that the class of sign I thought was in it was represented by a different sign in the training set.

The seventh image might be difficult to classify because it contains more than one sign in it and one of them is partially obscured.

The eighth image should be easy to classify, although the thin lines in the outline of the bicycle might prove more difficult at the resolution used.

The ninth image should be difficult to classify, since the photo was taken at an extreme angle to the normal vector of the sign, and the dataset was not augmented with this in mind.

The tenth image shows a sign skewed in the diagonal, which will make it difficult to classify.

The eleventh image should be fairly easy to classify, as the sign is clear and against a well-contrasted background.

The twelfth and last image will probably be more difficult to classify due to the sign's size in the image.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twenty-second cell of the Ipython notebook.

The results of the prediction were displayed beneath the 22nd code cell as an array of images with the respective labels and predictions shown. The way they are shown in the html version of the notebook makes it significantly easier to visualize than here.

The model was able to correctly guess 3 of the 12 traffic signs, which gives an accuracy of 25%. This does not compare favorably to the accuracy on the test set of 95.4%. This is very likely due to the way the internet images were selected and cropped/scaled. I believe that having a standard for cropping and scaling would increase that accuracy significantly. In addition, choosing images using the same criteria as those used for selecting the test set would improve the accuracy as well, since a lot of the images I chose had either a skew, some noise in the image, or some obscuring of the sign. The images that were correctly predicted seemed to resemble those found in the training set, which would explain the correct classification.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model with certainty percentages is located in the 25th cell of the Ipython notebook.

The predictions and certainty probabilities are displayed in the html version of the Ipython notebook in an easy to understand manner and significantly better than what would be possible here.

The correct predictions had very high probabilities (all of them near or above 90%). The incorrect predictions usually either had probabilities around 50% or much lower. There was one case, in the 11th image, in which the image was inaccurately classified and had a 92% certainty. This was an image of a pedestrian crossing, classified as road work, which, due to the blurriness of the image, could be part of the reason the image was misclassified.

![alt text][image4]

This sign was classified correctly with 94% certainty. This was somewhat surprising, considering the sign was skewed and shadowed.

![alt text][image5]

This sign was also classified correctly with 100% certainty, which wasn't surprising considering the sign was clearly visible and undistorted.

![alt text][image6]

This sign was also classified correctly with 88% certainty. The sign was mostly visible, other than being slightly obscured.

![alt text][image7]

This sign was not classified correctly, nor was the correct classification in the top 5 predictions. However, none of the incorrect predictions were above 50% certainty. This may be due to the sign being shifted up in the image.

![alt text][image8]

This sign was classified incorrectly as well, with none of the predictions being correct. However, in this case, the noise in the image as well as the slight shift in the sign's position and the scaling of it might account for the incorrect predictions.

![alt text][image9]

This sign was classified incorrectly, but it is not surprising considering this sign was not in the dataset (I did not realize the class I gave this sign was for a different sign).

![alt text][image10]

This image was classified incorrectly, which might be due to the fact that there were three signs in the image. This was done on purpose to see how the classifier would behave in this situation, considering the dataset only had images with individual signs.

![alt text][image11]

This image was classified incorrectly, which is no doubt due to the different shape of the sign. The sign in the dataset is triangular, while this sign is circular. When choosing internet images, I didn't realize the signs were different.

![alt text][image12]

This image was not classified correctly, although the correct prediction was in the the top 5 predictions with a probability of 2%, which is very low. This is probably due to the fact that the sign is squeezed along the x-axis and the model was not trained on photos of signs taken far from the normal vector of the sign.

![alt text][image13]

This image was also not classified correctly, but had a similar situation to the previous image, its correct prediction was in the top 5. This is probably due to a similar reason as the previous image, as well as the fact that there was another smaller different sign visible in the picture.

![alt text][image14]

Surprisingly, this image was not classified correctly, although it appeared to be an easy to classify image. It is unclear why the classifier could not correctly classify this image, and even provided a high certainty prediction (92%) on a class that is similar, but clearly not the same.

![alt text][image15]

This image was also not classified correctly and the predictions all had low certainties, which is probably due to the fact that there was more than one sign in the image.
