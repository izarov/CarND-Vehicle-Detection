##Vehicle Detection

####The goal of this Udacity SDCND project is to identify cars on the road from a video using traditional computer vision techniques. I use [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) to generate a feature vector and an ensemble of [SVM](https://en.wikipedia.org/wiki/Support_vector_machine) and [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) for classification.
---

**Vehicle Detection Project**

The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Apply a color transform and prepend binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize feature vectors and split data into train and test subsets.
* Fit one or more classifiers to the training data.
* Implement a sliding-window technique and use the trained classifiers to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Track centroids of detected vehicles and estimate a bounding box.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg

[pipeline1]: ./output_images/pipeline1.png
[pipeline2]: ./output_images/pipeline2.png
[pipeline3]: ./output_images/pipeline3.png
[pipeline4]: ./output_images/pipeline4.png
[pipeline5]: ./output_images/pipeline5.png
[video1]: ./project_video.mp4

### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

In this section I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The feature vector generated for each image is a concatenation of the following vectors:

* Spatial features: a 1D vector representation of the raw pixel values of the image, after resizing to 32x32 using linear interpolation
* Color histogram: a 1D vector containing a histogram for each color channel
* HOG: histogram of gradients feature vector for each color channel

The code for all of the above can be found in features.py or in the Detection.ipynb notebook. The HOG code is on lines 29-50. The complete feature vector is composed in feature_extraction.py.

First, I read in all the `vehicle` and `non-vehicle` training images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I explored the parameter space manually. Ultimately I decided to use the defaults for orientation, pixels per cell and cells per block as shown in the [HOG skimage documentation](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog). The defaults seem to strike a good combination of speed and granularity. I ran HOG separately on all color channels to provide richer information to the classifiers. 

* Orientation bins: 9
* Size of a cell: (8, 8) px
* Cells per block: 2
* Color space: YCrCb

Configurable parameters can be found in config.py

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I tried several classifiers. I got very good test accuracy using linear SVM, as well as SVM with non-linear kernels. I also tried decision trees including random forests and AdaBoost with decision trees. All classifiers performed satisfactory on the test set. I chose linear SVM for its simplicity, speed and high accuracy, as well as AdaBoost for its speed and theoretical approach. 

All classifiers that I trained suffered from overfitting, which was evident from the many false positives when used on the project video. The video and frames from it were not used in training, which showed that the combination of the approach taken and the training data set does not generalize very well. Data augmentation did not result in much improvement. To battle overfitting I decided to use an ensemble of my chosen classifiers, with both classifiers having to agree before returning a positive detection. Using a larger and less correlated training data set is likely to lead to further improvement.

The code can be found in classifiers.py. Before fitting the classifiers the feature vectors are normalized. I used sklearn's RobustScaler. Unlike the StandardScaler, which uses the mean and variance sample statistics, RobustScaler removes the median and rescales according to the quantile range, thus being more robust to outliers. Each feature dimension in the feature vector is normalized indepedently.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search was implemented following the approach (and reusing the code) from the SDCND module. A window of a fixed size is moved across the whole searchable region of an image, with a certain overlap with the previous window. For the video pipeline I decided to use a (112, 112) size window as that struck a good balance between detection accuracy and search speed. I used an overlap of 60% as using less overlap with the relatively large window size can lead to many missed detections.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I spent most of my time looking for parameters that minimized false positives and maximized false negatives without decreasing the speed. The pipeline was prone to misclassify regions of the road that had higher gradient. 

I used spatial, histogram and HOG features on all 3 channels in YCrCb color space. I increased window size to provide more information to the classifier. I also limited the search space in x and y coordinates, which eliminated a large section of high gradient portions of the video frames.

I tried using an SVM classifier with a probabilistic output, but that was unfeasibly slow to train and predict.

Ultimately most of the improvement with respect to false positives and negatives came from aggregating detection results over a series of video frames in a heatmap and tracking the centroids of detected objects.

Below are some example images. The second and 3rd image show a false positive and false negative respectively.
![pipeline1]
![pipeline2]
![pipeline3] 
![pipeline4]
![pipeline5]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video.

Here's a [link to my video result](./submission.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I assumed each blob corresponded to a vehicle and got a list bounding boxes to cover the area of each blob detected.

To further improve the robustness of the detection pipeline I then started tracking the boxes and centroid of detected objects. Once an object was detected from the thresholded heatmap it got assigned a centroid object. This is either an existing object if one is found within 100 px distance from the current detection (i.e., the new detection is considered part of a previously detected object) or a new centroid object.

Detected boxes were drawn on a frame only if they their centroid object was marked as active in at least 3 frames (not necessarily in order). I kept a history of detections and if an object did not appear for a large number of frames the centroid object was deleted.

The code for the heatmap can be found in heatmap.py, centroid objects in centroid.py and the detection process in detection.py. To generate the final video run `python main.py`

The object tracking technique helped smooth predictions and sufficiently reduced the number of reported false detections.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The detection pipeline suffers from a large number of false positives. I used techniques like heatmap with thresholding and object tracking to alleviate the problem and managed to get a pipeline that works correctly on the project video. However, I would say that I find the accuracy and robustness of the pipeline unsatisfying.

I think that a deep learning approach will be much more successful here and result in a much higher accuracy detection pipeline. There are multiple architectures known to deliver outstanding object detection results across a large number of classes, such as [R-CNN](https://arxiv.org/abs/1311.2524) or a similar but faster approach such as [Faster R-CNN](https://arxiv.org/abs/1506.01497). For very fast object detection, such as real-time detection in a car, I will probably use [YOLO](https://arxiv.org/abs/1506.02640), which looks at the whole image once, unlike other object detection pipelines which use a sliding window.

The project rubric calls for a traditional computer vision approach with as a hand engineered feature vector. I feel that given the advances in deep learning any hand engineered feature vector is going to be more time consuming to research, difficult to get right and the result inferior to the state of the art deep learning architectures. However, object detection is a critical task in self-driving cars and it is essential that whatever the pipeline it can be debugged and understood to a sufficient level of detail. I feel that an ensemble of deep learning model(s) in probabilistic combination with a traditional computer vision pipeline would probably be the most balanced and effective approach to car detection.
