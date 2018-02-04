## Writeup 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar_example.png
[image2]: ./output_images/hog_visualization.png
[image3]: ./output_images/spatially_binned_feature.png
[image4]: ./output_images/histogram_colors.png
[image5]: ./output_images/normalized_features.png
[image6]: ./output_images/hog_subsampling_window_search.png
[image7]: ./output_images/car_positions.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. How (and identify where in your code) I extracted HOG features from the training images.

The code for this step is contained in the 13th code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]


#### 2. Final choice of HOG parameters.

After a lot of experiments, I finally ended up using the 'YCrCb' color space and HOG parameters of `orientations=12` as it reepresents the number of orientation bins that the gradient information will be split up into in the histogram. Typical values are between 6 and 12 bins and setting it to 12 show good results.  `pixels_per_cell` and  `cells_per_block` remain same as in the case of the gray color space.

What is also significant here, is that after trying to get HOG features only for 1 channel, I ended up by considering 3 channels separatelly and then concatenating them together to form one big HOG features array.
This is visible through lines 14-16 of *find_cars* starting from the section 'Improved windows search'

During my experiments, the HLS color space and using the Hue channel only was also good.


#### 3. How I trained a classifier using my selected HOG features and color features.

I trained a linear SVM using a linear support vector machine. This can be found under the section *Train a classifier*.
In order to improve the number of extracted features from cars and non-cars images I have also used spatially binned color and histograms of color.

For the spatially binned color as features extraction, I went way down to 32x32 pixel resolutions, at such scale the car itself is still clearly identifiable by eye, and this means that the relevant features are still preserved at this resolution. The codecan be found under the section *Spatial binning of colors* Here is an example of the plot of the extracted features using the spatially binned color features extraction

![alt text][image3]

I have also used histograms of color to extract additional features (section *Histograms of colors*). I computed the histogram of the color channels separately and then Concatenate the histograms into a single feature vector.

![alt text][image4]

When combining all features together (HOG, spatially binned color, histograms of color), we can observe a difference in magnitude between these features. Because of that, we perform a normalization step by using StandardScaler from the Python's sklearn package. This is what I obtained after the normalization step:

![alt text][image5]


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

During my expirements, I decided to search windows starting from the bottom of the image till the half of its height.
I started with a window size of 96x96 and an overlapping of 30% on the x-axis and 50% on the y-axis, extracting all features from each window separatelly.

I later improved this window search:
- by appling a Hog Sub-sampling Window Search, that is to extract hog features once and then can be sub-sampled to get all of its overlaying windows
- by changing the region of interests in y-coordinate in which I was looking for windows to static values, that is

```
ystart = 400
ystop = 656
```
for an image size of 1280 × 720, where 1280 is the width and 720 the height.

- by using a multiscale approach visiblein *detect_vehicle*
- a search window overlap of 75%

![alt text][image6]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I used a Hog Sub-sampling Window Search using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

### Video Implementation

#### 1. Link to your final video output.  
Here's a [link to my video result](./videos_output/test_video.mp4)

Another analyzed [video](./videos_output/project_video.mp4)


#### 2. Some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  I alsoimplemented the idea of utilizing the fact that in the subsequent frames the cars are located at or near the same positions, while false positives are present only for 1-2 frames. I used the simplest way to implement this by using multi-frame accumulated heatmap: just storing the heatmap of the last N frames (N can be 5 or 8) and doing the same thresholding and labelling on the sum of these heatmaps.

```
# Add heat to each box in box list
...
heat = np.zeros_like(out_img[:,:,0]).astype(np.float)
heat = add_heat(heat, box_list)
# Apply threshold to help remove false positives
heat = apply_heat_threshold(heat, 5)
# Visualize the heatmap when displaying    
current_heatmap = np.clip(heat, 0, 255)
history.append(current_heatmap)
# use the accumulated history
heatmap = np.zeros_like(current_heatmap).astype(np.float)
for heat in history:
    heatmap = heatmap + heat
# Find final boxes from heatmap using label function
labels = label(heatmap)
...
```

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
![alt text][image7]

---

### Discussion

I used probably the simplest methods and techniques but at some moment was blocked by the features array size obtained from the cars and non-cars list. 
Instead of being something like ( batch_num x feature_num) and used pdb to debug each step.

I didn't use a mutliscaling approach during the windows search but to be considered. 

One of the approach, I think could be very interesting would be to use the recently released Mask R-CNN to perform an object detection of cars. So, we could use the convolutions + region detections + the proposed pixel mask to detect if a found object is a car or not. 
 

