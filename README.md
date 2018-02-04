# Vehicle Detection

In this project, my goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup ](https://github.com/sbatururimi/Vehicle-Detection/blob/master/writeup.md) for this project for more details.  


The Project
---

The goals / steps of this project are the following:

* Performing a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, we can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps we need to normalize our features and randomize a selection for training and testing.
* Implementing a sliding-window technique and use your trained classifier to search for vehicles in images.
* Runnin our pipeline on a video stream (starting with the test_video.mp4 and later implementing on full project_video.mp4) and creating a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimating a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing the pipeline on single frames are located in the `test_images` folder. The video called `project_video.mp4` is the video our pipeline should work well on. 

Important
---

To create an Anaconda environment (tested on MacOS), run the following command in the terminal from the root folder of the project:
```
 conda env create -f environment.yaml
 ```

 PS: for linux anaconda settings, you can find it here
 https://github.com/sbatururimi/Car-Behavioral-Cloning/blob/master/environment_ubuntu.yaml

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/sbatururimi/Vehicle-Detection/blob/master/LICENSE.md)

