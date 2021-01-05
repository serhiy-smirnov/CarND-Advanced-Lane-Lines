# **Advanced Lane Finding** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="output_images\result_test4.jpg " alt="Detection result in complex light conditions" />

Overview
---

In the first project of this program I used Python and OpenCV to detect lane lines in discrete image files and video clips. I built simple detection pipline based on the Canny Edge Detection and Hough Transform followed by linear regression approximation to fin average lines. This gives good results only on the straight segments of the road in even lighting/colour conditions of the road surface.
To cover more complicated cases like lanes with some curvature, different lighting conditions or slight variations in the road surface colour, I will take more complicated approach.
Also, one step further, to make the SW more usable for real selfßdriving car use cases, I will calculate real-world curvature of the detected lane, lane width and position of the car within the lane. To achieve this, I will start from the camera calibration to ensure that the images analyzed by detection algorithms are free of distortions introduced by the camera lenses.


The goals / steps of this project:
---

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

By submiting the project I ensure that the results meet the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/571/view)


[//]: # (Image References)

[image1]: ./output_images/calibration.jpg "Undistorted chessboard image"
[image2]: ./output_images/undistorted.jpg "Undstorted image from the lane detection sequence"
[image3]: ./output_images/threholded_gradient.jpg "Binary image based on the gradient threshold"
[image4]: ./output_images/threholded_colour.jpg "Binary image based on the colour threshold"
[image5]: ./output_images/region_of_interest_straight_lines1_small.jpg "Warp example - source image"
[image6]: ./output_images/warped_colour_with_roi_straight_lines1_small.jpg "Warp example - result image"
[image7]: ./output_images/lane_detection.jpg "Lane detection in the warped binary image"
[image8]: ./output_images/result.jpg "Lane detection rendered back into the perspective source image"
[video1]: ./output_project_video.mp4 "Processing result of the project video"

## Rubric Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

It is this file README.md which you're currently reading!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code section #3 of the IPython notebook located in "./P2.ipynb". `(here and below I refer to the section numbers, see the top-level comment line in each Python code cell)`

I implemented a class `Camera`, that contains all the methods needed to calculate calibration parameters of the camera and apply undistortion to the images. To achieve this it can hold calibration parameters during run time. To speed up lane detection runs, camera calibration step is optimized:
* when the script is run first time, it saves detected calibration parameters in the cache file `camera_cal\camera_calibration_parameters.p`
* during next runs, calibration parameters are read from the cached file
* if the cache file is not found, calibration sequence is started automatically

In the `calculate_calibration_parameters()` method I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection using `cv2.findChessboardCorners()`.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. Corresponding structures (`mtx` and `dist`) are saved in the Camera class and in the cache file as described above. I apply distortion correction using the `cv2.undistort()` function in the class method `undistort_image()`, that is used by the lane detection pipeline. In the method `undistort_test_images()` I use the same parameters to the camera calibration images to check if the calibration was successful: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To undistort images used for lane detection, `Camera` class is instantiated and used for initialization of the `ImageProcessor` class in code sections #11 and #12.
In the implementation of the image processing pipeline, `camera.undistort_image()` method is used as the first step to prepare the image for further processing, see code section #10, method `process_image()`.

An example of undstorted image from the lane detection sequence can be seen here:
![alt text][image2]
Notice differences on the ego car's hood and white car on the right.
More examples can be found in the `output_images` folder as `undistorted*.jpg` files.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.

Code section #4 contains gradient calculation and thresholding methods:
* `abs_sobel_threshold()` for directional gradient in x or y
* `mag_threshold()` for gradient magnitude
* `dir_threshold()` for gradient direction

`gradient_threshold()` method aggregates all the above using range of (50, 100) for directional gradient in x/y and gradient magnitude and (0.7, 1.3) for gradient direction thresholding.

Example of the binary image created with gradient thresholding can be seen below:
![alt text][image3]
More examples can be found in the `output_images` folder as `threholded_gradient_*.jpg` files.

In the method `colour_threshold()`, code section #5, I implement my colour thresholding approach.
As a first step I convert image to HLS colour space to align processing more closely with the way human vision perceives color attributes.
For the next step I tried different approaches, e.g. calculation of gradients on hue and saturation channels with separate thresholds, but the best result seems to be produced by thresholded gradient on luminance combined with thresholded saturation channel.

Example of the binary image created with colour thresholding can be seen below:
![alt text][image4]
More examples can be found in the `output_images` folder as `threholded_colour_*.jpg` files.

As a final step, `combined_threshold()` method in code section #6 combines both gradient and colour binary images together, which in many cases gives better results for lane detection.
Using combined approach you can adapt binary image generation to individual lighting conditions or colour of the road markings (improvement point for this project)

Examples of the combined binary images can be found in the `output_images` folder as `threholded_combined_*.jpg` files.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transformation is arranged as a class `PerspectiveTransform()`. During initialization it calculates source and destination points for transformation and provides two main functions - to warp the image to the birds-eye-view and back to perspective.

I chose the hardcode the source and destination points based on the input image size in the following manner:

```python
near_distance = ysize-1
far_distance = ysize*0.645

left_bottom = [0+70, near_distance]
right_bottom = [xsize-1-70, near_distance]

left_apex = [xsize/2-89, far_distance]
right_apex = [xsize/2-1+89, far_distance]

self.src = np.float32(
						[left_bottom,
						left_apex,
						right_apex,
						right_bottom])
self.dst = np.float32(
						[[0, ysize-1],
						[0, 0],
						[xsize-1, 0],
						[xsize-1, ysize-1]])
```

For the image size of 1280x720, this resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 70,   719     | 0, 719        | 
| 551,  464     | 0, 0          |
| 728,  464     | 1279, 0       |
| 1209, 719     | 1279, 719     |

I verified that my perspective transform was working as expected by drawing the `src`  points onto a test image to verify that the lines appear parallel in the warped image.

| Input image   | Warped image  | 
|:-------------:|:-------------:| 
| ![alt text][image5] | ![alt text][image6] | 
 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
