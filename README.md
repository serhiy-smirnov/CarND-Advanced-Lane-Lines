# **Advanced Lane Finding** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="output_images\result_test4.jpg " alt="Detection result in complex light conditions" />

Overview
---

In the first project of this program I used Python and OpenCV to detect lane lines in discrete image files and video clips. I built simple detection pipline based on the Canny Edge Detection and Hough Transform followed by linear regression approximation to find average lines. This gives good results only on the straight segments of the road in even lighting/colour conditions of the road surface.

To cover more complicated cases like lanes with some curvature, different lighting conditions or slight variations in the road surface colour, I will take more complicated approach in this project.
One step further, to make the SW more usable for real self-driving car use cases, I will calculate real-world curvature of the detected lane, lane width and position of the car within the lane. To achieve this, I will start from the camera calibration to ensure that the images analyzed by detection algorithms are free of distortions introduced by the camera lenses.


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
[image3]: ./output_images/thresholded_gradient.jpg "Binary image based on the gradient threshold"
[image4]: ./output_images/thresholded_colour.jpg "Binary image based on the colour threshold"
[image5]: ./output_images/region_of_interest_straight_lines1_small.jpg "Warp example - source image"
[image6]: ./output_images/warped_colour_with_roi_straight_lines1_small.jpg "Warp example - result image"
[image7]: ./output_images/lane_detection1.jpg "Lane detection in the warped binary image"
[image8]: ./output_images/lane_detection2.jpg "Lane detection in the warped binary image"
[image9]: ./output_images/result.jpg "Lane detection rendered back into the perspective source image"
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

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. Corresponding structures (`mtx` and `dist`) are saved in the Camera class and in the cache file as described above. I apply distortion correction using the `cv2.undistort()` function in the class method `undistort_image()`, that is used by the lane detection pipeline. In the method `undistort_test_images()` I use the same parameters for the camera calibration images to check if the calibration was successful: 

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

`gradient_threshold()` method aggregates all the above using ranges of
* (50, 100) for directional gradient in x/y and gradient magnitude
* (0.7, 1.3) for gradient direction thresholding.

Example of the binary image created with gradient thresholding can be seen below:
![alt text][image3]
More examples can be found in the `output_images` folder as `thresholded_gradient_*.jpg` files.

In the method `colour_threshold()`, code section #5, I implement my colour thresholding approach.
As a first step I convert image to HLS colour space to align processing more closely with the way human vision perceives color attributes.
For the next step I tried different approaches, e.g. calculation of gradients on hue and saturation channels with separate thresholds, but the best result seems to be produced by thresholded gradient on luminance combined with thresholded saturation channel.

Example of the binary image created with colour thresholding can be seen below:
![alt text][image4]
More examples can be found in the `output_images` folder as `thresholded_colour_*.jpg` files.

As a final step, `combined_threshold()` method in code section #6 combines both gradient and colour binary images together, which in many cases gives better results for lane detection.
Using combined approach you can adapt binary image generation to individual lighting conditions or colour of the road markings (potential improvement point for this project).

Examples of the combined binary images can be found in the `output_images` folder as `thresholded_combined_*.jpg` files.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transformation is arranged as a class `PerspectiveTransform()` in code section #7. During initialization it calculates source and destination points for transformation and provides two main functions - to warp the image to the birds-eye-view and back to perspective.

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

To detect single line of the lane road markings, I implemented the class `Line()` in code section #8.

It incorporates all the required state variables, like detection history, configuration of detection algorithms and current detection results.
Line detection is encapsulated into the method `find_line()`. It implements two possible approaches for activated pixels search, fits polynomial through the detected pixels of the image and updates all the internal status structures based on the line detection results.

Two main methods used to find activated pixels of the image are:
* search for line pixels in the image using sliding window

	It is used to detect the line in the image for the first time.
Sliding window approach is implemented in the method called `find_lane_pixels_histohramm_sliding_window()`. In this approach we run through the image from the bottom to the top using windows of specified width and height and collect all the activated pixels of the binary image that fall into these windows. Result of this search can be seen in the right image below. Search windows are visualized as green rectangles.
To find a good place for starting with the search (x position of the first window) I calculate histogram of the lower part of the image and take x position of the maximum on the left or on the right side, depending on which line we need.

	Some additional checks are implemented for the maximum detection, e.g. if the maximum value is smaller than a specified threshold, it is rejected and the line detection is skipped.

	Position of each next sliding window is adjusted based on the mean value of x coordinate of all the activated pixels found in the current window.
![alt text][image7]
More examples can be found in the `output_images` folder as `lane_detection_*.jpg` files.


* search for line pixels around previously detected polynomial line

	It is implemented in the method called `find_lane_pixels_search_around_polynomial()` and is used to speed up line detection in sub-sequent images, e.g. in a video clip.
For this method to work, we need coefficients of the previously detected polynomial line from the previous frame. Then we can collect all the activated pixels in a specified distance to the left and to the right of this line. We virtually draw two additional lines on the left and on the right from the previously detected line and check all the pixels in-between. Seacrh area is visualized in green and activated pixels found - in red/blue on the picture below.
![alt text][image8]

After the activated pixels are collected, I run `fit_polynomial()` method of the class that uses `np.polyfit()` function to find polynomial coefficients that minimize squared error in the second order. If the coefficients are found, they are added to the detection history that is used to stabilize detection result.

On top, this class also implements methods for visualization of the detected polynomial lines `visualize_lines()` and for managing of the detection history `reset_detection()`.

Curvature and base position calculation is implemented in the method `calculate_radius_and_base_pos()` and described in the next part.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Curvature calculation in the method `calculate_radius_and_base_pos()` of the `Line()` class (code section #8), uses approach described [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php).

Given that we fit ![equation1](https://latex.codecogs.com/gif.download?f%28y%29) rather than ![equation2](https://latex.codecogs.com/gif.download?f%28x%29), radius of curvature can be calculated as:

![equation4](https://latex.codecogs.com/gif.download?%5Clarge%20R_%7Bcurve%7D%3D%5Cfrac%7B%281+%28%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20y%7D%29%5E2%29%5E%7B3/2%7D%7D%7B%5Cleft%20%7C%20%5Cfrac%7B%5Cpartial%5E2%20x%7D%7B%5Cpartial%20y%5E2%7D%20%5Cright%20%7C%7D)

For the ![equation4](https://latex.codecogs.com/gif.download?y) I take the value closest to the car:
```python
y_eval = np.max(self.ally)
``` 

Base position calculation is done in the same function and assumes that the camera is mounted in the middle of the car, therefore the middle of the image is equal to the middle point of the car:
```python
self.line_base_pos = abs(midpoint - line_point) * self.xm_per_pix
```

Both lane curvature and base position values are calculated in meters using conversion coefficients from pixels to meters defined based on the approximate real dimentions of the warped image:
```python
self.ym_per_pix = 27/720 # meters per pixel in y dimension
self.xm_per_pix = 3.7/1000 # meters per pixel in x dimension
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented complete lane detection and visualization in the class `Lane()` in code section #9.

It uses two instances of the `Line()` class - for the left and for the right lines.

It also contains additional configuration parameters like default width of the lane and some thresholds for validity checks of the lane detection.
Main entry point is the method `find_lane()` that tries to find both left and right lines of the lane and then automatically runs validation of the results. During this validation step, left and right line detections are checked if their curvatures make sense and are not too different. In the worst case, one of the lines with lower detection confidence can be rejected.

During visualization in the method `generate_display_lane()`, polygon covering space between detected polynomials within the region of interest is generated and filled in with the green colour. If one of the lines is not detected, the other line is used as a base and the missing line is assumed at the previously detected lane width.

Additional method `display_lane_properties()` can render the following lane detection results directly into the picture:
* lane width
* lane curvature
* position of the car within the lane

To complete description of my image processing pipeline I want to mention the class `ImageProcessor()` in the code section #10, which contains all the necessary instances of camera, transformation and lane objects and also keeps configuration parameters like region of interest and debug output modes.
It provides `process_image()` method that can be used to process pictures from single test files as well as frames of a video sequence.

Here is an example of my result on a test image:

![alt text][image9]

More examples can be found in the `output_images` folder as `result_*.jpg` files.


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my implementation of the project I took approaches described in the lessons 6-8 of the program. It includes camera calibration fundamentals, working with gradients and different colour spaces, and some advanced computer vision techniques.
This scope of knowledge and methods is pretty much sufficient to handle standard situations like the main project video (see above) with slight curvature, moderately varying lighting conditions, colours of the lane markings and road surface.

However, when the situations include high-contrast areas due to different road surfaces, extreme highlights/shadows or too sharp curves - more sophisticated approaches are needed to handle such situations.

As a result, current implementation experiences some difficulties to handle `challenge_video.mp4` (see [the output result](./output_challenge_video.mp4)), while the `harder_challenge_video.mp4` is more tricky and confuses my current implementation even more (see [here](./output_harder_challenge_video.mp4))

To improve for handling of the challenge video I would see the following changes needed:
* improve gradients detection to better track situations when the line marking is assumed where it's only a change in the colour of the road surface
* implement better estimation of the middle point of the car and choosing the right spots on the histogramm as starting points for sliding window search. It can happen that the maximum on the histogram is not always true start of the line

To further improve for handling of the harder challenge video I can think about some additional approaches:
* cut the image into regions according to the lighting conditions (extreme light or extreme dark) and apply additional smoothing processing step to bring lighting conditions to some kind of average
* implement dynamic region of interest to account for steep curves. This will however require more complicated calculations of the transformation from pixels to meters.
* ideally, we need estimation of the road surface pose in relation to the car, since in the steep curves it may not be even, that can lead to errors in measurements
