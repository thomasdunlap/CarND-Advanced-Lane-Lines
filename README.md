# **Advanced Lane Finding Project**
![Image from final video][vid_img]
----
## The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undist]: ./output_images/undist.png "Undistorted"
[undist_road]: ./output_images/undist_road.png "Road Transformed"
[orig_sobel_x]: ./output_images/orig_sobel_x.png "Sobel X-axis Transformation"
[hls_thresh]: ./output_images/orig_hls_thresh.png "Histogram Threshold Image"
[final_thresh]: ./output_images/final_thresh.png "Final Threshold Example"
[grad_dir_comb_thresh]: ./output_images/grad_dir_comb_thresh.png "Gradient Directions"
[sobel_y_thres_mag]: ./output_images/sobel_y_thres_mag.png "Sobel Y-axis and  Magnitude Images"
[warped_mask]: ./output_images/warped_mask.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[vid_img]: ./output_images/video_img.png "Output"
[video1]: ./project_video.mp4 "Video"


----

### Writeup / README

#### 1. Provide a README that covers all the [rubric](https://review.udacity.com/#!/rubrics/571/view) points and how you addressed each one.   

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook [Advanced-Lane-Lines.ipynb](https://github.com/thomasdunlap/CarND-Advanced-Lane-Lines/blob/master/Advanced-Lane-Lines.ipynb).  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the three-dimensional space of the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z = 0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Distorted and undistorted checkerboard comparison.][undist]

## Pipeline

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Road image and undistorted road image.][undist_road]

I created an undistort function (IPython code block 4):

```python
def undistort(image, visualize=False):
    """
    Returns RGB numpy image, ret, mtx, and undistorted image, and can also do comparative visualization.
    """
    image = mpimg.imread(image)
    h, w = image.shape[:2]
    #objpoints, imgpoints = calibration_points()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

    # undistort
    dst = cv2.undistort(image, mtx, dist, None, mtx)

    if visualize:
        compare_images(image, dst, "Original Image", "Undistorted Image", grayscale=False)

    return image, mtx, dist, dst
```

It reads the image file path, converting the image to RGB using the `mpimg.imread()` function.  Then, using the OpenCV function `cv2.calibrateCamera()`, we pass in the `objpoints` and `imgpoints` from out ``

#### 2. Describe how you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image ().  Here's a example of my output for this step.  (note: this is not actually from one of the test images)

![Original image and sobel x-axis transformed image.][orig_sobel_x]

Here is a comparison between the original image, and a Sobel x-axis transformed image.  Sobel transformations basically return a binary image - if a specificed number of gradients are all in the same direction forming a line, the function returns 1's (white) for those pixels, and 0's (black) if lines aren't found.  X-axis Sobel transformations work well for identifying vertical lines, while y-axis Sobel transformations are better at vertical lines.  I then took a maginitude threshold of the combined x- and y-axis transformations, where only a 1 was returned if both Sobels had 1's in that location, otherwise it returned a black zero. Here's an example of a y-transformation and magnitude threshold images:

![Sobel y-axis and magnitude thresholded images.][sobel_y_thres_mag]

Then we took a binary image o the gradient direction, and further combined that with the magnitude threshold image:

![Gradient direction and combined thresholds images.][grad_dir_comb_thresh]

Finally, I looked at saturation.  We converted the RGB image to HLS (Hue, Lightness, Saturation), and then just kept a saturation-thresholded binary image:

![Thresholded image.][hls_thresh]

This was then combined with another image to do something:

![Final combination of all thresholds.][final_thresh]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code block 15 in the function `map_lane()`. Here is the main chunk of that block that draws the lane lines:

```python
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warped, warped, warped))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = perspective_transform(color_warp, inv=True)
    # Combine the result with the original image
    weighted_img = cv2.addWeighted(undistorted, 1, new_warp, 0.3, 0)

```

This function starts by creating `warp_zero` a blank image to draw lines on, and `color_warp`, which takes the `warped` grayscale image, and stacks in RGB layers. Next it creates `pts`, a horizontal stack of `pts_left` and `pts_right`, which hold our lane line drawing points.  The lane lines are then draw onto `color_warp` with `cv2.fillPoly()`, and the unwarped into `new_warp`.  `new_warp` now exists in the "real world" space, and is overlayed onto our undistorted image using `cv2.addWeighted()`.

I've additionally added a thumbnail video of `out_img`'s as the car moves forward in the video, and text displaying radius of curvature and vehicle offset in the rest of the code.  Here is an example of my result on a test image:

![Image of final video.][vid_img]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline becomes less stable when going around sharp curves, or in areas where there are a lot of shadows, or inconsistent lighting that the camera has to adjust to.  I'm sure the pipeline would also have trouble in snow or rain as well, or if there was construction  with a lot line confusing lines, or complete lack of lines.  It would also probably have a hard time in a city like Boston, where the roads can be difficult and confusing for even humans. We mostly have ideal conditions with the video, and you don't have to live long to know the world is rarely provides the conditions we expect.

Improvements could definitely be made by having a higher-quality camera that adjusts rapidly to light changes, possibly introducing precipitation-like noise to the image, and more testing under various conditions.  
