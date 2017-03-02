**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/calibproc.png "Undistorted Result"
[image2]: ./examples/undistorted.png "Road Transformed"
[image3]: ./examples/binary.png "Binary Example"
[image4]: ./examples/pointextract.png "Point extraction Example"
[image44]: ./examples/warped2.png "Warped Example"
[image5]: ./examples/warped.png "warped"
[image6]: ./examples/hist.png "Histogram"
[image7]: ./examples/fit.png "fit lane"
[image8]: ./examples/fit_poly.png "fit lane using latest fit"
[image9]: ./examples/finalimage.png "Final Image"
[video1]: ./.mp4 "Video"

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the python file camcalib.py included with this writeup. Also, the 20 chessboard images used for the calibration process can be found in the folder ./camera_cal

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Take into account that each chessboard image need to be converted to gray scale in order to be passed to the `findChessboardCorners` opencv function, also it is important to hightlight that this function requires the number of corners, in this case each chessboard image has 9 colums, 6 rows. 

After running this first part of the process, from 20 images processed only 17 returned corners.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The resulting values for the camera matrix and distortion coefficients were stored using pickle in the file `cam_mat.p` for later use in the lane detector process.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Once I have generated the `cam_mat.p` file with the camera matrix and distortion coefficients, as explained in the previous section, I can use those parameters to start undistorting images, thus, the first step is to load the parameters using the `loadcameramtx` function called automatically in the class initialization routine:

```python
def loadcameramtx(self, paramsfile):
    #load Camera Matrix and Distortion coeff
    try:
        with open(paramsfile, 'rb') as f:
            data = pickle.load(f)
            print('camera parameters loaded successfully')
    except OSError as e:
        print('Error opening camara data: {}'.format(e))
        raise ValueError
        #return False
    self.cameramtx = data['mtx']
    self.distCoeffs = data['dist']
    self.camera_center = [data['mtx'][0][2], data['mtx'][1][2]]
```

Then, any image can be undistorted using the `undistortimage` function passing it the image to undistort, the function will store the resulting image in the variable `self.undistorted`:

```python
def undistortimage(self, image):
    self.undistorted = cv2.undistort(image, 
                                    self.cameramtx, 
                                    self.distCoeffs, 
                                    None, self.cameramtx)
```
This is a sample image before and after applying the distortion correction process:

![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
The process to convert the input image to binary is inside the `convert2binary` function as described below. In this function I used HSV and HLS color spaces, from those I have extracted the S channels from both and added them to generate an initial image, then I generate the gradient magnitud using the resulting gradient x and y images from the Sobel operator and at the end used the combined channels and gradient magnitud to threshold the binary image. 

```python
def convert2binary(self, image, thr=150 ):
    alfa = 1.0
    beta = 1.0
    lamb = 0.0
    self.shape = image.shape
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_H,hsv_S,hsv_V = cv2.split(HSV)
    HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hls_H,hls_L,hls_S = cv2.split(HLS)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gradmag = np.abs(sobelx)+np.abs(sobely) 
    scale = gradmag.max()/255
    gradmag = gradmag/scale
    final_img = cv2.addWeighted(hls_S, alfa, hsv_S, beta, lamb)
    binary = np.zeros_like(final_img)
    binary[(final_img > 200) | (gradmag > 120)] = 255
    self.binary = binary
```
This is a sample binary image obtained using this process:

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The first step in the process to do the perspective transform is to identify the set of origin and destination points that will form the warped image, for this matter I have defined the function `setPerspectivPoints` inside the `lanedetector` class. This function uses a technique similar to how a person makes a drawing in paper, it start identifiying the vanishing point (in our case, this point is the camera center points which were provided it in the camera matrix), then you have to define two horizontal lines, the firts one is the base of the drawing (in our case the base of the image, shape[0]) and the second one is the line where the eye looking at the image is located. Then, to identify the upper destination point, you just trace a line from the eye point throught two points (defined as source points) and project that line until it reaches the vertical line of the source points at the base of the image, see the following image that better explain this process:


![alt text][image4]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 556, 451      | 0, 0        | 
| 766, 451      | 1080, 0      |
| 1180, 720     | 1080, 2234      |
| 100, 720      | 0, 2234        |


After identifying the source and destination points for the region of interest I applied the `getPerspectiveTransform` opencv function to obtain the Perspective Transformation Matrix `M` and its inverse used to unwarp the image at the end of the lane detection process:

```python
		# difine the four source points of the input image
        src_p = np.array([P1,   # upper left corner
                          P3,   # upper right corner
                          P4,   # bottom right corner
                          P0], dtype=np.float32) # bottom left corner
        # difine the four destination points of the output image
        self.sizey = int(np.abs(self.shape[0]-Ptr[1]))
        dst_p = np.array([[0,0],                   # upper left corner
                          [Ptr[0]-Ptl[0], 0],      # upper right corner
                          [P4[0]-P0[0], self.sizey],  # bottom rith corner
                          [0, self.sizey]], dtype=np.float32) # bottom left corner
        # find M matrix used to warp perspective    
        self.M = cv2.getPerspectiveTransform(src_p, dst_p)
        # find the inverse matrix
        self.Minv = cv2.getPerspectiveTransform(dst_p, src_p)
```

After using the point extration routine, I now have the parameters needed to apply the `warpPerspective` opencv function to each image I need to transform. I have defined a function `warp_perspective` inside the lanedetector class that takes the image, make a mask to extract only the region of interest, and then apply the `warpPerspective` opencv function, the returned warped image is then stored in the self.warped variable of the class as follows:

```python
def warp_perspective(self, image):
    # create the mask image
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, self.vertices_left, (255, 255, 255))
    mask = cv2.fillPoly(mask, self.vertices_right, (255, 255, 255))
    masked = cv2.bitwise_and(image, mask)
    # image warp
    # warp the image
    self.warped = cv2.warpPerspective(masked, self.M, 
                                     (image.shape[1],self.sizey), 
                                     flags=(cv2.INTER_LINEAR))

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image44]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The next step in the process is to identify lane pixels in the binary warped image. To do that, I used the histogram and slide window technique, I first take the incoming binary warped image and generate a histogram given the following result as an example:

![alt text][image5]
![alt text][image6]

This technique assumes the lanes are situated in the area at the picks of the histogram, thus, from that points a window is slided from botton to top of the image, centering the window at the mean of points when the windows has more than `minpix` which in my case was defined as 50 pixels. This will identify nonzero x and y indices which are later used to fit the second order polynomial over the detected lane pixels. See following image as an example of the process:

![alt text][image7]

The previous process is run only the first time to begin lane detection but after the first lane is detected, the next time the `findlanelast` routine is invoqued which takes the last polynomial identified and create a reagion of interest of about +-100 pixels along and around it. The slide window is not needed because the position of the lane should not change to much. The new pixels are identified an a new second order polynomial is fitted for the left and right lanes.

In the following image you can see how different the region around the last polynoial differst from the region detectected using the slide window.
![alt text][image8]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The following code snip from the `drawlane` function inside the `lanedetector` class depics the process of finding the radious of curvature and the offset of the car with respect to the center of the lane. 


```python   
#Generate some fake data to represent lane-line pixels
# Define conversions in x and y from pixels space to meters
ym_per_pix = 3.048/390.862 # meters per pixel in y dimension
xm_per_pix = 3.7/912.6 # meters per pixel in x dimension
y_eval = np.max(ploty)
# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty.reshape(ploty.shape[0],)*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty.reshape(ploty.shape[0],)*ym_per_pix, right_fitx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
#print(left_curverad, 'm', right_curverad, 'm')
#txt = 'left Curvature: {:06.2f}m right Curvature: {:06.2f}m'.format(left_curverad, right_curverad)
font = cv2.FONT_HERSHEY_SIMPLEX
# Example values: 632.1 m    626.2 m
self.final_image = cv2.putText(self.final_image,
                           'left Curvature: {:06.2f} m right Curvature: {:06.2f} m'.format(left_curverad[0], right_curverad[0]), 
                            (50,50), 
                            font, 
                            .6, 
                             (255,0,0), 2)
        
# find lane center
lane_centerx = func_left(self.warped.shape[0]) + (912.6/2.0) #Use only left lane to identify center
offset = (lane_centerx - self.camera_center[0]) * xm_per_pix  
self.final_image = cv2.putText(self.final_image,
                            'Vehicle Offset: {:06.2f} m'.format(offset), 
                            (50,70), 
                            font, 
                            .6, 
                             (255,0,0), 2)

```
Then this information is printed in the upper left corner of the final image as shown in the below example.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The `drawlane` function inside the `lanedetector` class is in charge of drawing the detected lane. After drawing the lane over the warped image, the warped image is then unwarped with the following code:

```python
out_warped = np.zeros_like(self.warped)
        out_warped = np.dstack((out_warped, out_warped, out_warped))
        ret = cv2.fillPoly(out_warped, [vertices.astype(np.int32)], (0,255,0))
        ret = cv2.polylines(out_warped,[p_left.astype(np.int32)], False, (0,0,255), 10)
        ret = cv2.polylines(out_warped, [p_righ.astype(np.int32)], False, (255,0,0), 10)  
        unwarped = cv2.warpPerspective(out_warped, self.Minv, 
                            (image.shape[1], image.shape[0]), 
                            flags=(cv2.INTER_LINEAR)) # | cv2.WARP_FILL_OUTLIERS))
```

Then, the unwarped image and the original image is then added to obtain the final result:

```python
self.final_image = cv2.addWeighted(unwarped, 0.5, image, 1., 0.)
```

The last part of the `drawlane` function identifies the radious of curvature, the offset with respecto to the center of the lane and put that information in the final image.

These are examples of the final result after completing all the stesp described previously:

![alt text][image9]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first version of this implementation showed a lot of wobbly lines, this was solved in part generating better binary images and implemented a moving average over the polynomial coefficients. After these two improvements the lane lines appear more smooth over the entire video.

Also, regarding the lane lines over the images with excesive light, the best results were obtained using the HSV and HSL color spaces combination, no other binary technique help me to display them correctly.

Hope this writeup helps to explains the whole process.

Cheers!
