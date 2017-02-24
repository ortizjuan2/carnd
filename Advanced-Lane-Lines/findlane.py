import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from sys import exit
import numpy as np
from tqdm import trange

gamma = 0.8
S_left = np.array([0,0,0], dtype=np.float32).reshape(1,3)
S_right = np.array([0,0,0], dtype=np.float32).reshape(1,3)


def findlane(warped):
    global S_left
    global S_right
    global gamma
    
    hist = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped, warped, warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(hist.shape[0]/2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin# + (window)
        win_xleft_high = leftx_current + margin# - (window)
        win_xright_low = rightx_current - margin# + (widow)
        win_xright_high = rightx_current + margin# - (window)
        # Draw the windows on the visualization image
        ret = cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        ret = cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) 
                        & (nonzeroy < win_y_high) 
                        & (nonzerox >= win_xleft_low) 
                        & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) 
                         & (nonzeroy < win_y_high) 
                         & (nonzerox >= win_xright_low) 
                         & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        #print('left: {}'.format(len(good_left_inds)))
        #print('right: {}'.format(len(good_right_inds)))
 
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2).reshape(1,3)
    right_fit = np.polyfit(righty, rightx, 2).reshape(1,3)
    if S_left.all() == 0:
        S_left = left_fit
        S_right = right_fit
    else:
        #S_left = (gamma)*left_fit + (1-gamma)*S_left
        #left_fit = S_left
        #S_right = (gamma)*right_fit + (1-gamma)*S_right
        #right_fit = S_right
        S_left = np.append(S_left, left_fit, axis=0)
        S_right = np.append(S_right, right_fit, axis=0)
        if S_left.shape[0] > 20:
            S_left = S_left[1:,:]
            S_right = S_right[1:,:]
        left_fit = S_left.mean(axis=0)
        right_fit = S_right.mean(axis=0)


## 
    fleft = np.poly1d(left_fit.reshape(3,))
    fright = np.poly1d(right_fit.reshape(3,))
    # Generate x and y values for plotting
    #ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    #left_fitx = fleft(ploty)
    #right_fitx = fright(ploty)
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.show()

    return [fleft, fright]





