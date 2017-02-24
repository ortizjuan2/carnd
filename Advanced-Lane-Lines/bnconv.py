#/usr/bin/python!
import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from sys import exit
import numpy as np
from tqdm import trange

def convert2binary(image, thr=150 ):
    alfa = 1.0
    beta = 1.0
    lamb = 0.0
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_H,hsv_S,hsv_V = cv2.split(HSV)
    HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hls_H,hls_L,hls_S = cv2.split(HLS)
    #
    #r = image[:,:,2]
    #r_bin = np.zeros_like(r)
    #r_bin[r > 100] = 255
    #
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                               
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gradmag = np.abs(sobelx)+np.abs(sobely) 
    scale = gradmag.max()/255
    gradmag = gradmag/scale

    final_img = cv2.addWeighted(hls_S, alfa, hsv_S, beta, lamb)
    binary = np.zeros_like(final_img)
    binary[(final_img > 200) | (gradmag > 120)] = 255
    return binary



if __name__ == '__main__':
    # read test image
    filenames = glob('./test_images/*.jpg')
    for i in trange(len(filenames)):
        image = cv2.imread(filenames[i])
        binary = convert2binary(image)
        ret = cv2.imwrite('./undist2/bianry{:02d}.jpg'.format(i), binary)


    ##
    #plt.subplot(231)
    #plt.imshow(hsv_H, cmap='gray')
    #plt.subplot(232)
    #plt.imshow(hsv_S, cmap='gray')
    #plt.subplot(233)
    #plt.imshow(hsv_V, cmap='gray')
    #plt.subplot(234)
    #plt.imshow(hls_H, cmap='gray')
    #plt.subplot(235)
    #plt.imshow(hls_L, cmap='gray')
    #plt.subplot(236)
    #plt.imshow(hls_S, cmap='gray')
    #plt.show()
    ####

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    #sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    #gradmag = np.abs(sobelx)+np.abs(sobely) 
    #scale = gradmag.max()/255
    #gradmag = gradmag/scale
    #binary = np.zeros_like(final_img)
    #binary[(final_img > thr) | (gradmag > thr)] = 255
"""

# Convert to HLS color space and separate the S channel
# Note: img is the undistorted image
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
s_channel = hls[:,:,2]

# Grayscale image
# NOTE: we already saw that standard grayscaling lost color information for the lane lines
# Explore gradients in other colors spaces / color channels to see what might work better
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel x
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# Threshold color channel
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Stacked thresholds')
ax1.imshow(color_binary)

ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined_binary, cmap='gray')




"""




