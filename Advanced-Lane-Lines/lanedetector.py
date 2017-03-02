import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from sys import exit
import numpy as np




class lane(object):
    
    def __init__(self, paramsfile):
        self.distance_from_vp = 65
        self.golden_ratio = 1.6180339887498949
        self.border = 100
       # camera matrix and distortion coefficients
        self.cameramtx = None
        self.distCoeffs = None
        self.camera_center = None
        self.shape = [720, 1280]
        # draw region of interest
        self.vertices_left = None 
        self.vertices_right = None 
        self.loadcameramtx(paramsfile)
        self.setPerspectivPoints()
        # binary image
        self.binary = None
        # undistorted binary image
        self.undistorted = None
        # warped binary image
        self.warped  = None
        # final image
        self.final_image = None
        # inverse matrix to unwarp image
        M = None
        Minv = None
        # store upto maxlast lanes
        self.maxlast = 50
        # store last lanes found to return its average
        self.S_left = np.array([0,0,0], dtype=np.float32).reshape(1,3)
        self.S_right = np.array([0,0,0], dtype=np.float32).reshape(1,3)
        # define threshold of required points to consider a lane valid
        self.thrvalidleft = 95162 # mean - std of lane indicators
        self.thrvalidright = 16275 # mean - std of lane indicators
        self.lastlanefound = False
        self.margin = 80
        self.thrnotfound = 10
        self.cntnotfound = 0

    def findlane(self, warped):
        gamma = 0.3 
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
            win_xleft_low = leftx_current - self.margin# + (window)
            win_xleft_high = leftx_current + self.margin# - (window)
            win_xright_low = rightx_current - self.margin# + (widow)
            win_xright_high = rightx_current + self.margin# - (window)
            # Draw the windows on the visualization image
            ret = cv2.rectangle(out_img,
                                (win_xleft_low,win_y_low),
                                (win_xleft_high,win_y_high),
                                (0,255,0), 2) 
            ret = cv2.rectangle(out_img,
                                (win_xright_low,win_y_low),
                                (win_xright_high,win_y_high),
                                (0,255,0), 2) 
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
        if self.S_left.all() == 0:
            self.S_left = left_fit
            self.S_right = right_fit
        else:
            self.S_left = (gamma)*left_fit + (1-gamma)*self.S_left
            left_fit = self.S_left
            self.S_right = (gamma)*right_fit + (1-gamma)*self.S_right
            right_fit = self.S_right
            #self.S_left = np.append(self.S_left, left_fit, axis=0)
            #self.S_right = np.append(self.S_right, right_fit, axis=0)
            # store only upto maxlast lanes detected
            #if self.S_left.shape[0] > self.maxlast:
            #    self.S_left = self.S_left[1:,:]
            #    self.S_right = self.S_right[1:,:]
            #left_fit = self.S_left.mean(axis=0)
            #right_fit = self.S_right.mean(axis=0)
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



    def findlanelast(self, warped):
        gamma = 0.3
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # take last fit to draw region of interest
        fleft = np.poly1d(self.S_left[-1].reshape(3,))
        fright = np.poly1d(self.S_right[-1].reshape(3,))
        pointsleft = fleft(nonzeroy)
        left_lane_inds = ((nonzerox > (pointsleft - self.margin)) 
                        & (nonzerox < (pointsleft + self.margin)))
        #
        pointsright = fright(nonzeroy)
        right_lane_inds =  ((nonzerox > (pointsright - self.margin)) 
                        & (nonzerox < (pointsright + self.margin)))
        # check if there are anought points
        if ((len(left_lane_inds) <  self.thrvalidleft) 
            | (len(right_lane_inds) < self.thrvalidright)):
            print('Not enought points')
            return [False, None, None]
         # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2).reshape(1,3)
        right_fit = np.polyfit(righty, rightx, 2).reshape(1,3)
        # moving average
        self.S_left = (gamma)*left_fit + (1-gamma)*self.S_left
        left_fit = self.S_left
        self.S_right = (gamma)*right_fit + (1-gamma)*self.S_right
        right_fit = self.S_right
        fleft = np.poly1d(left_fit.reshape(3,))
        fright = np.poly1d(right_fit.reshape(3,))
        ## Create an image to draw on and an image to show the selection window
        #out_img = np.dstack((warped, warped, warped))
        #window_img = np.zeros_like(out_img)
        ## Color in left and right line pixels
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        ## Generate a polygon to illustrate the search window area
        ## And recast the x and y points into usable format for cv2.fillPoly()
        #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        #left_line_pts = np.hstack((left_line_window1, left_line_window2))
        #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        #right_line_pts = np.hstack((right_line_window1, right_line_window2))

        ## Draw the lane onto the warped blank image
        #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #plt.imshow(result)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        return [True, fleft, fright]


    def drawlane(self, image):
        # convert to binary to higlight lane
        self.convert2binary(image)
        # undistort image
        self.undistortimage(self.binary)
        self.warp_perspective(self.undistorted)
        #func_left, func_right = findlane(warped)
        if self.lastlanefound == True:
            ret, func_left, func_right = self.findlanelast(self.warped)
            if ret != True:
                self.cntnotfound += 1
                if self.cntnotfound >= self.thrnotfound:
                    func_left, func_right = self.findlane(self.warped)
                    self.cntnotfound = 0
                else:
                    print('using last lane found {}'.format(self.cntnotfound))
                    func_left = np.poly1d(self.S_left[-1].reshape(3,))
                    func_right = np.poly1d(self.S_right[-1].reshape(3,))
            else:    
                self.cntnotfound = 0
        else:
            func_left, func_right = self.findlane(self.warped)
            self.lastlanefound = True
        ploty = np.linspace(0, self.warped.shape[0]-1, self.warped.shape[0]/2 )
        ploty = ploty.reshape(ploty.shape[0], 1)
        left_fitx = func_left(ploty)
        right_fitx = func_right(ploty)
        p_left = np.append(left_fitx, ploty, axis=1)
        p_righ = np.append(right_fitx, ploty, axis=1)  
        vertices = np.append(p_left, p_righ[::-1], axis=0)
        #vertices = vertices.reshape(1, vertices.shape[0], vertices.shape[1])
        out_warped = np.zeros_like(self.warped)
        out_warped = np.dstack((out_warped, out_warped, out_warped))
        ret = cv2.fillPoly(out_warped, [vertices.astype(np.int32)], (0,255,0))
        ret = cv2.polylines(out_warped,[p_left.astype(np.int32)], False, (0,0,255), 10)
        ret = cv2.polylines(out_warped, [p_righ.astype(np.int32)], False, (255,0,0), 10)  
        unwarped = cv2.warpPerspective(out_warped, self.Minv, 
                            (image.shape[1], image.shape[0]), 
                            flags=(cv2.INTER_LINEAR)) # | cv2.WARP_FILL_OUTLIERS))
        self.final_image = cv2.addWeighted(unwarped, 0.5, image, 1., 0.)
        #   
        # Generate some fake data to represent lane-line pixels
        # Define conversions in x and y from pixels space to meters
        #ym_per_pix = 30/2234 # meters per pixel in y dimension
        #xm_per_pix = 3.7/926 # meters per pixel in x dimension
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
        #lane_centerx = (func_right(image.shape[0]) - func_left(image.shape[0])) / 2.0
        #lane_centerx = func_left(image.shape[0]) + lane_centerx
        lane_centerx = func_left(self.warped.shape[0]) + (912.6/2.0) #Use only left lane to identify lane center
        offset = (lane_centerx - self.camera_center[0]) * xm_per_pix  
        
        self.final_image = cv2.putText(self.final_image,
                            'Vehicle Offset: {:06.2f} m'.format(offset), 
                            (50,70), 
                            font, 
                            .6, 
                             (255,0,0), 2)

 
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

    def convert2binary(self, image, thr=150 ):
        alfa = 1.0
        beta = 1.0
        lamb = 0.0
        self.shape = image.shape
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
        self.binary = binary


    def undistortimage(self, image):
        self.undistorted = cv2.undistort(image, 
                                        self.cameramtx, 
                                        self.distCoeffs, 
                                        None, self.cameramtx)
            
    def setPerspectivPoints(self):
        ## image warp into bird's eye view ##  
        # Vanishing point
        Pv = [*self.camera_center]
        P0 = [self.border, self.shape[0]]
        # find left vanishing line
        line_left_pv = np.cross([*P0,1], [*Pv,1])
        #
        P4 = [self.shape[1]-self.border, self.shape[0]]
        # find right vanishing line
        line_right_pv = np.cross([*P4,1], [*Pv,1])
        #
        # define upper horizontal line used to form the shape
        # of the mask.
        y1 = Pv[1] + self.distance_from_vp
        x1 = (-line_left_pv[1]*y1 - line_left_pv[2])/line_left_pv[0]
        x2 = (-line_right_pv[1]*y1 - line_right_pv[2])/line_right_pv[0]
        # find left border of the warped image
        line_left_border = np.cross([*P0,1], [self.border,0,1])
        # find right border of the warped image
        line_right_border = np.cross([*P4,1], [self.shape[1]-self.border,0,1])
        # define upper points of the trapezoid mask
        P1 = [x1, y1]
        P3 = [x2, y1]
        # The mask if divided between left and right part
        # below points are the vertical line dividing the two sides
        P2 = [Pv[0], y1]
        P5 = [Pv[0], P0[1]]
        # draw region of interest
        self.vertices_left = np.array([[P0, P1, P2, P5]], dtype=np.int32)
        self.vertices_right = np.array([[P5, P2, P3, P4]], dtype=np.int32)
        # identify eye position in the scene
        eye = ((self.shape[0]-Pv[1]) * self.golden_ratio)+Pv[1]
        Pe = [Pv[0], eye]
        # identify eye's left line
        line_left_eye = np.cross([*Pe,1], [*P1,1])
        # found crossing point between eye left line
        # and left border of the warped image
        Ptl = np.cross(line_left_border, line_left_eye)
        Ptl = Ptl / Ptl[2]
        Ptl = [Ptl[0], Ptl[1]]
        # identify eye's right line
        line_right_eye = np.cross([*Pe,1], [*P3,1])
        # found crossing point between eye right line
        # and right border fo the warped image
        Ptr = np.cross(line_right_border, line_right_eye)
        Ptr = Ptr/Ptr[2]
        Ptr = [Ptr[0], Ptr[1]]
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
                                    flags=(cv2.INTER_LINEAR)) # | cv2.WARP_FILL_OUTLIERS))



