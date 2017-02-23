import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from sys import exit
import numpy as np
from tqdm import trange
from undistort import loadcameramtx, undistortimage

distance_from_vp = 45
golden_ratio = 1.6180339887498949
border = 100


def warp_perspective(image, camera_center):
    ## image warp into bird's eye view ##  
    # Vanishing point
    Pv = [*camera_center]
    P0 = [border, image.shape[0]]
    # find left vanishing line
    line_left_pv = np.cross([*P0,1], [*Pv,1])
    #
    P4 = [image.shape[1]-border, image.shape[0]]
    # find right vanishing line
    line_right_pv = np.cross([*P4,1], [*Pv,1])
    #
    # define upper horizontal line used to form the shape
    # of the mask.
    y1 = Pv[1] + distance_from_vp
    x1 = (-line_left_pv[1]*y1 - line_left_pv[2])/line_left_pv[0]
    x2 = (-line_right_pv[1]*y1 - line_right_pv[2])/line_right_pv[0]
    # find left border of the warped image
    line_left_border = np.cross([*P0,1], [border,0,1])
    # find right border of the warped image
    line_right_border = np.cross([*P4,1], [image.shape[1]-border,0,1])
    # define upper points of the trapezoid mask
    P1 = [x1, y1]
    P3 = [x2, y1]
    # The mask if divided between left and right part
    # below points are the vertical line dividing the two sides
    P2 = [Pv[0], y1]
    P5 = [Pv[0], P0[1]]
    # draw region of interest
    vertices_left = np.array([[P0, P1, P2, P5]], dtype=np.int32)
    vertices_right = np.array([[P5, P2, P3, P4]], dtype=np.int32)
    # create the mask image
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, vertices_left, (255, 255, 255))
    mask = cv2.fillPoly(mask, vertices_right, (255, 255, 255))
    masked = cv2.bitwise_and(image, mask)
    # image warp
    # identify eye position in the scene
    eye = ((image.shape[0]-Pv[1]) * golden_ratio)+Pv[1]
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
    sizey = int(np.abs(image.shape[0]-Ptr[1]))
    dst_p = np.array([[0,0],                   # upper left corner
                      [Ptr[0]-Ptl[0], 0],      # upper right corner
                      [P4[0]-P0[0], sizey],  # bottom rith corner
                      [0, sizey]], dtype=np.float32) # bottom left corner
    # find M matrix used to warp perspective    
    M = cv2.getPerspectiveTransform(src_p, dst_p)
    # find the inverse matrix
    Minv = cv2.getPerspectiveTransform(dst_p, src_p)
    # warp the image
    warped = cv2.warpPerspective(masked, M, 
                                (image.shape[1],sizey), 
                                flags=(cv2.INTER_LINEAR)) # | cv2.WARP_FILL_OUTLIERS))
    return [warped, Minv] 


    


if __name__ == '__main__':
    filenames = glob('./test_images/*.jpg')
    cameraMatrix, distCoeffs = loadcameramtx()
    for i in trange(len(filenames)):
        img = cv2.imread(filenames[i])
        dst, newcameramtx, roi = undistortimage(img, cameraMatrix, distCoeffs)
        camera_center = [newcameramtx[0][2], newcameramtx[1][2]]
        warped, Minv = warp_perspective(dst, camera_center)
        # Unwarp image after filling lane to test
        alfa=0.4
        beta=1.
        lamb=0.
        unwarp = cv2.warpPerspective(warped, 
                                     Minv, 
                                     (img.shape[1], img.shape[0]), 
                                     flags=(cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS))
        mks = np.zeros_like(unwarp)
        mks[unwarp != 0] = 255
        mks_ret = cv2.addWeighted(mks, alfa, unwarp, beta, lamb)
        final_img = cv2.addWeighted(mks_ret, alfa, dst, beta, lamb)
        # at the end crop the image
        x,y,w,h = roi
        final_img = final_img[y:y+h, x:x+w]
        ret = cv2.imwrite('./undist2/roi{:02d}.jpg'.format(i), final_img)
        #ret = cv2.imwrite('./undist2/warped{:02}.jpg'.format(i), warped)


