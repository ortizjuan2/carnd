#/usr/bin/python!
import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from sys import exit
import numpy as np

camera_matrix = './cam_mat.p'

def undistortimage(image, cameraMatrix, newcameramtx, distCoeffs):
        #h,  w = image.shape[:2]
        #newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,(w,h),1,(w,h))
        dst = cv2.undistort(image, cameraMatrix, distCoeffs, None, newcameramtx)
        # crop the image
        #x,y,w,h = roi
        #dst = dst[y:y+h, x:x+w]
        return dst



def loadcameramtx(camera_matrix=camera_matrix):
    #load Camera Matrix and Distortion coeff
    try:
        with open(camera_matrix, 'rb') as f:
            data = pickle.load(f)
    except OSError as e:
        print('Error opening camara data: {}'.format(e))
        return [False, None, None]
    cameraMatrix = data['mtx']
    distCoeffs = data['dist']
    return [True, cameraMatrix, distCoeffs]



if __name__ == '__main__':
    # read test image
    image = cv2.imread('./test_images/test2.jpg')
    #
    # load camera matrix and distortion coefficients
    cameraMatrix, distCoeffs = loadcameramtx(camera_matrix)
    dst = undistortimage(image, cameraMatrix, distCoeffs)
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(dst)
    plt.show()
