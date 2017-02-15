#/usr/bin/python!
import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from sys import exit
import numpy as np

camera_matrix = './cam_mat.p'

def undistortimage(image, cameraMatrix, distCoeffs):
        dst = cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)
        return dst


if __name__ == '__main__':
    #load Camera Matrix and Distortion coeff
    try:
        with open(camera_matrix, 'rb') as f:
            data = pickle.load(f)
    except OSError as e:
        print('Error opening camara data: {}'.format(e))
        exit(-1)
    cameraMatrix = data['mtx']
    distCoeffs = data['dist']
    # read test image
    image = cv2.imread('./test_images/test2.jpg')
    #
    dst = undistortimage(image, cameraMatrix, distCoeffs)
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(dst)
    plt.show()
