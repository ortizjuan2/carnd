#/usr/bin/python!
import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from sys import exit
import numpy as np
from tqdm import trange


cam_imgs_files = './camera_cal'

def camera_cal(imgfiles):
    filenames = glob(imgfiles+'/*.jpg')
    if len(filenames) == 0:
        print('No images files found.')
        exit(-1)
    # define object points list
    objp = np.zeros((9*6, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    #define image points list
    objpoints = []
    imgpoints = []
    # start processing each chessboard image
    for i in trange(len(filenames)):
        gray = cv2.cvtColor(cv2.imread(filenames[i]), cv2.COLOR_BGR2GRAY)
        # find corners in image
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            # corners found in image
            objpoints.append(objp)
            imgpoints.append(corners)
    # use object points and image points to calibrate camera
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None)

    # camera calibration complete, save mtx and dist matrices
    cam_mat = {'mtx':cameraMatrix, 'dist':distCoeffs, 'rvecs':rvecs, 'tvecs':tvecs}
    with open('./cam_mat.p', 'wb') as f:
        pickle.dump(cam_mat, f)
    # end
    print('Camera calibration data saved to disk!: cam_mat.p')


if __name__ == '__main__':
    camera_cal(cam_imgs_files)
