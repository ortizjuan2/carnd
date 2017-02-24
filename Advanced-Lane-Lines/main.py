import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from sys import exit
import numpy as np
from tqdm import trange
from moviepy.editor import VideoFileClip
#
from bnconv import convert2binary
from undistort import undistortimage, loadcameramtx
from warpimage import warp_perspective
from findlane import findlane

image_width = 1280
image_height = 720

params = None


def loadparams(paramsfile):
    params = {}
    params['cameramtx'] = None
    params['distCoeffs'] = None
    params['camera_center'] = None
    params['newcameramtx'] = None
    params['image_roi'] = None
    ret, cameramtx, distCoeffs = loadcameramtx(paramsfile) 
    if ret == True:
       newcameramtx, image_roi = cv2.getOptimalNewCameraMatrix(cameramtx,
                                    distCoeffs,
                                    (image_width, image_height),
                                    1,
                                    (image_width, image_height))
    else:
        print('Error opening camera parameters file.')
        print('Please run camera calibration process.')
        return ret
    
    params['cameramtx'] = cameramtx
    params['distCoeffs'] = distCoeffs
    #params['camera_center'] = [newcameramtx[0][2], newcameramtx[1][2]]
    params['camera_center'] = [cameramtx[0][2], cameramtx[1][2]]
    #params['newcameramtx'] = newcameramtx
    params['newcameramtx'] = cameramtx
    #params['image_roi'] = image_roi
    params['image_roi'] = [0, 0, 1280, 720]
    return [ret, params]



def process_image(image):
    global params
    # convert to binary to higlight lane
    binary = convert2binary(image)
    # undistort image
    undistorted = undistortimage(binary,
                                 params['cameramtx'],
                                 params['newcameramtx'], 
                                 params['distCoeffs'])
    warped, Minv = warp_perspective(undistorted, params['camera_center'])
    func_left, func_right = findlane(warped)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0]/2 )
    ploty = ploty.reshape(ploty.shape[0], 1)
    left_fitx = func_left(ploty)
    right_fitx = func_right(ploty)
    p_left = np.append(left_fitx, ploty, axis=1)
    p_righ = np.append(right_fitx, ploty, axis=1)  
    vertices = np.append(p_left, p_righ[::-1], axis=0)
    #vertices = vertices.reshape(1, vertices.shape[0], vertices.shape[1])
    out_warped = np.zeros_like(warped)
    out_warped = np.dstack((out_warped, out_warped, out_warped))
    ret = cv2.fillPoly(out_warped, [vertices.astype(np.int32)], (0,255,0))
    ret = cv2.polylines(out_warped,[p_left.astype(np.int32)], False, (0,0,255), 10)
    ret = cv2.polylines(out_warped, [p_righ.astype(np.int32)], False, (255,0,0), 10)  
    unwarped = cv2.warpPerspective(out_warped, Minv, 
                        (image.shape[1], image.shape[0]), 
                        flags=(cv2.INTER_LINEAR)) # | cv2.WARP_FILL_OUTLIERS))
    final_image = cv2.addWeighted(unwarped, 0.5, image, 1., 0.)
    #x,y,w,h = params['image_roi']
    #final_image = final_image[y:y+h, x:x+w]
    return final_image

def show(image, gray=True):
    if gray == True:
        plt.imshow(image, cmap='gray')
    else:
        b,g,r = cv2.split(image)
        image = cv2.merge((r,g,b))
        plt.imshow(image)
    plt.show()



if __name__ == '__main__':
    [ret, params] = loadparams('./cam_mat.p')
    if ret != True:
        exit(-1)
    test_output = 'test.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    # clip1 = VideoFileClip('harder_challenge_video.mp4')
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(test_output, audio=False)
"""
filenames = sorted(glob('./video/*.jpg'))
for i in trange(len(filenames)):
    image = cv2.imread(filenames[i])
    dst = process_image(image)
    ret = cv2.imwrite('./video_warped/main{:02d}.jpg'.format(i), dst)        

gamma = 0.2
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
img_gamma = cv2.LUT(image, table)


"""
