import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob
from sys import exit
import numpy as np
from tqdm import trange
from moviepy.editor import VideoFileClip
#
#from bnconv import convert2binary
#from undistort import undistortimage, loadcameramtx
#from warpimage import warp_perspective
import lanedetector


lane = None

def process_image(image):
    global lane
    lane.drawlane(image)
    return lane.final_image


if __name__ == '__main__':
    lane = lanedetector.lane('./cam_mat.p')
    test_output = 'result.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    # clip1 = VideoFileClip('harder_challenge_video.mp4')
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(test_output, audio=False)


