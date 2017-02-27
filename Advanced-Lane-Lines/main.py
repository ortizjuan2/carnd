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



"""
lane = lanedetector.lane('./cam_mat.p')
filenames = sorted(glob('video/*.jpg'))
for i in trange(len(filenames)):
    image = cv2.imread(filenames[i])
    dst = process_image(image)
    ret = cv2.imwrite('./undist2/new{:04d}.jpg'.format(i), dst)

'slsls{:2.3f}'.format(12.0233033)


for i in range(45,lane.S_right.shape[0]):
    fright = np.poly1d(lane.S_right[i].reshape(3,))
    ploty = np.linspace(0, self.warped.shape[0]-1, self.warped.shape[0]/2 )
    ploty = ploty.reshape(ploty.shape[0], 1)
    right_fitx = fright(ploty)
    plt.plot(right_fitx, ploty)
plt.show()


"""
