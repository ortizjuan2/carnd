import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import trange
from glob import glob
import pickle
from pathlib import Path
from moviepy.editor import VideoFileClip
#
#from vhdetector import model, detector
import vhdetector


model = None
detector = None



def process_image(image):
    global model
    global detector
    dst = detector.draw_boxes(image, model)
    return dst

    

if __name__ == '__main__':
    model = vhdetector.model()
    detector = vhdetector.detector()
    test_output = 'result.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    # clip1 = VideoFileClip('harder_challenge_video.mp4')
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(test_output, audio=False)


#    files = sorted(glob('video/*.jpg'))
#    model = vhdetector.model()
#    detector = vhdetector.detector()
#
#    #draw boxes
#    for i in trange(len(files)):
#        image = cv2.imread(files[i])
#        dst = process_image(image)
#        ret = cv2.imwrite('video2/img{:04d}.jpg'.format(i), dst)
#






