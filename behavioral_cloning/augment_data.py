import h5py
from tqdm import trange
import cv2
import numpy as np




dataset = './train.h5' 


def augment(data):
    print('Opening dataset: {} ... '.format(dataset), end='')
    try:
        h5ds = h5py.File(data,'a')
    except OSError as e:
        print(e)
        return -1

    m = h5ds['images'].shape[0] 
    
    print('{:>5d} images found!'.format(m))
    
    ## Generating fake images
    
    print('Generating fake images:')

    for i in trange(m):
        img = h5ds['images'][i]
        label = h5ds['labels'][i] 
        ## Flipped
        flipped = cv2.flip(img, 1) # Flip around y axis
        h5ds['images'].resize((m+(i*3)+3, *img.shape))
        h5ds['labels'].resize((m+(i*3)+3, 1))
        h5ds['images'][-3:] =  flipped.reshape(1, *flipped.shape)
        h5ds['labels'][-3:] = -label 
        ## Motion Blur
        kernel_motion_blur = np.zeros((5,5))
        kernel_motion_blur[int((5-1)/2),:] = np.ones(5)
        kernel_motion_blur /= 5
        blured = cv2.filter2D(img, -1, kernel_motion_blur)
        blured_flipped = cv2.flip(blured, 1)
        h5ds['images'][-2:] = blured.reshape(1, *blured.shape)
        h5ds['labels'][-2:] = label 
        ## Blured and Flipped
        h5ds['images'][-1:] = blured_flipped.reshape(1, *blured_flipped.shape)
        h5ds['labels'][-1:] = -label 


 
 #           img = cv2.imread(logdata['center'][index[i]])
 #           if img is not None:
 #               img = cv2.cvtColor(img[20:140,:,:], cv2.COLOR_BGR2RGB)
 #               train_images.resize((train_images.shape[0]+1, *img.shape))
 #               train_labels.resize((train_labels.shape[0]+1, 1))
 #               train_images[-1,:] = img.reshape(1,*img.shape)
 #               train_labels[-1:] = logdata['steering'][index[i]].reshape(1,1)
    
    ## See current amount of shapes after flipping

    m = h5ds['images'].shape[0] 

    
    
    # TODO: do others augmentations
    
    h5ds.close()
    print('Final amount of images: {}'.format(m))
    return 0    
 



if __name__ == '__main__':
    augment(dataset)

