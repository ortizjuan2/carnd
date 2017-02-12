# python
import pandas
import h5py
import glob
import numpy as np
import cv2
from random import shuffle
from tqdm import trange

IMG_DATA = './IMG/'
CSV_FILE_NAME = './driving_log.csv'
MIN_SPEED = 0.0

def getimage(filename):
    img = cv2.imread(filename)
    rows, cols, chan = img.shape
    dst = img[int(rows/2):rows]
    dst = cv2.resize(dst, (OUTPUT_WIDTH, OUTPUT_HIGH), 0, 0, cv2.INTER_LINEAR)
    return dst

def show_video(filenames, t=320):
    cv2.namedWindow('Center Camera', cv2.WINDOW_AUTOSIZE)
    #cv2.startWindowThread()
    #rows, cols, chan = frame_size
    #sz = rows*cols*chan
    #speed = 0 # DEBUG
    for i in range(len(filenames)):
        #img = getimage(filenames[i])
        img = cv2.imread(filenames[i])
        cv2.imshow('Center Camera', img)
        cv2.waitKey(t)
    cv2.destroyAllWindows()

## Index(['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'], dtype='object')
def get_data(imgsdir, csvfile):
    try:
        h5_train = h5py.File('train.h5','w')
        h5_test = h5py.File('test.h5','w')
        logdata = pandas.read_csv(csvfile)
    except OSError as e:
        print(e)
        return -1
    index = logdata['center'].index.tolist()
    shuffle(index)
    shuffle(index)
    shuffle(index) 
    m_train = int(len(index)*0.9) # 90% of data for training
    #m_test = len(index) - m_train # % of data for testing


    train_images = h5_train.create_dataset('images', 
                                    (1,120,320,3), 
                                    dtype=np.uint8, 
                                    maxshape=(None,120,320,3))
    train_labels = h5_train.create_dataset('labels', (1,1), 
                                    dtype=np.float32,
                                    maxshape=(None, 1))
    test_images = h5_test.create_dataset('images', 
                                    (1,120,320,3), 
                                    dtype=np.uint8, 
                                    maxshape=(None,120,320,3))
    test_labels = h5_test.create_dataset('labels', (1,1), 
                                    dtype=np.float32,
                                    maxshape=(None, 1))
    # Read first training image and label
    img = None
    ntrain = 0
    while img is None:
        img = cv2.imread(logdata['center'][index[ntrain]])
        ntrain += 1
    img = cv2.cvtColor(img[20:140,:,:], cv2.COLOR_BGR2RGB)
    train_images[:] = img.reshape(1,*img.shape)
    train_labels[:] = logdata['steering'][index[ntrain]].reshape(1,1)
    
    # Read first test image and label
    img = None
    ntest = 0
    while img is None:
        img = cv2.imread(logdata['center'][index[m_train+ntest]])
        ntest += 1
    img = cv2.cvtColor(img[20:140,:,:], cv2.COLOR_BGR2RGB)
    test_images[:] = img.reshape(1,*img.shape)
    test_labels[:] = logdata['steering'][index[m_train+ntest]].reshape(1,1)
    
    print('Generating training dataset:')
    for i in trange(ntrain, m_train):
    #for i in trange(300):
        if logdata['speed'][index[i]] >= MIN_SPEED:
            #img = cv2.cvtColor(cv2.imread(data['center'][index[i]]), cv2.COLOR_BGR2GRAY)
            img = cv2.imread(logdata['center'][index[i]])
            if img is not None:
                img = cv2.cvtColor(img[20:140,:,:], cv2.COLOR_BGR2RGB)
                train_images.resize((train_images.shape[0]+1, *img.shape))
                train_labels.resize((train_labels.shape[0]+1, 1))
                train_images[-1,:] = img.reshape(1,*img.shape)
                train_labels[-1:] = logdata['steering'][index[i]].reshape(1,1)
    
    print('Generating test dataset:')
    for i in trange(m_train+ntest, len(index)):
    #for i in trange(300):
        if logdata['speed'][index[i]] >= MIN_SPEED:
            #img = cv2.cvtColor(cv2.imread(data['center'][index[i]]), cv2.COLOR_BGR2GRAY)
            img = cv2.imread(logdata['center'][index[i]])
            if img is not None:
                img = cv2.cvtColor(img[20:140,:,:], cv2.COLOR_BGR2RGB)
                test_images.resize((test_images.shape[0]+1, *img.shape))
                test_labels.resize((test_labels.shape[0]+1, 1))
                test_images[-1,:] = img.reshape(1,*img.shape)
                test_labels[-1:] = logdata['steering'][index[i]].reshape(1,1)
 
    h5_train.close()
    h5_test.close()
    return 0    
        
#    img_names = glob.glob(dir + 'center*.jpg')
#    shuffle(img_names)
#    shuffle(img_names)
#    telemetri_data = pandas.read_csv(LOG_NAME)
#    features = np.array(())
#    steerdata = np.array(())
#    for i in trange(len(img_names)):
#    #for i in trange(100):
#        img = cv2.cvtColor(cv2.imread(img_names[i]), cv2.COLOR_BGR2GRAY)
#        dst = img[int(img.shape[0]/2):img.shape[0], 0:img.shape[1]]
#        features = np.append(features, dst)
#        idx = telemetri_data[telemetri_data['center'] == img_names[i][-38:]].index.tolist()
#        steering = telemetri_data['steering'][idx].get_values()
#        steerdata = np.append(steerdata, steering)
#    features = features.reshape(len(img_names), *dst.shape)
#    #features = features.reshape(100, *img.shape)
#    return [features, steerdata]



if __name__ == '__main__':
    get_data(IMG_DATA, CSV_FILE_NAME)
    


    
