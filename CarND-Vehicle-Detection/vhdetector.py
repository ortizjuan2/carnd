import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import trange
from glob import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
from pathlib import Path
from scipy.ndimage.measurements import label


class detector(object):
    def __init__(self):
        cam = Path('cam_mat.p')
        self.heat_map = np.zeros(shape=[720, 1280], dtype=np.int32)
        self.prev_boxes = []
        self.frame_count = 0
        self.max_frames = 5 # number of frames to wait before processing heat_map
 
        if cam.is_file() == True:
            data = pickle.load(open('cam_mat.p', 'rb'))
            self.cameramtx = data['mtx']
            self.imagecenter = (int(self.cameramtx[0][2]), int(self.cameramtx[1][2]))
            self.distCoeffs = data['dist']
        else:
            print('Camera matrix not found. It is required to identify region of interest.')
            raise Exception('ERROR: Camera matrix not found!')
        self.golden_ratio = 1.6180339887498949
        #self.shape=[720, 1280]
        self.boxes = self.get_boxes()
 
#    def undistortimage(self, image):
#        return cv2.undistort(image,
#                             self.cameramtx,
#                             self.distCoeffs,
#                             None, self.cameramtx)
    def get_boxes(self):
        offset_base = 12
        offset = 12
        ys = []
        shape = [720, 1280]
        shift = 0.3
        while (offset+self.imagecenter[1]) < shape[0]:
            y = int(offset_base+(offset*self.golden_ratio))
            if (y+self.imagecenter[1]) < shape[0]: ys.append(y)
            offset = y
        boxes = []
        for i in range(2, len(ys)):
            w = ys[i]-ys[1]
            #n = int(image.shape[1] / w) # number of boxes
            x_start = 0
            while (x_start+w) <= shape[1]:
                p1 = (x_start, ys[1]+self.imagecenter[1])
                p2 = (x_start+w, ys[i]+self.imagecenter[1])
                boxes.append([p1, p2])
                x_start = int(x_start+(w*shift))
        return boxes
 
    def draw_boxes(self, dst, model):
        #on_boxes = []
        #heat_map = np.zeros_like(dst[:,:,0])
        #dst = self.undistortimage(image)
        self.frame_count += 1
        for box in self.boxes:
            rec = dst[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            pred = model.predict(rec)
            if pred == 1:
                #on_boxes.append(box)
                self.heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # process heat_map each x frames
        if self.frame_count <= self.max_frames:
            for box in self.prev_boxes:
                dst = cv2.rectangle(dst, box[0], box[1], (0,0,255), 2)
        else:
            self.frame_count = 0 # reset frame count
            self.prev_boxes = [] # reset previous boxes
            self.heat_map[self.heat_map <= 3] = 0
            labels = label(self.heat_map)
            self.heat_map = np.zeros(shape=[720, 1280], dtype=np.int32)
            if labels[1] > 0:
                for carnum in range(1, labels[1]+1):   
                    nonzero = (labels[0] == carnum).nonzero()
                    nonzerox = np.array(nonzero[1])
                    nonzeroy = np.array(nonzero[0])
                    box = ((np.min(nonzerox), np.min(nonzeroy)),
                            (np.max(nonzerox), np.max(nonzeroy)))
                    self.prev_boxes.append(box)
                    dst = cv2.rectangle(dst, box[0], box[1], (0,0,255), 2)
 
 
        #if len(on_boxes) >= 1:
        #    for box in on_boxes:
        #        dst = cv2.rectangle(dst, box[0], box[1], (255,0,0))
               
        return dst



#class detector(object):
#    def __init__(self):
#        cam = Path('cam_mat.p')
#        if cam.is_file() == True:
#            data = pickle.load(open('cam_mat.p', 'rb'))
#            self.cameramtx = data['mtx']
#            self.imagecenter = (int(self.cameramtx[0][2]), int(self.cameramtx[1][2]))
#            self.distCoeffs = data['dist']
#        else:
#            print('Camera matrix not found. It is required to identify region of interest.')
#            raise Exception('ERROR: Camera matrix not found!')
#        self.golden_ratio = 1.6180339887498949
#        #self.shape=[720, 1280]
#        self.boxes = self.get_boxes()
#
# 
##    def undistortimage(self, image):
##        return cv2.undistort(image,
##                             self.cameramtx,
##                             self.distCoeffs,
##                             None, self.cameramtx)
# 
#    def get_boxes(self):
#        offset_base = 12 
#        offset = 0
#        ys = []
#        shape = [720, 1280]
#        shift = 0.3
#        while (offset+self.imagecenter[1]) < shape[0]:
#            y = int(offset_base+(offset*self.golden_ratio))
#            if (y+self.imagecenter[1]) < shape[0]: ys.append(y)
#            offset = y
#        boxes = []
#        for i in range(2, len(ys)):
#            w = ys[i]-ys[1]
#            #n = int(image.shape[1] / w) # number of boxes
#            x_start = 0
#            while (x_start+w) <= shape[1]:
#                p1 = (x_start, ys[1]+self.imagecenter[1])
#                p2 = (x_start+w, ys[i]+self.imagecenter[1])
#                boxes.append([p1, p2])
#                x_start = int(x_start+(w*shift))
#        return boxes
#
#    def draw_boxes(self, dst, model):
#        on_boxes = []
#        heat_map = np.zeros_like(dst[:,:,0])
#        #dst = self.undistortimage(image)
#        for box in self.boxes:
#            rec = dst[box[0][1]:box[1][1], box[0][0]:box[1][0]]
#            pred = model.predict(rec)
#            if pred == 1:
#                on_boxes.append(box)
#                heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
#        heat_map[heat_map <= 1] = 0
#        labels = label(heat_map)
#        if labels[1] > 0:
#            for carnum in range(1, labels[1]+1):    
#                nonzero = (labels[0] == carnum).nonzero()
#                nonzerox = np.array(nonzero[1])
#                nonzeroy = np.array(nonzero[0])
#                box = ((np.min(nonzerox), np.min(nonzeroy)),
#                        (np.max(nonzerox), np.max(nonzeroy)))
#                dst = cv2.rectangle(dst, box[0], box[1], (0,0,255))
#
#        #if len(on_boxes) >= 1:
#        #    for box in on_boxes:
#        #        dst = cv2.rectangle(dst, box[0], box[1], (255,0,0))
#                
#        return dst

           


class model(object):
    def __init__(self):
        f_svc = Path('svc.p')
        self.scaler = None
        if f_svc.is_file() == False:
            self.svc = None
            print('run python vhdetector.py, to train model.')
            self.trained = False
            #raise Exception('ERROR: Model not trained!')
        else:
            print('Model already trained, loading model from disk.')
            self.svc = pickle.load(open('svc.p', 'rb'))
            self.scaler = pickle.load(open('scaler.p', 'rb'))
            # check the accuracy of your classifier on the test dataset
            #print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
            self.trained = True

    def predict(self, image):
        features = self.extract_feature_array(image)
        # Fit a per-column scaler                
        # Apply the scaler to X
        scaled_X = self.scaler.transform(features)
        return self.svc.predict(scaled_X)[0]
   
#    def extract_feature_array(self, image):
#        orient = 9
#        pix_per_cell = 8
#        cell_per_block = 2
#        #block_norm =  'L2'
#        #gray = cv2.cvtColor(image[int(image.shape[0]/2):,:], cv2.COLOR_BGR2GRAY)
#        image = cv2.resize(image, (32,32), interpolation=cv2.INTER_CUBIC)
#        #b,g,r = cv2.split(image)
#        #b = cv2.equalizeHist(b)
#        #g = cv2.equalizeHist(g)
#        #r = cv2.equalizeHist(r)
#        #image = cv2.merge((r,g,b))
#        #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#        y,cr,cb = cv2.split(image)
#        y = cv2.equalizeHist(y)
#        features = []
#        for ch in ([y, cr, cb]):
#            feature_array = hog(ch,
#                            orientations=orient,
#                            pixels_per_cell=(pix_per_cell,pix_per_cell),
#                            cells_per_block=(cell_per_block, cell_per_block),
#                            #block_norm=snorm,
#                            visualise=False,
#                            feature_vector=False)
#            features.append(feature_array.ravel())
#        features = np.max((features), axis=0)
#        features = np.append(features, gray.ravel())
#        return features.reshape(1,-1)
 
    def extract_feature_array(self, image):
        orient = 9
        pix_per_cell = 8
        cell_per_block = 3
        #block_norm =  'L2-Hys'
        features = []
        img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        img = cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y,cr,cb = cv2.split(img)
        y = cv2.equalizeHist(y)
        ch0_hist = np.histogram(y, 32, (0,256))
        ch1_hist = np.histogram(cr, 32, (0,256))
        ch2_hist = np.histogram(cb, 32, (0,256))
        for ch in ([y, cr, cb]):
            feature_array = hog(ch,
                                orientations=orient,
                                pixels_per_cell=(pix_per_cell,pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                #block_norm=block_norm,
                                visualise=False,
                                feature_vector=False)
            features.append(feature_array.ravel())
        #features = np.max((features), axis=0)
        features = np.hstack((features))
        features = np.hstack((features, img.ravel(), ch0_hist[0], ch1_hist[0], ch2_hist[0]))
        return features.reshape(1,-1)
 
       
    def trainmodel(self):
        f_features = Path('features.p')
        f_svc = Path('svc.p')

        features = {}

        if f_features.is_file() == False:

            # full set of images
            filenames_cars = sorted(glob('vehicles/*/*.png'))
            filenames_nocars = sorted(glob('non-vehicles/*/*.png'))

            # Extract car features from training images
            image = cv2.imread(filenames_cars[0])
            #image = cv2.resize(image, (32,32), interpolation=cv2.INTER_CUBIC)
            features['cars'] = self.extract_feature_array(image)
            print('Extracting cars features:')
            for i in trange(1, len(filenames_cars)):
                image = cv2.imread(filenames_cars[i])
                #image = cv2.resize(image, (32,32), interpolation=cv2.INTER_CUBIC)
                feature_array = self.extract_feature_array(image)
                features['cars'] = np.append(features['cars'], feature_array, axis=0)
                # add flipped image
                image = cv2.flip(image, 1)
                feature_array = self.extract_feature_array(image)
                features['cars'] = np.append(features['cars'], feature_array, axis=0)


            # Extract non-car features from training images
            image = cv2.imread(filenames_nocars[0])
            #image = cv2.resize(image, (32,32), interpolation=cv2.INTER_CUBIC)
            features['nocars'] = self.extract_feature_array(image)
            print('Extracting non-cars features:')
            for i in trange(1, len(filenames_nocars)):
                image = cv2.imread(filenames_nocars[i])
                #image = cv2.resize(image, (32,32), interpolation=cv2.INTER_CUBIC)
                feature_array = self.extract_feature_array(image)
                features['nocars'] = np.append(features['nocars'], feature_array, axis=0)
                # add flipped image
                image = cv2.flip(image, 1)
                feature_array = self.extract_feature_array(image)
                features['nocars'] = np.append(features['nocars'], feature_array, axis=0)

            pickle.dump(features, open('features.p', 'wb'))

        else:
            print('Loading features from disk...')
            features = pickle.load(open('features.p', 'rb'))

        # Define a labels vector based on features lists
        y = np.hstack((np.ones(len(features['cars'])),
                       np.zeros(len(features['nocars']))))

        # Create an array stack of feature vectors
        X = np.vstack((features['cars'], features['nocars'])).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        self.scaler = X_scaler
        pickle.dump(X_scaler, open('scaler.p', 'wb'))
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        # check if svm was already trained
        
        if f_svc.is_file() == False:

            print('Training linear svm...')
            svc = self.trainsvm(X_train, y_train)
            # save trained model
            pickle.dump(svc, open('svc.p', 'wb'))
        else:
            print('Model already trained, loading model from disk...')
            svc = pickle.load(open('svc.p', 'rb'))
            # check the accuracy of your classifier on the test dataset
       
        print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

        # make predictions
        #pred = svc.predict(X_test[0:10].reshape(10, -1))
        #print('My SVC predicts: {}'.format(pred))
        #y_ = y_test[0:10]
        #print('For labels: {}'.format(y_))
        ## result = np.sqrt(np.sum((pred-y_)**2))
        #result = np.mean(pred == y_)
        #print('Accuracy over 10 samples: {:03.3f}'.format(result*100))

   
    def trainsvm(self, X_train, y_train):
        # Use a linear SVC (support vector classifier)
        svc = LinearSVC(max_iter=2000)
        # Train the SVC
        svc.fit(X_train, y_train)
        return svc



if __name__ == '__main__':
    # small set of images
    #filenames_cars= sorted(glob('vehicles_smallset/*/*.jpeg'))
    #filenames_nocars = sorted(glob('non-vehicles_smallset/*/*.jpeg'))

    model = model()

    #check if features where already extracted
    if model.trained == False:
        model.trainmodel()
 



