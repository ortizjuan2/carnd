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



# Class used to implement the sliding window and to call the
# predict function of the trained SVM model

class detector(object):
    def __init__(self):
        #self.n = 0
        self.on_boxes = {}
        # used to extract camera center coordinates
        cam = Path('cam_mat.p')
        #self.heat_map = np.zeros(shape=[720, 1280], dtype=np.int32)
        #self.prev_boxes = []
        self.frame_count = 0
        # number of frames to wait before processing heat_map
        self.max_frames = 5  
        # load camera file to extract center of image
        if cam.is_file() == True:
            data = pickle.load(open('cam_mat.p', 'rb'))
            self.cameramtx = data['mtx']
            self.imagecenter = (int(self.cameramtx[0][2]), int(self.cameramtx[1][2]))
            self.distCoeffs = data['dist']
        else:
            print('Camera matrix not found. It is required to identify region of interest.')
            raise Exception('ERROR: Camera matrix not found!')
        # golden ratio is used to calculate sliding windows
        self.golden_ratio = 1.6180339887498949
        self.boxes = self.get_boxes()
 

    # Calculate windows used in the sliding window process
    def get_boxes(self):
        #offset_base = 10
        #offset = 0
        imcy = self.imagecenter[1]
        ys_pairs = [[0+imcy, 76+imcy],
                    [0+imcy, 123+imcy],
                    [11+imcy, 76+imcy],
                    [11+imcy, 94+imcy],
                    [11+imcy, 123+imcy],
                    [11+imcy, 199+imcy],
                    [29+imcy, 76+imcy],
                    [29+imcy, 123+imcy]]
        shape = [720, 1280]
        boxes = []
        for pair in ys_pairs:
            w = int((pair[1]-pair[0]) * self.golden_ratio)
            x_start = 0
            shift = int(w * 0.5)
            while (x_start + w) <= shape[1]:
                x = 20
                y = 20
                p1 = (x_start, pair[0])
                p2 = (x_start+w, pair[1])
                around = [[(0 if (p1[0]-x)<0 else (p1[0]-x),(p1[1]+y)),((p2[0]-x),(p2[1]+y)), 0],
                          [((p1[0]),(p1[1]+y)),((p2[0]),(p2[1]+y)), 0],
                          [((p1[0]+x),(p1[1]+y)),((p2[0]+x),(p2[1]+y)), 0],
                          [(0 if (p1[0]-x)<0 else (p1[0]-x),(p1[1])),((p2[0]-x),(p2[1])), 0],
                          [((p1[0]+x),(p1[1])),(1280 if (p2[0]+x)>1280 else (p2[0]+x),(p2[1])), 0],
                          [(0 if (p1[0]-x)<0 else (p1[0]-x),(p1[1]-y)),((p2[0]-x),(p2[1]-y)), 0],
                          [((p1[0]),(p1[1]-y)),((p2[0]),(p2[1]-y)), 0],
                          [((p1[0]+x),(p1[1]-y)),(1280 if (p2[0]+x)>1280 else (p2[0]+x),(p2[1]-y)), 0]]
                box =  [p1, p2, 0, 4, around] # [rectangle, frame count, usage count decreasing]
                boxes.append(box)
                x_start += shift
        return boxes

    # Function used to implement the sliding window technique
    def draw_boxes(self, dst, model, color_code):
        self.frame_count += 1
        if self.frame_count > self.max_frames: # process only after more that 5 frames 
            self.frame_count = 0
            # iterate over all the defined windows to find cars
            for box in self.boxes:
                rec = dst[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                pred = model.predict(rec, color_code)
                if pred == 1: # if car detected then put it on-box
                    if (box[0],box[1]) not in self.on_boxes:
                        self.on_boxes[(box[0],box[1])] = [0, box]
                    
        # if we have detections
        if len(self.on_boxes) > 0: 
            keys = list(self.on_boxes.keys())
            for k in keys:
                box = self.on_boxes[k][1] # take the box to process it
                rec = dst[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                pred = model.predict(rec, color_code)
                parent_found = False
                if pred == 1: # if detected increase use counter for this box/window
                    box[2] += 1
                    parent_found = True
                child_found = False
                # iterate over all the child windows to try to confirm match
                for box_around in box[4]:
                    rec = dst[box_around[0][1]:box_around[1][1], box_around[0][0]:box_around[1][0]]
                    pred = model.predict(rec, color_code)
                    if pred == 1:
                        box[2] += 1
                        box_around[2] = 1
                        child_found = True
                    else: 
                        box_around[2] = 0
                # if current window inside the on_box collection does not
                # gived a match and niether their child windows then remove it
                if not (child_found or parent_found):
                    box[2] = 0
                    del self.on_boxes[k]

        ## prcoess windows left on on_boxes with a score > 10
        if len(self.on_boxes):
            # create heat map
            heat_map = np.zeros_like(dst[:,:,0])
            for k in self.on_boxes:
                if self.on_boxes[k][1][2] > 10:
                    box = self.on_boxes[k][1]
                    # draw current window on heat_map
                    heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
                    # draw child windows that have score > 0 
                    for box_around in box[4]:
                        if box_around[2] == 1:
                            heat_map[box_around[0][1]:box_around[1][1], box_around[0][0]:box_around[1][0]] += 1
            # use the label function from scikit image
            labels = label(heat_map)
            # if labels were found, process them
            if labels[1] > 0:
                for carnum in range(1, labels[1]+1):   
                    # find the upper left and lower right corners of the 
                    # rectangle
                    nonzero = (labels[0] == carnum).nonzero()
                    nonzerox = np.array(nonzero[1])
                    nonzeroy = np.array(nonzero[0])
                    box = ((np.min(nonzerox), np.min(nonzeroy)),
                        (np.max(nonzerox), np.max(nonzeroy)))
                    # draw final window on image
                    dst = cv2.rectangle(dst, box[0], box[1], (0,0,255), 2)
        return dst




## Class used to implement SVM model

class model(object):
    def __init__(self):
        # file name with the svm trained model
        f_svc = Path('svc.p')
        self.scaler = None
        # load trained model
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

    # Wrapper of the svm predict function
    def predict(self, image, color_code):
        features = self.extract_feature_array(image, color_code)
        # Fit a per-column scaler                
        # Apply the scaler to X
        scaled_X = self.scaler.transform(features)
        return self.svc.predict(scaled_X)[0]
   
    # function used to extract the features from each image
    def extract_feature_array(self, image, color_code):
        orient = 9
        pix_per_cell = 8
        cell_per_block = 3
        features = []
        # convert image to YCrCb color space
        if color_code == 'BGR':
            img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        img = cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)
        y,cr,cb = cv2.split(img)
        # smooth y channel to remove noise
        y = cv2.equalizeHist(y)
        # calculate histogram over each image channel
        ch0_hist = np.histogram(y, 32, (0,256))
        ch1_hist = np.histogram(cr, 32, (0,256))
        ch2_hist = np.histogram(cb, 32, (0,256))
        for ch in ([y, cr, cb]):
            # calculate hog features over each image channel
            feature_array = hog(ch,
                                orientations=orient,
                                pixels_per_cell=(pix_per_cell,pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                visualise=False,
                                feature_vector=False)
            features.append(feature_array.ravel())
        # stack HOG features, image and histogram to generate final features of the input image
        features = np.hstack((features))
        features = np.hstack((features, img.ravel(), ch0_hist[0], ch1_hist[0], ch2_hist[0]))
        return features.reshape(1,-1)
 
    # function used to train the model using the given images of cars no-cars     
    def trainmodel(self):
        f_features = Path('features.p')
        f_svc = Path('svc.p')

        features = {}

        if f_features.is_file() == False:

            # full set of images
            filenames_cars = sorted(glob('vehicles/*/*.*'))
            filenames_nocars = sorted(glob('non-vehicles/*/*.*'))
            # Extract car features from training images
            image = cv2.imread(filenames_cars[0])
            features['cars'] = self.extract_feature_array(image, 'BGR')
            print('Extracting cars features:')
            for i in trange(1, len(filenames_cars)):
                image = cv2.imread(filenames_cars[i])
                feature_array = self.extract_feature_array(image, 'BGR')
                features['cars'] = np.append(features['cars'], feature_array, axis=0)
                # add flipped image
                image = cv2.flip(image, 1)
                feature_array = self.extract_feature_array(image, 'BGR')
                features['cars'] = np.append(features['cars'], feature_array, axis=0)
            # Extract non-car features from training images
            image = cv2.imread(filenames_nocars[0])
            features['nocars'] = self.extract_feature_array(image, 'BGR')
            print('Extracting non-cars features:')
            for i in trange(1, len(filenames_nocars)):
                image = cv2.imread(filenames_nocars[i])
                feature_array = self.extract_feature_array(image, 'BGR')
                features['nocars'] = np.append(features['nocars'], feature_array, axis=0)
                # add flipped image
                image = cv2.flip(image, 1)
                feature_array = self.extract_feature_array(image, 'BGR')
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
        # save scaler for later use
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

    # wrapper to train the linear svm model using train data 
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
 



