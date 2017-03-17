**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/classes.png
[image11]:./examples/car_ycrcb.png
[image2]: ./examples/hog_result.png
[image3]: ./examples/imgsub.png
[image4]: ./examples/examples.png
[image51]: ./examples/heat1.png
[image52]: ./examples/heat2.png
[image53]: ./examples/heat3.png
[image6]: ./examples/label.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file `vhdetector.py`. It is implemented in the class `model` inside the funcion `extract_feature_array`. 
This function will receive and input image and the the Color espace `RGB` or `BGR`, then the input image is converted to YCrCb color space and then resized to 32x32 pixels. Then histogram is calculated over each Y, Cr and Cb channels and also the HOG features are calculated over the same channels.
At the end the final features are the result of the HOG features,  the resized image and the histograms of each color channel.

![alt text][image1]

I have explored different color spaces, but the one which gives better results predicting cars was the  `YCrCb` color space. 
Regarding  `HOG` parameters, I ended up using the following:
```python
185         orient = 9
186         pix_per_cell = 8
187         cell_per_block = 3
```

Here is an example using the `YCrCb` color space and HOG parameters described above:

![alt text][image11]
![alt text][image2]


```python
184     def extract_feature_array(self, image, color_code):
185         orient = 9
186         pix_per_cell = 8
187         cell_per_block = 3
188         features = []
189         # convert image to YCrCb color space
190         if color_code == 'BGR':
191             img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
192         else: img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
193         img = cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)
194         y,cr,cb = cv2.split(img)
195         # smooth y channel to remove noise
196         y = cv2.equalizeHist(y)
197         # calculate histogram over each image channel
198         ch0_hist = np.histogram(y, 32, (0,256))
199         ch1_hist = np.histogram(cr, 32, (0,256))
200         ch2_hist = np.histogram(cb, 32, (0,256))
201         for ch in ([y, cr, cb]):
202             # calculate hog features over each image channel
203             feature_array = hog(ch,
204                                 orientations=orient,
205                                 pixels_per_cell=(pix_per_cell,pix_per_cell),
206                                 cells_per_block=(cell_per_block, cell_per_block),
207                                 visualise=False,
208                                 feature_vector=False)
209             features.append(feature_array.ravel())
210         # stack HOG features, image and histogram to generate final features of the input image
211         features = np.hstack((features))
212         features = np.hstack((features, img.ravel(), ch0_hist[0], ch1_hist[0], ch2_hist[0]))
213         return features.reshape(1,-1)
```


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters but the ones that gives me better results were the ones described above. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the code located inside the `trainmodel()`function. This function check if the features were already extracted, if not then images of cars and not cars are readed and passed to the `extract_feature_array()` function to create the features. If features were already extracted then they are readed from disk.

Then features are scaled using the function `StandardScaler` from `sklearn.preprocessing` package, also the scaler is stored on diks for later use.

Data is split randomly using the function `train_test_split()` from the `sklearn.model_selection` package, as described below:

```python
268         # Split up data into randomized training and test sets
269         rand_state = np.random.randint(0, 100)
270         X_train, X_test, y_train, y_test = train_test_split(
271             scaled_X, y, test_size=0.2, random_state=rand_state)
```
This is the complete `trainmodel()` function.

```python
216     def trainmodel(self):
217         f_features = Path('features.p')
218         f_svc = Path('svc.p')
219 
220         features = {}
221 
222         if f_features.is_file() == False:
223 
224             # full set of images
225             filenames_cars = sorted(glob('vehicles/*/*.*'))
226             filenames_nocars = sorted(glob('non-vehicles/*/*.*'))
227             # Extract car features from training images
228             image = cv2.imread(filenames_cars[0])
229             features['cars'] = self.extract_feature_array(image, 'BGR')
230             print('Extracting cars features:')
231             for i in trange(1, len(filenames_cars)):
232                 image = cv2.imread(filenames_cars[i])
233                 feature_array = self.extract_feature_array(image, 'BGR')
234                 features['cars'] = np.append(features['cars'], feature_array, axis=0)
235                 # add flipped image
236                 image = cv2.flip(image, 1)
237                 feature_array = self.extract_feature_array(image, 'BGR')
238                 features['cars'] = np.append(features['cars'], feature_array, axis=0)
239             # Extract non-car features from training images
240             image = cv2.imread(filenames_nocars[0])
241             features['nocars'] = self.extract_feature_array(image, 'BGR')
242             print('Extracting non-cars features:')
243             for i in trange(1, len(filenames_nocars)):
244                 image = cv2.imread(filenames_nocars[i])
245                 feature_array = self.extract_feature_array(image, 'BGR')
246                 features['nocars'] = np.append(features['nocars'], feature_array, axis=0)
247                 # add flipped image
248                 image = cv2.flip(image, 1)
249                 feature_array = self.extract_feature_array(image, 'BGR')
250                 features['nocars'] = np.append(features['nocars'], feature_array, axis=0)
251             pickle.dump(features, open('features.p', 'wb'))
252         else:
253             print('Loading features from disk...')
254             features = pickle.load(open('features.p', 'rb'))
255         # Define a labels vector based on features lists
256         y = np.hstack((np.ones(len(features['cars'])),
257                        np.zeros(len(features['nocars']))))
258         # Create an array stack of feature vectors
259         X = np.vstack((features['cars'], features['nocars'])).astype(np.float64)
260         # Fit a per-column scaler
261         X_scaler = StandardScaler().fit(X)
262         self.scaler = X_scaler
263         # save scaler for later use
264         pickle.dump(X_scaler, open('scaler.p', 'wb'))
265         # Apply the scaler to X
266         scaled_X = X_scaler.transform(X)
267 
268         # Split up data into randomized training and test sets
269         rand_state = np.random.randint(0, 100)
270         X_train, X_test, y_train, y_test = train_test_split(
271             scaled_X, y, test_size=0.2, random_state=rand_state)
272 
273         # check if svm was already trained
274 
275         if f_svc.is_file() == False:
276             print('Training linear svm...')
277             svc = self.trainsvm(X_train, y_train)
278             # save trained model
279             pickle.dump(svc, open('svc.p', 'wb'))
280         else:
281             print('Model already trained, loading model from disk...')
282             svc = pickle.load(open('svc.p', 'rb'))
283             # check the accuracy of your classifier on the test dataset
284         print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
285 
286     # wrapper to train the linear svm model using train data 
287     def trainsvm(self, X_train, y_train):
288         # Use a linear SVC (support vector classifier)
289         svc = LinearSVC(max_iter=2000)
290         # Train the SVC
291         svc.fit(X_train, y_train)
292         return svc
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to divide the botton part of the image in horizontal sections of different heights. The heights were calculated based on the perspective towars the vanishing point, that's why i neede to use the camera file from previous project to identify the center of the camera on the image.

Below is the result of the division on an actual test image and the final set of windows:

![alt text][image3]

The code for the definition of the windows is located in the function `get_boxes()` inside the `detector` class:

```python

 44     # Calculate windows used in the sliding window process
 45     def get_boxes(self):
 46         #offset_base = 10
 47         #offset = 0
 48         imcy = self.imagecenter[1]
 49         ys_pairs = [[0+imcy, 76+imcy],
 50                     [0+imcy, 123+imcy],
 51                     [11+imcy, 76+imcy],
 52                     [11+imcy, 94+imcy],
 53                     [11+imcy, 123+imcy],
 54                     [11+imcy, 199+imcy],
 55                     [29+imcy, 76+imcy],
 56                     [29+imcy, 123+imcy]]
 57         shape = [720, 1280]
 58         boxes = []
 59         for pair in ys_pairs:
 60             w = int((pair[1]-pair[0]) * self.golden_ratio)
 61             x_start = 0
 62             shift = int(w * 0.5)
 63             while (x_start + w) <= shape[1]:
 64                 x = 20
 65                 y = 20
 66                 p1 = (x_start, pair[0])
 67                 p2 = (x_start+w, pair[1])
 68                 around = [[(0 if (p1[0]-x)<0 else (p1[0]-x),(p1[1]+y)),((p2[0]-x),(p2[1]+y)), 0],
 69                           [((p1[0]),(p1[1]+y)),((p2[0]),(p2[1]+y)), 0],
 70                           [((p1[0]+x),(p1[1]+y)),((p2[0]+x),(p2[1]+y)), 0],
 71                           [(0 if (p1[0]-x)<0 else (p1[0]-x),(p1[1])),((p2[0]-x),(p2[1])), 0],
 72                           [((p1[0]+x),(p1[1])),(1280 if (p2[0]+x)>1280 else (p2[0]+x),(p2[1])), 0],
 73                           [(0 if (p1[0]-x)<0 else (p1[0]-x),(p1[1]-y)),((p2[0]-x),(p2[1]-y)), 0],
 74                           [((p1[0]),(p1[1]-y)),((p2[0]),(p2[1]-y)), 0],
 75                           [((p1[0]+x),(p1[1]-y)),(1280 if (p2[0]+x)>1280 else (p2[0]+x),(p2[1]-y)), 0]]
 76                 box =  [p1, p2, 0, 4, around] # [rectangle, frame count, usage count decreasing]
 77                 boxes.append(box)
 78                 x_start += shift
 79         return boxes

```

Then, the code to perform the slide window technique is located in the `draw_boxes()` function inside the `detector` class.

```python
 81     # Function used to implement the sliding window technique
 82     def draw_boxes(self, dst, model, color_code):
 83         self.frame_count += 1
 84         if self.frame_count > self.max_frames: # process only after more that 5 frames 
 85             self.frame_count = 0
 86             # iterate over all the defined windows to find cars
 87             for box in self.boxes:
 88                 rec = dst[box[0][1]:box[1][1], box[0][0]:box[1][0]]
 89                 pred = model.predict(rec, color_code)
 90                 if pred == 1: # if car detected then put it on-box
 91                     if (box[0],box[1]) not in self.on_boxes:
 92                         self.on_boxes[(box[0],box[1])] = [0, box]
 93 
 94         # if we have detections
 95         if len(self.on_boxes) > 0:
 96             keys = list(self.on_boxes.keys())
 97             for k in keys:
 98                 box = self.on_boxes[k][1] # take the box to process it
 99                 rec = dst[box[0][1]:box[1][1], box[0][0]:box[1][0]]
100                 pred = model.predict(rec, color_code)
101                 parent_found = False
102                 if pred == 1: # if detected increase use counter for this box/window
103                     box[2] += 1
104                     parent_found = True
105                 child_found = False
106                 # iterate over all the child windows to try to confirm match
107                 for box_around in box[4]:
108                     rec = dst[box_around[0][1]:box_around[1][1], box_around[0][0]:box_around[1][0]]
109                     pred = model.predict(rec, color_code)
110                     if pred == 1:
111                         box[2] += 1
112                         box_around[2] = 1
113                         child_found = True
114                     else:
115                         box_around[2] = 0
116                 # if current window inside the on_box collection does not
117                 # gived a match and niether their child windows then remove it
118                 if not (child_found or parent_found):
119                     box[2] = 0
120                     del self.on_boxes[k]
121 
122         ## prcoess windows left on on_boxes with a score > 10
123         if len(self.on_boxes):
124             # create heat map
125             heat_map = np.zeros_like(dst[:,:,0])
126             for k in self.on_boxes:
127                 if self.on_boxes[k][1][2] > 10:
128                     box = self.on_boxes[k][1]
129                     # draw current window on heat_map
130                     heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
131                     # draw child windows that have score > 0 
132                     for box_around in box[4]:
133                         if box_around[2] == 1:
134                             heat_map[box_around[0][1]:box_around[1][1], box_around[0][0]:box_around[1][0]] += 1
135             # use the label function from scikit image
136             labels = label(heat_map)
137             # if labels were found, process them
138             if labels[1] > 0:
139                 for carnum in range(1, labels[1]+1):
140                     # find the upper left and lower right corners of the 
141                     # rectangle
142                     nonzero = (labels[0] == carnum).nonzero()
143                     nonzerox = np.array(nonzero[1])
144                     nonzeroy = np.array(nonzero[0])
145                     box = ((np.min(nonzerox), np.min(nonzeroy)),
146                         (np.max(nonzerox), np.max(nonzeroy)))
147                     # draw final window on image
148                     dst = cv2.rectangle(dst, box[0], box[1], (0,0,255), 2)
149         return dst


```


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on an image using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./results.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video in the `on_boxes` dictionary inside the `detector` class.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


```python
122         ## prcoess windows left on on_boxes with a score > 10
123         if len(self.on_boxes):
124             # create heat map
125             heat_map = np.zeros_like(dst[:,:,0])
126             for k in self.on_boxes:
127                 if self.on_boxes[k][1][2] > 10:
128                     box = self.on_boxes[k][1]
129                     # draw current window on heat_map
130                     heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
131                     # draw child windows that have score > 0 
132                     for box_around in box[4]:
133                         if box_around[2] == 1:
134                             heat_map[box_around[0][1]:box_around[1][1], box_around[0][0]:box_around[1][0]] += 1
135             # use the label function from scikit image
136             labels = label(heat_map)
137             # if labels were found, process them
138             if labels[1] > 0:
139                 for carnum in range(1, labels[1]+1):
140                     # find the upper left and lower right corners of the 
141                     # rectangle
142                     nonzero = (labels[0] == carnum).nonzero()
143                     nonzerox = np.array(nonzero[1])
144                     nonzeroy = np.array(nonzero[0])
145                     box = ((np.min(nonzerox), np.min(nonzeroy)),
146                         (np.max(nonzerox), np.max(nonzeroy)))
147                     # draw final window on image
148                     dst = cv2.rectangle(dst, box[0], box[1], (0,0,255), 2)

```

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


### Here are three frames and their corresponding heatmaps:

![alt text][image51]
![alt text][image52]
![alt text][image53]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part for me was how to select the windows to divide the image, too much windows make the processing of the images very slow and does not give good results, too little windows does not help to detect the car. I found the division of the image based on its perspective very useful and give me better results, but still there are frames where the windows do not capture the car and thus is not able to detect it.
EDIT:
I have noticed that the combination of HOG features with the color image and the histogram of color space gives better results that using only the HOG features, also when trying to identify which color space to use I have try RGB, LSH and YCrCb but was YCrCb which helped to pass from 97% to 98.8% in test accuracy, that was why I decided to use the later one. Not sure if adding more than one color space as features can improve the detection rate, this is something to try in the feature.

After working on the sliding window technique I think this is the part where more improvements can be made, the SVM model works very well when the image you try to identify is some way centered in the rectangle you are passing to the model and if your slide windows does not capture the object in any window or if it is partly captured, the model will not work as expected. To try to solve this problem I used a focused search around the main windows, this means that I have defined parent windows and for each parent window they have child windows around with a shift of 20 pixels in all directions. This helps a lot but I think this can also be improved making the child windows parent windows each time a child window has a detection, in this way I could be like a recursive window positioning that can vary position depending on the detection a not some fixed windows placement. This was not implemented due to time constraints.

In general the linear SVM model worked very well, so I think the use of other models using deep learning can improve the detection rate and also help on removing the need to have the car very well centered in the window because the NN can learn other details of the car that can be detected partially.
