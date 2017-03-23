# python 3.5
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from sys import exit
from random import randint
from sklearn.model_selection import KFold
from tqdm import trange

IMGROWS = 120
IMGCOLS = 320
IMGCHAN = 3

EPOCH = 10

# Class used to pass chuncks of images and labels to the
# keras fit_generator function
class get_data:
    def __init__(self, datafile):
        self.last_index = 0
        try: # open dataset
            self.h5 = h5py.File(datafile, 'r')
        except OSError as e:
            print(e)
            exit(-1)
        self.size = self.h5['labels'].shape[0]
        print('open data done!')
    # generator definition
    def next_batch(self, train, batch_size=128):
        i = 0
        train = list(train)
        iter_by_epoch = int(len(train) / batch_size)
        while 1:
            lower_idx = i*batch_size
            upper_idx = lower_idx + batch_size
            imgs = self.h5['images'][train[lower_idx:upper_idx]]
            imgs = imgs.astype(np.float32)/255.0
            labels = self.h5['labels'][train[lower_idx:upper_idx]]
            yield (imgs, labels)
            i += 1
            if i >= iter_by_epoch:
                i = 0


 # CNN model definition
def get_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, # 5 by 5 kernel with 24 filters
                        border_mode='valid',
                        subsample=(2,2), # strides
                        dim_ordering='tf', # use tf ordering for the sample shape
                        input_shape=(IMGROWS, IMGCOLS, IMGCHAN),
                        activation='elu')) # ELU Activation
    model.add(Convolution2D(36, 5, 5, # 5 by 5 kernel with 36 filters
                        border_mode='valid',
                        subsample=(2,2), # strides
                        dim_ordering='tf',
                        activation='elu'))
    model.add(Convolution2D(48, 5, 5, # 5 by 5 kernel with 48 filters
                        border_mode='valid',
                        subsample=(2,2), # strides
                        dim_ordering='tf',
                        activation='elu'))
    model.add(Convolution2D(64, 3, 3, # 3 by 3 kernel with 64 filters
                        border_mode='valid',
                        subsample=(2,2), # strides
                        dim_ordering='tf',
                        activation='elu'))
    model.add(Convolution2D(64, 3, 3, # 3 by 3 kernel with 64 filters
                        border_mode='valid',
                        subsample=(1,1), # strides
                        dim_ordering='tf'))
    # Adding Batch Normalization layer
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    model.add(Flatten()) # Flatten the layer to create first fully connected layer
    model.add(Dense(256)) # First fully connected layer
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    model.add(Dropout(0.4)) # add dropout layer to avoid overfitting
    model.add(Dense(128))  # Second fully connected layer
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    model.add(Dropout(0.4)) # additional dropout layer to avoid overfitting
    model.add(Dense(64)) # third fully connected layer
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    model.add(Dense(16)) # fifth fully connected layer
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    model.add(Dense(1)) # output layer
    #
    return model



if __name__ == '__main__':
    histsummary = np.array(())
    # open training and test datasets
    datatrain = get_data('train.h5')
    datatest = get_data('test.h5')
    # open trained model
    if os.path.exists('./model.h5'):
        model = load_model("model.h5")
        print("Loaded model from disk")
    else: # if model was not trained before, so create one
        model = get_model()
    # define adam optimizer to start training the network
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    # comile model and use mse as the loss function
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    # use K Fold cross validation
    kf = KFold(n_splits=128)
    kfgen = kf.split(datatrain.h5['images'])
    for i in range(EPOCH):
        train, test = kfgen.__next__()
        history = model.fit_generator(datatrain.next_batch(train),
                            samples_per_epoch=datatrain.size,
                            nb_epoch=1,
                            verbose=1)
        histsummary = np.append(histsummary, history.history['loss'])
    # evaluate model using test dataset
    testimgs = datatest.h5['images']
    testimgs = testimgs.astype(np.float32)/255.0
    testlabels = datatest.h5['labels']
    score = model.evaluate(testimgs,
                            testlabels,
                            batch_size=128,
                            verbose=1)
    print('Test score: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))

    # save trained model to disk
    model.save('model.h5')
    print("Saved model to disk")
    # plot loss history to see if model was decreasing loss error
    plt.plot(histsummary, '-b')
    plt.xlabel('epoch')
    plt.ylabel('Training loss')
    plt.title('Training History')
    plt.grid(True)
    plt.show()
