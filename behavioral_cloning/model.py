# python

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers import Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D
from keras.optimizers import SGD, Adam
#from keras.models import model_from_json
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

EPOCH = 5

class get_data:
    def __init__(self, datafile):
        self.last_index = 0
        try:
            self.h5 = h5py.File(datafile, 'r')
            #self.h5test = h5py.File(test, 'r')
        except OSError as e:
            print(e)
            exit(-1)
        self.size = self.h5['labels'].shape[0]
        #self.test_size = self.h5test['labels'].shape[0]
        print('open data done!')
        
    def next_batch(self, train, batch_size=128):
        i = 0
        train = list(train)
        iter_by_epoch = int(len(train) / batch_size)
        while 1:
            #imgs = self.h5['images'][self.last_index:self.last_index+batch_size]
            lower_idx = i*batch_size
            upper_idx = lower_idx + batch_size
            imgs = self.h5['images'][train[lower_idx:upper_idx]]
            imgs = imgs.astype(np.float32)/255.0
            #labels = self.h5['labels'][self.last_index:self.last_index+batch_size]
            labels = self.h5['labels'][train[lower_idx:upper_idx]]
            yield (imgs, labels)
            i += 1
            if i >= iter_by_epoch:
                i = 0


 
def get_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, # 3 by 3 kernel with 16 filters
                        border_mode='valid', 
                        subsample=(2,2), # strides
                        dim_ordering='tf', # use tf ordering for the sample shape
                        input_shape=(IMGROWS, IMGCOLS, IMGCHAN),
                        activation='elu'))
    #model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, # 3 by 3 kernel with 16 filters
                        border_mode='valid', 
                        subsample=(2,2), # strides
                        dim_ordering='tf', # use tf ordering for the sample shape
                        activation='elu'))
    #model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, # 3 by 3 kernel with 16 filters
                        border_mode='valid', 
                        subsample=(2,2), # strides
                        dim_ordering='tf', # use tf ordering for the sample shape
                        activation='elu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, 3, 3, # 3 by 3 kernel with 16 filters
                        border_mode='valid', 
                        subsample=(2,2), # strides
                        dim_ordering='tf', # use tf ordering for the sample shape
                        activation='elu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, 3, 3, # 3 by 3 kernel with 16 filters
                        border_mode='valid', 
                        subsample=(1,1), # strides
                        dim_ordering='tf')) # use tf ordering for the sample shape
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    #model.add(Convolution2D(128,3,3, # 3 by 3 kernel with 16 filters
    #                        border_mode='same', 
    #                        subsample=(1,1), # strides
    #                        dim_ordering='tf', # use tf ordering for the sample shape
    #                        input_shape=(IMGROWS, IMGCOLS, IMGCHAN),
    #                        activation='relu'))

    #model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    #model.add(Dense(8192))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(4096))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    #model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    #model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    model.add(Dropout(0.4))
    model.add(Dense(128))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    model.add(Dense(16))
    model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99))
    model.add(Activation('elu'))
    model.add(Dense(1))
    #model.add(Activation('tanh'))
    #
    return model



if __name__ == '__main__':
    histsummary = np.array(())
    datatrain = get_data('train.h5')
    datatest = get_data('test.h5')
    if os.path.exists('./model.h5'):
        #### load model
        #json_file = open('./model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #model = model_from_json(loaded_model_json)
        # load weights into new model
        model = load_model("model.h5")
        print("Loaded model from disk")
    else:
        model = get_model()


#    data = pickle.load(open('train.p', 'rb'))
#    data['features'] = data['features'].astype(np.float32)
#    data['features'] = data['features']/255.0
#
#    test = pickle.load(open('test.p', 'rb'))
#    test['features'] = test['features'].astype(np.float32)
#    test['features'] = test['features']/255.0
#

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

    #checkpointer = ModelCheckpoint(filepath='model_weights.h5', verbose=1,
    #                                save_best_only=False,
    #                                save_weights_only=True)

    kf = KFold(n_splits=128)
    kfgen = kf.split(datatrain.h5['images'])
    for i in trange(EPOCH):
        train, test = kfgen.__next__()
        history = model.fit_generator(datatrain.next_batch(train), 
                            samples_per_epoch=datatrain.size, 
                            nb_epoch=1, 
                            verbose=1)
        histsummary = np.append(histsummary, history.history['loss'])
                            #callbacks=[checkpointer])
    #history = model.fit(data['features'][0:1000].reshape(*data['features'][0:1000].shape, 1), 
    #                    data['steering'][0:1000], 
    #                    batch_size=128, nb_epoch=10, 
    #                    verbose=1, 
    #                    validation_split=0.2, 
    #                    shuffle=True)

    #score = model.evaluate_generator(datatest.next_batch(128),
    #                        val_samples=datatest.size)
    #print('Test score: {}'.format(score[0]))
    #print('Test accuracy: {}'.format(score[1]))


    #plot training history
    #plt.plot(history.history['val_loss'], '-r')
    #plt.plot(history.history['loss'], '-b')
    #plt.legend(['val_loss', 'training_loss'])
    plt.plot(histsummary, '-b')
    plt.xlabel('epoch')
    plt.ylabel('Training loss')
    plt.title('Training History')
    plt.grid(True)
    plt.show()

    #### serialize model to JSON
    #model_json = model.to_json()
    #with open("model.json", "w") as json_file:
    #    json_file.write(model_json)
    # serialize weights to HDF5
    #model.save_weights("model.h5")
    model.save('model.h5')
    print("Saved model to disk")


