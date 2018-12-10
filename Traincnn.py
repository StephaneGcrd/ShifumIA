"""
Ce programme permet de creer (ou charger) et entrainer un reseau de neurones

"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, \
    BatchNormalization, Conv2D, LeakyReLU, ZeroPadding2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.models import model_from_json
from loaddata import Load

class Train_Model():
    def __init__(self,ModelName,nbPhotos,lrate,epochs):
        self.DATA_X = []
        self.DATA_Y = []
        self.nbPhotos= nbPhotos
        self.ModelName= ModelName
        self.num_train = 0
        self.height = 0
        self.width =0
        self.depth=0
        self.lrate = lrate
        self.epochs = epochs
        self.model = None



    def LoadData(self):
        print("Loading Data = ")
        L = Load(128,128,self.nbPhotos)
        L.LoadAllPicturesAndLabels()
        plt.imshow(L.DATA_X[1050])
        plt.show()
        print(L.DATA_LABELS[1050])
        self.DATA_Y = L.DATA_LABELS
        self.DATA_X = np.expand_dims(L.DATA_X, axis=3)
        self.num_train, self.height, self.width, self.depth = self.DATA_X.shape

        print("Train array shape = " + str(self.DATA_X.shape))



    def CreateModel(self):
        print("Model init")
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.DATA_X.shape[1:], padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(3, activation='softmax'))

        decay = self.lrate/self.epochs
        sgd = SGD(lr=self.lrate, momentum=0.9, decay=decay, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model = model
        print("Model Created")




    def LoadModel(self):
        json_file = open(self.ModelName+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        CnnModel = model_from_json(loaded_model_json)
        # load weights into new model
        CnnModel.load_weights(self.ModelName+".h5")
        print("Loaded model ( "+self.ModelName+" ) from disk")


        decay = self.lrate/self.epochs
        sgd = SGD(lr=self.lrate, momentum=0.9, decay=decay, nesterov=False)
        CnnModel.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        self.model = CnnModel

    def Train(self):
        self.model.fit(self.DATA_X, self.DATA_Y, batch_size=32, epochs=self.epochs, shuffle=True)

    def SaveModel(self):
        model_json = self.model.to_json()
        with open(self.ModelName+'.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(self.ModelName+".h5")
        print("Saved model ( "+self.ModelName+" ) to disk")



CNN = Train_Model("model_x",1000,0.0004,3)
CNN.LoadData()

####### Choisir entre "CreateModel" ou "LoadModel"
#CNN.CreateModel()
CNN.LoadModel()

#######

CNN.Train()
CNN.SaveModel()