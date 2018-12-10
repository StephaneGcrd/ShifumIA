import numpy as np
from skimage import data, io
import cv2 as cv

from keras.utils import to_categorical
import glob



class Load():
    def __init__(self, imageHeight, imageWidth, NbPicturesToLoad):
        self.DATA_X = []
        self.DATA_LABELS = []
        self.DATA_X_TEMPORARY = []
        self.NUMBER_PHOTO_PAPER = 0;
        self.NUMBER_PHOTO_ROCK = 0;
        self.NUMBER_PHOTO_SCISSORS = 0;
        self.NAME_PATH_PAPER = "images_paper/"
        self.NAME_PATH_ROCK = "images_rock/"
        self.NAME_PATH_SCISSORS = "images_scissors/"
        self.IMAGE_HEIGHT = imageHeight;
        self.IMAGE_WIDTH = imageWidth;
        self.NB_PHOTO_MAX_TO_LOAD = NbPicturesToLoad;

    def LoadImagesFromFolder(self, NAME_PATH):

        for img in glob.glob(NAME_PATH + "/*.png"):
            # pour chague image .png dans le dossier,



            if len(self.DATA_X_TEMPORARY) == 0:
                #Si l'array est vide, alors on charge la premiere photo dans l'array temporaire
                self.DATA_X_TEMPORARY = self.LoadImage(img)
            else:
                #Sinon on charge l'image
                ImageLoaded = self.LoadImage(img)
                #Puis on l'ajoute a l'array temporaire en la concatenant a cet array
                self.AddImageToDATAX(ImageLoaded)

            # On incremente le nombre de photos
            self.IncrementCountImage(NAME_PATH)
            # Si ce nombre depasse le seuil de photo qu'on a defini, la boucle s'arrete
            if self.CheckIfNumberIsMax(NAME_PATH): break;

    def LoadImage(self, img):
        Image = io.imread(img)

        Image = np.expand_dims(Image, axis=0)
        return Image



    def AddImageToDATAX(self, img):
        self.DATA_X_TEMPORARY = np.concatenate((self.DATA_X_TEMPORARY, img), axis=0)

        #Si l'array temporaire est complet, alors on le charge dans l'array DATA_X et on le vide
        if self.DataXTemporaryIsFull():
            self.AddDataXTemporaryToDataX()
            self.FlushDataXTemporary()

    def DataXTemporaryIsFull(self): return len(self.DATA_X_TEMPORARY) % 50 == 0
    def FlushDataXTemporary(self): self.DATA_X_TEMPORARY = []

    def AddDataXTemporaryToDataX(self):
        if len(self.DATA_X) == 0:
            self.DATA_X = self.DATA_X_TEMPORARY
        else:
            self.DATA_X = np.concatenate((self.DATA_X, self.DATA_X_TEMPORARY), axis=0)


    def IncrementCountImage(self, ImageCategory):

        if ImageCategory == self.NAME_PATH_PAPER:
            self.NUMBER_PHOTO_PAPER = self.NUMBER_PHOTO_PAPER + 1

        elif ImageCategory == self.NAME_PATH_ROCK:
            self.NUMBER_PHOTO_ROCK = self.NUMBER_PHOTO_ROCK + 1

        elif ImageCategory == self.NAME_PATH_SCISSORS:
            self.NUMBER_PHOTO_SCISSORS = self.NUMBER_PHOTO_SCISSORS + 1

        else:
            print("Error : the path is not corresponding in the code and function call")
            print("Function call = "+ ImageCategory)
            print("code param = "+ self.NAME_PATH_SCISSORS +"  "+self.NAME_PATH_ROCK+"   "+self.NAME_PATH_PAPER)

    def CheckIfNumberIsMax(self, ImageCategory):
        if ImageCategory == self.NAME_PATH_PAPER:
            NumberToCheck = self.NUMBER_PHOTO_PAPER

        elif ImageCategory == self.NAME_PATH_ROCK:
            NumberToCheck = self.NUMBER_PHOTO_ROCK

        elif ImageCategory == self.NAME_PATH_SCISSORS:
            NumberToCheck = self.NUMBER_PHOTO_SCISSORS

        else:
            print("Error : the path is not corresponding in the code and function call")
            print("Function call = "+ ImageCategory)
            print("code param = "+ self.NAME_PATH_SCISSORS +"  "+self.NAME_PATH_ROCK+"   "+self.NAME_PATH_PAPER)

        if NumberToCheck == self.NB_PHOTO_MAX_TO_LOAD:
            return True
        else:
            return False


    def createLabels(self):
        LabelRock = 1
        LabelScissors = 2

        N_PHOTO_TOTAL = self.NUMBER_PHOTO_PAPER + self.NUMBER_PHOTO_ROCK + self.NUMBER_PHOTO_SCISSORS;
        self.DATA_LABELS = np.zeros(N_PHOTO_TOTAL)
        PosEndPaperInArray = self.NUMBER_PHOTO_PAPER + 1
        PosEndRockInArray = PosEndPaperInArray + self.NUMBER_PHOTO_ROCK
        self.DATA_LABELS[PosEndPaperInArray:PosEndRockInArray] = LabelRock
        self.DATA_LABELS[PosEndRockInArray + 1:] = LabelScissors
        self.DATA_LABELS = to_categorical(self.DATA_LABELS)
        print self.DATA_LABELS

    def LoadAllPicturesAndLabels(self):
        self.LoadImagesFromFolder("images_paper/")
        self.LoadImagesFromFolder("images_rock/")
        self.LoadImagesFromFolder("images_scissors/")
        self.createLabels()






