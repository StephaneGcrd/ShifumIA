import cv2
import time
import os
from keras.optimizers import SGD
from keras.models import model_from_json

from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import pylab
import cv2
import time
import scipy.signal
from skimage import img_as_ubyte




def convolve2d(image, kernel):
    # This function which takes an image and a kernel
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).

    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output = np.zeros_like(image)  # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):  # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y:y + 3, x:x + 3]).sum()
    return output


def show_webcam(mirror=False):
    i = 0
    json_file = open('model_x.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    CnnModel = model_from_json(loaded_model_json)
    # load weights into new model
    CnnModel.load_weights("model_x.h5")
    print("Loaded model from disk")

    CnnModel.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    time.sleep(2)
    cam = cv2.VideoCapture(0)

    while True:
        ret_val, img = cam.read()
        if mirror:
            img_init = cv2.flip(img, 1)
            img = img_init[0:500, 0:500]
# Convert the image to grayscale (1 channel)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.medianBlur(img,15)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            cv2.rectangle(img_init, (0, 0), (500, 500), (0, 255, 0), 5)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img_init, (0, 590), (100, 620), (0, 0, 0), 50)

            img = cv2.resize(img, (128, 128))



            DATA_X = np.expand_dims(img, axis=0)
            DATA_X = np.expand_dims(DATA_X, axis=3)

            score = CnnModel.predict(DATA_X);
            bottomLeftCornerOfText = (10, 600)
            fontScale = 1
            fontColor = (0, 255, 0)
            lineType = 2

            cv2.putText(img_init, print_pred(score),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cv2.imshow('my webcam', img_init)
            time.sleep(0.1)
            i += 1


        if cv2.waitKey(1) == 27:

            break  # esc to quit

    cv2.destroyAllWindows()

def print_pred(score):
    """
    print("Papier : "+str(score[0][0]*100)+" %")
    print("Pierre : " + str(score[0][1] * 100) + " %")
    print("Ciseaux : " + str(score[0][2] * 100) + " %")
    """
    if (score[0][0] > score[0][1]) and (score[0][0] > score[0][2]):
        return "Papier"
    elif (score[0][1] > score[0][0]) and (score[0][1] > score[0][2]):
        return "Pierre"
    else:
        return "Ciseaux"



def main():


    show_webcam(mirror=True)


if __name__ == '__main__':
    main()