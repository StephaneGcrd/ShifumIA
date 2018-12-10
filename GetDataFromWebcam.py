import cv2
from skimage import data, io, morphology, filters,segmentation
import time
from skimage.transform import rescale,resize, downscale_local_mean
import imutils
import numpy as np



def show_webcam(mirror=False):
    i = 500
    base = i
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img_init = cv2.flip(img, 1)
            img = img_init[0:500, 0:500]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.medianBlur(img,15)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            #img_init[0:500, 0:500] = np.expand_dims(img, axis=3)
            img = cv2.resize(img, (128, 128))
            cv2.imshow('my webcam', img)
            io.imsave("images_rock/"+str(i)+".png", img)
            time.sleep(0.05)
            i += 1
            print(i)

        if (i == base+500):
            break


        if cv2.waitKey(1) == 27:

            break  # esc to quit

    cv2.destroyAllWindows()




def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()