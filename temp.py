import cv2
import numpy as np
from sklearn import preprocessing
import os

img_dir = "ImageDivision/Temp/"

labels = ['light_dysplastic','normal_intermediate','severe_dysplastic']
encoder = preprocessing.LabelEncoder()
encoder.fit(labels)

ratios=[]
labels=[]

def main():

    for subdir, dirs, files in os.walk(img_dir):
        #label name
        l = str(subdir).split('/')[2]

        for file in files:
            le=encoder.transform([str(l)])[0]
            img = subdir + os.sep + file

            r=calc_ratio(img)

            ratios.append(r)
            labels.append(le)

    # print('Labels:',labels)
    # print('Ratios:',ratios)
    l  = set(labels)
    diff = []
    for i in l:
        temp = []
        for j in range(0,len(ratios)):
            if(labels[j]==i):
                temp.append(ratios[j])
        diff.append(float(sum(temp))/len(temp))

    print(diff)

def calc_ratio(img):
            frame = cv2.imread(img)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #BGR
            lower_nn = np.array([30,150,50])
            upper_nn = np.array([255,255,180])
            mask = cv2.inRange(hsv, lower_nn, upper_nn)
            # cv2.imshow('mask',mask)
            #k = cv2.waitKey(10000) & 0xFF
            cv2.imwrite("mask.bmp",mask)

            lower_wn = np.array([110,100,100])
            upper_wn = np.array([130,255,255])
            mask = cv2.inRange(hsv, lower_wn, upper_wn)
            # cv2.imshow('mask',mask)
            #k = cv2.waitKey(10000) & 0xFF
            cv2.imwrite("mask1.bmp",mask)


            img = cv2.imread('mask.bmp', cv2.IMREAD_GRAYSCALE)
            n_white_pix = np.sum(img == 255)
            #print('Number of white pixels:', n_white_pix)

            img = cv2.imread('mask1.bmp', cv2.IMREAD_GRAYSCALE)
            wn_white_pix = np.sum(img == 255)
            #print('Number of white pixels:', wn_white_pix)

            r=(wn_white_pix-n_white_pix)/float(wn_white_pix)
            cv2.destroyAllWindows()
            return r

if __name__ == '__main__':
        main()
