import cv2
import numpy as np


def calc_elon(f):
            ret,th = cv2.threshold(f,127,255, 0)
            #--- Find all the contours in the binary image ---
            _, contours,hierarchy = cv2.findContours(th,2,1)
            cnt = contours
            big_contour = []
            max1 = 0
            for i in cnt:
               area = cv2.contourArea(i) #--- find the contour having biggest area ---
               if(area > max1):
                    max1 = area
                    big_contour = i

            x_dim = []
            y_dim = []
            for i in big_contour:
                x_dim.append(i[0][0])
                y_dim.append(i[0][1])
            (x,y),(MA,ma),angle = cv2.fitEllipse(big_contour)
            return (MA/ma)

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

            img1 = cv2.imread('mask1.bmp', cv2.IMREAD_GRAYSCALE)
            wn_white_pix = np.sum(img1 == 255)
            #print('Number of white pixels:', wn_white_pix)

            r=(wn_white_pix-n_white_pix)/float(wn_white_pix)
            cv2.destroyAllWindows()

            f = np.bitwise_xor(img,img1)

            ker_rat = calc_elon(f)
            cyto_rat = calc_elon(img1)

            return r,ker_rat,cyto_rat


print(calc_ratio("ImageDivision/Temp/carcinoma_in_situ/149056321-149056343-002-d.bmp"))
