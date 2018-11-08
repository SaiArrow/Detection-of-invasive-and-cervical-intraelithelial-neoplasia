import cv2
import numpy as np
from sklearn import preprocessing
import os
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

seed=42

#img_dir = "ImageDivision/Temp2/"
#labels = ['abnormal','normal']
# img_dir = "ImageDivision/Temp/"
# labels = ['carcinoma_in_situ','moderate_dysplastic','normal_superficiel']
img_dir = "ImageDivision/Red and Blue/"
labels = ['carcinoma_in_situ','light_dysplastic','moderate_dysplastic','normal_columnar','normal_intermediate','normal_superficiel','severe_dysplastic']
encoder = preprocessing.LabelEncoder()
encoder.fit(labels)

ratios=[]
labels=[]
cyto_rat = []
kern_ratio = []
def main():

    for subdir, dirs, files in os.walk(img_dir):
        #label name
        l = str(subdir).split('/')[2]

        for file in files:
            le=encoder.transform([str(l)])[0]
            img = subdir + os.sep + file

            r,k,c=calc_ratio(img)

            ratios.append(r)
            kern_ratio.append(k)
            cyto_rat.append(c)
            labels.append(le)

    print('Labels:',labels)
    print('Ratios:',ratios)
    print("KerneElong :",kern_ratio)
    print("CytoElong :",cyto_rat)
    print(len(ratios))
    print(len(kern_ratio))
    print(len(cyto_rat))
    X=[]
    y=labels
    for i in range(0,len(ratios)):
        temp=[]
        temp.append(ratios[i])
        temp.append(kern_ratio[i])
        temp.append(cyto_rat[i])
        X.append(temp)
    print('X',X)
    print('y',y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)


    Models = [KNeighborsClassifier(n_neighbors=5), SVC(kernel="linear", C=0.08), MLPClassifier(alpha=1),
              DecisionTreeClassifier(max_depth=5), GaussianNB(), QuadraticDiscriminantAnalysis(),
              GaussianProcessClassifier(1.0 * RBF(1.0)), LogisticRegression(random_state=0, solver="liblinear")]
    for i in Models:
        i.fit(X_train, y_train)
        y = i.predict(X_test)
        print(accuracy_score(y_test, y))
        #cm = confusion_matrix(y_test, y)
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #cax = ax.matshow(cm)
        #plt.title('Confusion matrix of the classifier')
        #fig.colorbar(cax)
        #plt.xlabel('Predicted')
        #plt.ylabel('True')
        #plt.show()


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

            ellipse = cv2.fitEllipse(big_contour)
            cv2.ellipse(f, ellipse, (98, 255, 100), 2)
            #cv2.imshow('ellipse', f)
            #k = cv2.waitKey(2000) & 0xFF

            return (MA/ma)


def calc_ratio(img):
            frame = cv2.imread(img)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #BGR
            lower_nn = np.array([30,150,50])
            upper_nn = np.array([255,255,180])
            mask = cv2.inRange(hsv, lower_nn, upper_nn)
            #cv2.imshow('mask',mask)
            #k = cv2.waitKey(2000) & 0xFF
            cv2.imwrite("mask.bmp",mask)

            lower_wn = np.array([110,100,100])
            upper_wn = np.array([130,255,255])
            mask = cv2.inRange(hsv, lower_wn, upper_wn)
            #cv2.imshow('temp',mask)
            #k = cv2.waitKey(2000) & 0xFF
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
            cv2.imwrite("ellipse_k.bmp", f)
            cyto_rat = calc_elon(img1)
            cv2.imwrite("ellipse_c.bmp", img1)

            return r,ker_rat,cyto_rat


if __name__ == '__main__':
        main()
