import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from pandas.plotting import scatter_matrix
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")



seed = 42

df=pd.read_csv("AJA/risk_factors_cervical_cancer.csv")
df = df.drop(columns="STDs: Time since last diagnosis")
df = df.drop(columns="STDs: Time since first diagnosis")




columns = list(df.keys())
features = [[-5]*len(df['Biopsy'])]*len(columns)
for i in range(0,len(columns)):
    j = np.where(np.array(df[columns[i]])=='?')
    temp = []
    for k in range(0,len(df[columns[i]])):
        if not (k in j[0]):
            temp.append(float(df[columns[i]][k]))
            features[i][k] = float(df[columns[i]][k])
    features[:] = [x if x != -5 else float(sum(temp))/len(temp) for x in features]


X = []
for i in range(0,len(features[0])):
    temp = []
    for j in range(0,len(features)):
        temp.append(features[j][i])
    X.append(temp)

y=df["Biopsy"]


def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Cervical Risk Feature Correlation')
    labels=list(df.keys())
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    fig.colorbar(cax)
    plt.show()

correlation_matrix(df)



X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=seed)


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


Models = [KNeighborsClassifier(n_neighbors=5),SVC(kernel="linear", C=0.05),MLPClassifier(alpha=1),DecisionTreeClassifier(max_depth=5),GaussianNB(),QuadraticDiscriminantAnalysis(),GaussianProcessClassifier(1.0 * RBF(1.0)),LogisticRegression(random_state=0,solver="liblinear")]


for i in Models:
    i.fit(X_train,y_train)
    y = i.predict(X_test)
    print(accuracy_score(y_test,y))
    cm = confusion_matrix(y_test, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
