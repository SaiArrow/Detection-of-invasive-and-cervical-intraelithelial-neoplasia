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

import seaborn as sns
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


columns = list(df.keys())
features = []
for i in range(0,len(columns)):
    j = np.where(np.array(df[columns[i]])=='?')
    temp = []
    # print(j[0])
    for k in range(0,len(df[columns[i]])):
        if not (k in j[0]):
            # print(i,k)
            temp.append(float(df[columns[i]][k]))
        else :
            temp.append(float(sum(temp)/len(temp)))
    features.append(temp)

print(features)
X = []
for i in range(0,len(features[0])):
    temp = []
    for j in range(0,len(features)):
        temp.append(features[j][i])
    X.append(temp)

y=df["Biopsy"]


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
