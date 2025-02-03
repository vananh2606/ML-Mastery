# Logistic Regression (Classification)
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = "../Data/pima-indians-diabetes.data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=None)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(f"Logistic Regression (LRC): {results.mean()}")

# Linear Discriminant Analysis
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

filename = "../Data/pima-indians-diabetes.data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=None)
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
print(f"Linear Discriminant Analysis (LDA): {results.mean()}")

# K-Nearest Neighbors Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

filename = "../Data/pima-indians-diabetes.data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=None)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(f"K-Nearest Neighbors Classification: {results.mean()}")

# Gaussian Naive Bayes
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

filename = "../Data/pima-indians-diabetes.data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = KFold(n_splits=10, random_state=None)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print(f"Gaussian Naive Bayes (GNB): {results.mean()}")

# Decision Tree Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

filename = "../Data/pima-indians-diabetes.data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = KFold(n_splits=10, random_state=None)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(f"Decision Tree Classification: {results.mean()}")

# Support Vector Machines Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

filename = "../Data/pima-indians-diabetes.data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = KFold(n_splits=10, random_state=None)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(f"Support Vector Machines Classification (SVC): {results.mean()}")
