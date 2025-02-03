### Compare Algorithms

# Classification
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
filename = '../Data/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# prepare models
models = []
models.append(('Logistic Regression (LRC)', LogisticRegression()))
models.append(('Linear Discriminant Analysis (LDA)', LinearDiscriminantAnalysis()))
models.append(('Gaussian Naive Bayes (GNB)', GaussianNB()))
models.append(('KNN Classification', KNeighborsClassifier()))
models.append(('Decision Tree Classification', DecisionTreeClassifier()))
models.append(('SVM Classification (SVC)', SVC()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Classification Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Get more detailed evaluation of best performing model
# Split into training and validation datasets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=7)

# Fit the model
best_model = LogisticRegression()
best_model.fit(X_train, Y_train)

# Make predictions
predictions = best_model.predict(X_validation)

# Evaluate predictions
print("Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_validation, predictions))
print("\nClassification Report:")
print(classification_report(Y_validation, predictions))

# Regression
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# load dataset
filename = '../Data/housing.csv'
names = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# prepare models
models = []
models.append(('Linear Regression (LR)', LinearRegression()))
models.append(('Lasso Regression (L1)', Lasso()))
models.append(('Ridge Regression (L2)', Ridge()))
models.append(('ElasticNet Regression (L1+L2)', ElasticNet()))
models.append(('KNN Regression', KNeighborsRegressor()))
models.append(('Decision Tree Regression', DecisionTreeRegressor()))
models.append(('SVM Regression (SVR)', SVR()))

# evaluate each model in turn
results = []
names = []
scoring = 'neg_mean_squared_error'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Regression Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Evaluate best model in more detail
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=7)

best_model = LinearRegression()
best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_validation)

print("\nMean Squared Error:")
print(mean_squared_error(Y_validation, predictions))

print("\nModel Coefficients:")
for name, coef in zip(names[:-1], best_model.coef_):
   print(f"{name}: {coef:.4f}")