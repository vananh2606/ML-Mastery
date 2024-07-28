# Load CSV Using Python Standard Library
import csv
import numpy
filename = '../Data/pima-indians-diabetes.data.csv'
raw_data = open(filename, "rt", encoding='utf-8')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
print(data.shape)

# Load CSV using NumPy
from numpy import loadtxt
filename = '../Data/pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rb')
data = loadtxt(raw_data, delimiter=",")
print(data.shape)

# Load CSV using Pandas
from pandas import read_csv
filename = '../Data/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print(data.shape)

