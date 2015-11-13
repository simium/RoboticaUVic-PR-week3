import numpy as np
from scipy.linalg import lstsq
from sklearn import linear_model

# First define some handy shortcuts
dot = np.dot
inv = np.linalg.inv

print 'Solving Q2:'
loadedData = np.array([map(float, l.split()) for l in open('housing.data')])
trainingData = np.array([map(float, l.split()) for l in open('housing_training.data')])
testingData = np.array([map(float, l.split()) for l in open('housing_testing.data')])
y = loadedData[:,-1]
print len(y),'data points in housing.data'

# Dummy features
X = np.ones((len(y),1))

# Fitting the parameters: theta = (X'*X)^-1*X'*y
theta = dot(dot(inv(dot(X.T, X)), X.T), y)
print 'theta =', theta

# MSE = (1/N)*sum((y-X*theta)^2)
MSE = sum((y-dot(X, theta))**2) / len(y)
print 'MSE =', MSE
