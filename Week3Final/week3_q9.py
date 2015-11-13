import sys
import numpy as np
from scipy.linalg import lstsq
from sklearn import linear_model
from scipy.optimize import fmin_l_bfgs_b
import pylab

# First define some handy shortcuts
dot = np.dot
inv = np.linalg.inv

loadedData = np.array([map(float, l.split()) for l in open('housing.data')])
trainingData = np.array([map(float, l.split()) for l in open('housing_training.data')])
testingData = np.array([map(float, l.split()) for l in open('housing_testing.data')])
y = loadedData[:,-1]
#print len(y),'data points in housing.data'

print '\nSolving 9:'

dataCRIMCol = 0
dataZNCol = 1
dataINDUSCol = 2
dataCHASCol = 3
dataNOXCol = 4
dataRMCol = 5
dataAGECol = 6
dataDISCol = 7
dataRADCol = 8
dataTAXCol = 9
dataPTRATIOCol = 10
dataBCol = 11
dataLSTATCol = 12
dataMEDVCol = 13

columnCRIM = trainingData[:,dataCRIMCol:dataCRIMCol+1]
columnZN = trainingData[:,dataZNCol:dataZNCol+1]
columnINDUS = trainingData[:,dataINDUSCol:dataINDUSCol+1]
columnCHAS = trainingData[:,dataCHASCol:dataCHASCol+1]
columnNOX = trainingData[:,dataNOXCol:dataNOXCol+1]
columnRM = trainingData[:,dataRMCol:dataRMCol+1]
columnAGE = trainingData[:,dataAGECol:dataAGECol+1]
columnDIS = trainingData[:,dataDISCol:dataDISCol+1]
columnRAD = trainingData[:,dataRADCol:dataRADCol+1]
columnTAX = trainingData[:,dataTAXCol:dataTAXCol+1]
columnPTRATIO = trainingData[:,dataPTRATIOCol:dataPTRATIOCol+1]
columnB = trainingData[:,dataBCol:dataBCol+1]
columnLSTAT = trainingData[:,dataLSTATCol:dataLSTATCol+1]
columnMEDV = trainingData[:,dataMEDVCol:dataMEDVCol+1]

columnTestCRIM = testingData[:,dataCRIMCol:dataCRIMCol+1]
columnTestZN = testingData[:,dataZNCol:dataZNCol+1]
columnTestINDUS = testingData[:,dataINDUSCol:dataINDUSCol+1]
columnTestCHAS = testingData[:,dataCHASCol:dataCHASCol+1]
columnTestNOX = testingData[:,dataNOXCol:dataNOXCol+1]
columnTestRM = testingData[:,dataRMCol:dataRMCol+1]
columnTestAGE = testingData[:,dataAGECol:dataAGECol+1]
columnTestDIS = testingData[:,dataDISCol:dataDISCol+1]
columnTestRAD = testingData[:,dataRADCol:dataRADCol+1]
columnTestTAX = testingData[:,dataTAXCol:dataTAXCol+1]
columnTestPTRATIO = testingData[:,dataPTRATIOCol:dataPTRATIOCol+1]
columnTestB = testingData[:,dataBCol:dataBCol+1]
columnTestLSTAT = testingData[:,dataLSTATCol:dataLSTATCol+1]
columnTestMEDV = testingData[:,dataMEDVCol:dataMEDVCol+1]

printstuff = False

def reg_lin_reg(theta, data_train_with_bias, labels_train, L):
    # Some renaming
    X = data_train_with_bias
    y = labels_train

    # Actual calculation
    y_pred = dot(X, theta)
    MSE = sum((y_pred-y)**2)/len(y)
    res = MSE + L*sum(theta)

    return res

def reg_lin_reg_deriv(theta, data_train_with_bias, labels_train, L):
    # Some renaming
    X = data_train_with_bias
    Xt = X.transpose()
    y = labels_train
    N = len(y)

    # Actual calculation
    y_pred = dot(X, theta)
    m = 2.0/N
    part1 = m*dot(Xt, y_pred-y)
    regularizer = 2*L*theta
    res = part1+regularizer

    return res

Ones = np.ones((len(columnMEDV),1))

Xtrain = np.hstack((Ones, columnCRIM.reshape(len(Ones),1)))
Xtrain = np.hstack((Xtrain, columnZN.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnINDUS.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnCHAS.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnNOX.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnRM.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnAGE.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnDIS.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnRAD.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnTAX.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnPTRATIO.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnB.reshape(len(Xtrain),1)))
Xtrain = np.hstack((Xtrain, columnLSTAT.reshape(len(Xtrain),1)))

Xtest = np.hstack((Ones, columnTestCRIM.reshape(len(Ones),1)))
Xtest = np.hstack((Xtest, columnTestZN.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestINDUS.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestCHAS.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestNOX.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestRM.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestAGE.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestDIS.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestRAD.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestTAX.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestPTRATIO.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestB.reshape(len(Xtest),1)))
Xtest = np.hstack((Xtest, columnTestLSTAT.reshape(len(Xtest),1)))

y = trainingData[:,-1]
print y.shape
print 'Validating f and f_prime...'
theta = np.zeros(14)
res = reg_lin_reg(theta, Xtrain, y, L=1)
print res
res = reg_lin_reg_deriv(theta, Xtrain, y, L=1)
print res

theta = np.ones(14)
theta.fill(0.01)
res = reg_lin_reg(theta, Xtrain, y, L=1)
print res
res = reg_lin_reg_deriv(theta, Xtrain, y, L=1)
print res

theta = np.zeros(14)
x,f,d = fmin_l_bfgs_b(reg_lin_reg,theta,reg_lin_reg_deriv,args=(Xtrain,y,1.0))

print 'Using theta given by fmin_l_bfgs_b:'
theta = x
X = Xtrain
y = trainingData[:,-1]

MSE = sum((y-dot(X, theta))**2) / len(y)
print 'MSE (training data) =', MSE
y_mean = sum(y)/len(y)
y_var = sum((y-y_mean)**2)/len(y)
FVUTest = MSE/y_var
print 'R2 (training data)=', 1 - FVUTest

X = Xtest
y = testingData[:,-2]

MSE = sum((y-dot(X, theta))**2) / len(y)
print 'MSE (testing data) =', MSE
y_mean = sum(y)/len(y)
y_var = sum((y-y_mean)**2)/len(y)
FVUTest = MSE/y_var
print 'R2 (testing data)=', 1 - FVUTest

N = len(loadedData)
k = 3 # valid k are 1, 2, 11, 22, 23, 46, 253, 506
print '#####'
folds = []
for i in range(0,k):
    folds.append(loadedData[i*(N/k):(i+1)*(N/k)])

lambdas = [10**i for i in range(-6,7)]
MSE_mins = []

for L in lambdas:
    MSE_sum = 0
    for i in range(0,k):
        theta = np.zeros(6)
        LSTAT = np.empty((0,1))
        y = np.empty((0,1))

        for j in range(0,k):
            if i != j:
                data = folds[j][:,dataLSTATCol:dataLSTATCol+1]
                LSTAT = np.vstack((LSTAT,data))

                data = folds[j][:,dataMEDVCol:dataMEDVCol+1]
                y = np.vstack((y,data))

        LSTAT2 = LSTAT**2
        LSTAT3 = LSTAT**3
        LSTAT4 = LSTAT**4
        LSTAT5 = LSTAT**5
        Ones = np.ones((len(LSTAT),1))
        X = np.hstack((Ones, LSTAT.reshape(len(Ones),1)))
        X = np.hstack((X, LSTAT2.reshape(len(X),1)))
        X = np.hstack((X, LSTAT3.reshape(len(X),1)))
        X = np.hstack((X, LSTAT4.reshape(len(X),1)))
        X = np.hstack((X, LSTAT5.reshape(len(X),1)))

        x,f,d = fmin_l_bfgs_b(reg_lin_reg,theta,reg_lin_reg_deriv,args=(X,y[:,-1],L))

        theta = x
        LSTAT = folds[i][:,dataLSTATCol:dataLSTATCol+1]
        LSTAT2 = LSTAT**2
        LSTAT3 = LSTAT**3
        LSTAT4 = LSTAT**4
        LSTAT5 = LSTAT**5
        Ones = np.ones((len(LSTAT),1))
        X = np.hstack((Ones, LSTAT.reshape(len(Ones),1)))
        X = np.hstack((X, LSTAT2.reshape(len(X),1)))
        X = np.hstack((X, LSTAT3.reshape(len(X),1)))
        X = np.hstack((X, LSTAT4.reshape(len(X),1)))
        X = np.hstack((X, LSTAT5.reshape(len(X),1)))
        y = folds[i][:,-1]

        MSE = sum((y-dot(X, theta))**2) / len(y)
        y_mean = sum(y)/len(y)
        y_var = sum((y-y_mean)**2)/len(y)
        FVUTest = MSE/y_var
        R2 = 1 - FVUTest
        MSE_sum = MSE_sum+MSE
        #print L, MSE

    MSE_mins.append(MSE_sum/(k-1))

pylab.plot(lambdas, MSE_mins, '-b')
pylab.xscale('log')
pylab.title('Lambda vs. MSE')
pylab.legend(['MSE(Lambda)'])
pylab.draw()

idx = np.argmin(MSE_mins)
best_lambda = lambdas[idx]

print 'Validating min lambda'
LSTAT = trainingData[:,dataLSTATCol:dataLSTATCol+1]
LSTAT2 = LSTAT**2
LSTAT3 = LSTAT**3
LSTAT4 = LSTAT**4
LSTAT5 = LSTAT**5
Ones = np.ones((len(LSTAT),1))
X = np.hstack((Ones, LSTAT.reshape(len(Ones),1)))
X = np.hstack((X, LSTAT2.reshape(len(X),1)))
X = np.hstack((X, LSTAT3.reshape(len(X),1)))
X = np.hstack((X, LSTAT4.reshape(len(X),1)))
X = np.hstack((X, LSTAT5.reshape(len(X),1)))

y = trainingData[:,-1]

theta = np.zeros(6)
x,f,d = fmin_l_bfgs_b(reg_lin_reg,theta,reg_lin_reg_deriv,args=(X,y,best_lambda))

theta = x

LSTAT = testingData[:,dataLSTATCol:dataLSTATCol+1]
LSTAT2 = LSTAT**2
LSTAT3 = LSTAT**3
LSTAT4 = LSTAT**4
LSTAT5 = LSTAT**5
Ones = np.ones((len(LSTAT),1))
X = np.hstack((Ones, LSTAT.reshape(len(Ones),1)))
X = np.hstack((X, LSTAT2.reshape(len(X),1)))
X = np.hstack((X, LSTAT3.reshape(len(X),1)))
X = np.hstack((X, LSTAT4.reshape(len(X),1)))
X = np.hstack((X, LSTAT5.reshape(len(X),1)))

y = testingData[:,-1]

MSE = sum((y-dot(X, theta))**2) / len(y)
y_mean = sum(y)/len(y)
y_var = sum((y-y_mean)**2)/len(y)
FVUTest = MSE/y_var
R2 = 1 - FVUTest

print MSE
print R2

pylab.show()
