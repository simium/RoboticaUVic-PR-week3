import numpy as np
from scipy.linalg import lstsq
from sklearn import linear_model

# First define some handy shortcuts
dot = np.dot
inv = np.linalg.inv

loadedData = np.array([map(float, l.split()) for l in open('housing.data')])
trainingData = np.array([map(float, l.split()) for l in open('housing_training.data')])
testingData = np.array([map(float, l.split()) for l in open('housing_testing.data')])
y = loadedData[:,-1]
print len(y),'data points in housing.data'

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

print '\nSolving Q4:'

target = trainingData[:,dataMEDVCol:dataMEDVCol+1]
targetTest = testingData[:,dataMEDVCol:dataMEDVCol+1]
Ones = np.ones((len(target),1))

# I add all the columns one by one so it's easier to choose which one
# to add/remove
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

mTheta = lstsq(Xtrain, target)[0]
print mTheta.shape
print Xtrain.shape
target_pred = dot(Xtrain, mTheta)

msePred = sum((target-target_pred)**2)/len(target)
meanTarget = sum(target)/len(target)
varianceTarget = sum((target-meanTarget)**2)/len(target)
FVU = msePred/varianceTarget

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

mThetaTest = lstsq(Xtest, targetTest)[0]

# use theta from training set, not from testing set
target_pred_test = dot(Xtest, mTheta)

msePredTest = sum((targetTest-target_pred_test)**2)/len(targetTest)
meanTargetTest = sum(targetTest)/len(targetTest)
varianceTargetTest = sum((targetTest-meanTargetTest)**2)/len(targetTest)
FVUTest = msePredTest/varianceTargetTest

print 'MSE training set:', msePred
print 'MSE testing set:', msePredTest
print 'R2 of testing set against theta from training set:', 1 - FVUTest,'\n'
