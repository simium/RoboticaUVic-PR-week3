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

print '\nSolving Q5:'

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

def solveQ5(trainData, testData, columnLabel):
    target = trainingData[:,dataMEDVCol:dataMEDVCol+1]
    targetTest = testingData[:,dataMEDVCol:dataMEDVCol+1]
    Ones = np.ones((len(target),1))

    # Fitting the parameters: theta = (X'*X)^-1*X'*y
    Xtrain = np.hstack((Ones, trainData.reshape(len(Ones),1)))

    #firstCol = columnRM**2
    #secondCol = columnLSTAT**2
    #thirdCol = columnB**2
    #fourthCol = columnZN**2
    firstCol = trainData
    secondCol = trainData**2
    thirdCol = trainData**3
    fourthCol = trainData**4
    Xtrain = np.hstack((Xtrain, firstCol.reshape(len(Xtrain),1)))
    Xtrain = np.hstack((Xtrain, secondCol.reshape(len(Xtrain),1)))
    Xtrain = np.hstack((Xtrain, thirdCol.reshape(len(Xtrain),1)))
    Xtrain = np.hstack((Xtrain, fourthCol.reshape(len(Xtrain),1)))

    mTheta = lstsq(Xtrain, target)[0]
    target_pred = dot(Xtrain, mTheta)

    msePred = sum((target-target_pred)**2)/len(target)
    meanTarget = sum(target)/len(target)
    varianceTarget = sum((target-meanTarget)**2)/len(target)
    FVU = msePred/varianceTarget

    Xtest = np.hstack((Ones, testData.reshape(len(Ones),1)))

    #firstCol = columnTestRM**2
    #secondCol = columnTestLSTAT**2
    #thirdCol = columnTestB**2
    #fourthCol = columnTestZN**2
    firstCol = testData
    secondCol = testData**2
    thirdCol = testData**3
    fourthCol = testData**4
    Xtest = np.hstack((Xtest, firstCol.reshape(len(Xtest),1)))
    Xtest = np.hstack((Xtest, secondCol.reshape(len(Xtest),1)))
    Xtest = np.hstack((Xtest, thirdCol.reshape(len(Xtest),1)))
    Xtest = np.hstack((Xtest, fourthCol.reshape(len(Xtest),1)))

    mThetaTest = lstsq(Xtest, targetTest)[0]
    target_pred_test = dot(Xtest, mTheta)

    msePredTest = sum((targetTest-target_pred_test)**2)/len(targetTest)
    meanTargetTest = sum(targetTest)/len(targetTest)
    varianceTargetTest = sum((targetTest-meanTargetTest)**2)/len(targetTest)
    FVUTest = msePredTest/varianceTargetTest

    print '###',columnLabel,'###'
    print 'MSE training set:', msePred
    print 'MSE testing set:', msePredTest
    print 'R2 of testing set against theta from training set:', 1 - FVUTest,'\n'


solveQ5(columnCRIM, columnTestCRIM, "CRIM")
solveQ5(columnZN, columnTestZN, "ZN")
solveQ5(columnINDUS, columnTestINDUS, "INDUS")
solveQ5(columnCHAS, columnTestCHAS, "CHAS")
solveQ5(columnNOX, columnTestNOX, "NOX")
solveQ5(columnRM, columnTestRM, "RM")
solveQ5(columnAGE, columnTestAGE, "AGE")
solveQ5(columnDIS, columnTestDIS, "DIS")
solveQ5(columnRAD, columnTestRAD, "RAD")
solveQ5(columnTAX, columnTestTAX, "TAX")
solveQ5(columnPTRATIO, columnTestPTRATIO, "PTRATIO")
solveQ5(columnB, columnTestB, "B")
solveQ5(columnLSTAT, columnTestLSTAT, "LSTAT")
