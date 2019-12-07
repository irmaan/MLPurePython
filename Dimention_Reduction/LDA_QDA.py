import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pylab as plt


def readData():
    trainData = pd.read_csv('trainset.csv')
    testData = pd.read_csv('testset.csv')
    return trainData.iloc[: , 1:] , testData.iloc[: , 1:]

def calculateTrainProperties():
    covarianceMatrix = []
    numOfDataForClass = []
    allData = []
    mo = [] # mean
    probabilities = []

    for i in range(1,12):

        probabilities.append(trainDataset[trainDataset.y == i].shape[0] * 1.0 / trainDataset.shape[0])

        numOfDataForClass.append(trainDataset[trainDataset.y == i].shape[0])

        mo.append(trainDataset[trainDataset.y == i].iloc[:, 1:].mean())

        allData.append(trainDataset[trainDataset.y == i])

        covarianceMatrix.append(np.cov(trainDataset[trainDataset.y == i].iloc[: , 1:].T))

    return allData, mo,probabilities,covarianceMatrix ,numOfDataForClass


def runLDA():
    covarianceMatrix = np.cov(trainDataset.iloc[:, 1:].T)
    inverseOfCovariance = np.linalg.inv(covarianceMatrix)
    classLabels = testDataset.iloc[: , 0]
    testLDA = testDataset.iloc[: , 1:]

    numCorrect = 0
    numIncorrect = 0

    for i in range(testDataset.shape[0]):
        discriminantFunc = []
        for j in range(11):
            X = testLDA.iloc[i]
            term1 = np.dot(mo[j] , inverseOfCovariance)
            term1 = np.dot(term1 , X)
            term2 = 0.5 * np.dot(mo[j] , inverseOfCovariance)
            term2 = np.dot(term2 , mo[j])
            term3 = np.log(probability[j])
            discriminantFunc.append(term1 - term2 + term3)
        tempLabels = discriminantFunc.index(max(discriminantFunc)) + 1

        if tempLabels == classLabels[i]:
            numCorrect += 1
        else:
            numIncorrect += 1
    return (numCorrect  / (numCorrect + numIncorrect))


def runQDA():
    classLabels = testDataset.iloc[:, 0]
    LDA_test = testDataset.iloc[:, 1:]

    numCorrect = 0
    numIncorrect = 0
    for i in range(testDataset.shape[0]):
        discriminantFunc = []
        for k in range(11):
            coavarianceMatrix = covarianceMatrix[k]
            InversCovariance = np.linalg.inv(coavarianceMatrix)
            covarianceDet = np.linalg.det(coavarianceMatrix)
            X = LDA_test.iloc[i]
            term1 = - 0.5 * np.log(covarianceDet)
            term2 = 0.5 * np.dot((X - mo[k]) , InversCovariance)
            term2 = np.dot(term2 , (X - mo[k]))
            term3 = np.log(probability[k])
            discriminantFunc.append(term1 - term2 + term3)
        tempLabel = discriminantFunc.index(max(discriminantFunc)) + 1

        if tempLabel == classLabels[i]:
            numCorrect += 1
        else:
            numIncorrect += 1
    return  (numCorrect  / (numCorrect + numIncorrect))


def runKnn():
    knn = KNN(n_neighbors = 5)
    knn.fit(trainDataset.iloc[:, 1:], trainDataset.iloc[:, 0])
    predictions = knn.predict(testDataset.iloc[:, 1:])
    return accuracy_score(testDataset.iloc[:, 0], predictions)



trainDataset , testDataset= readData()
allData,mo,probability,covarianceMatrix,numDataForClass = calculateTrainProperties()
accLDA=runLDA()
print('LDA Accuracy :',accLDA)
accQDA=runQDA()
print('QDA Accuracy :',accQDA)
accKNN= runKnn()
print('KNN Accuracy :',accKNN)

