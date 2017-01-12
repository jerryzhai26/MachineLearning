from math import log
import operator
from treePlotter import *
def calcShannonEnt(dataSet):                 #计算熵
	numEntries = len(dataSet)
	labelCount = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCount.keys():
			labelCount[currentLabel] = 0
		labelCount[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCount:
		prob = labelCount[key]/numEntries
		shannonEnt -= prob*log(prob, 2)
	return shannonEnt

def createDataSet():
	dataSet = [[1,1,'yes'],[1, 1, 'yes'],[1,0,'no'], [0, 1, 'no'], [0,1,'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

myDat, labels = createDataSet()
# print(myDat)
# print(calcShannonEnt(myDat))
def splitDataSet(dataSet, axis, value):       #挑出所有满足条件的数据集，axis为下标，value为目标值
	retDataSet = []
	for  featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])                     #extend 和append的区别
			retDataSet.append(reducedFeatVec)	
	return retDataSet

# print(splitDataSet(myDat, 0, 1))
def chooseBestFeatureToSplit(dataSet): 
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	baseInfoGain = 0.0
	bestFeature = 0
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if(infoGain > baseInfoGain):
			baseInfoGain = infoGain
			bestFeature = i
	return bestFeature

# print(chooseBestFeatureToSplit(myDat))

def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount:
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel: {}}
	del(labels[bestFeat])
	featValue = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValue)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree

# print(createTree(myDat, labels))

def classify(inputTree, featLabels, testVec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

# myTree = retrieveTree(0)
# print(classify(myTree, labels, [1,0]))
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print(lensesTree)
createPlot(lensesTree)