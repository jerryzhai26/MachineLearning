import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator
from os import listdir

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')         #为matplotlib引入中文包

def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = np.zeros((numberOfLines,3))             #zeros 生成矩阵的维度由zeros函数第一个参数决定，可以为标量或向量, 第二个参数用例规定矩阵元素的数据类型
	# print(returnMat)
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

##############################################################数据归一化############################################################
def autoNorm(dataSet):
	minVals = dataSet.min(0)              #0代表按列操作，1代表按行操作
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = np.zeros(np.shape(dataSet))      #shape获取矩阵的维数
	m = dataSet.shape[0]                      #shape作为矩阵的函数，返回维数后取第一维的值
	normDataSet = dataSet - np.tile(minVals, (m, 1))   #tile重复数组
	normDataSet = normDataSet/np.tile(ranges, (m,1))
	return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
	dataSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistance = sqDiffMat.sum(1)
	distances = sqDistance ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0)
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def datingClassTest():
	hoRadio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRadio)
	errorCount = 0
	print(numTestVecs)
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m], datingLabels[numTestVecs:m], 5)
		print('the classifier came back with ', classifierResult, ' the real answer is ', datingLabels[i])
		if classifierResult != datingLabels[i]:
			errorCount += 1
	print('the total error rate is %f' %(errorCount/float(numTestVecs)))


def img2vector(filename):
	returnVect = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		linestr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(linestr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = np.zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		if(classifierResult != classNumStr):
			errorCount += 1
	print("the total number of errors is %d" %errorCount)
	print("the total error rate is %f" %(errorCount/float(mTest)))

# datingClassTest()
handwritingClassTest()
# dataMat, labelVector = file2matrix('datingTestSet2.txt')
# # print(dataMat)
# # print(labelVector[0:20])

# ##############################################################数据可视化############################################################

# labelVector = np.array(labelVector)
# fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.scatter(dataMat[:,1], dataMat[:,2], 15.0*np.array(labelVector), 15.0*np.array(labelVector))
# # ax.scatter(dataMat[:,1], dataMat[:,2], c = 15.0*np.array(labelVector))
# idx_1 = np.where(labelVector == 1)
# idx_2 = np.where(labelVector == 2)
# idx_3 = np.where(labelVector == 3)
# print(labelVector[idx_2])
# p1 = plt.scatter(dataMat[idx_1,1], dataMat[idx_1,2], 15*np.array(labelVector[idx_1]), c = 'm', label = '不喜欢')
# plt.scatter(dataMat[idx_2,1], dataMat[idx_2,2], 15*np.array(labelVector[idx_2]), c = 'c', label = '魅力一般')
# plt.scatter(dataMat[idx_3,1], dataMat[idx_3,2], 15*np.array(labelVector[idx_3]), c = 'r', label = '极具魅力')
# plt.legend(loc = 'upper left')
# plt.legend(prop=zhfont1)
# plt.show()

# normMat, ranges, minVals = autoNorm(dataMat)
# print(normMat)
# print(ranges)
# print(minVals)