import numpy as np
import pandas as pd
import random  

def creatDataSet():
    group = np.array([[1.2, 0.9], [1.0, 1.1], [0.3, 0.4],
                      [1.0, 1.0], [0, 0], [0.1, 0.1]])
    labels = ['A', 'A', 'B', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataset, labels, k):  # inX表示输入点
    dataSetSize = dataset.shape[0]  # shape[0]表示多少行 此处为4
    # np.tile表示样本复制成(datasetsize,1)那么大
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    sortedDistIndicies = distance.argsort()  # argsort函数返回的是数组值从小到大的索引值
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        # dict.get()  键voteLable存在则返回对应键值否则返回0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    # dict.items()把键和键值变成一组元组，然后以列表方式存储，key=lambda x:x[1]表示按键值排序

    # print(sortedClassCount)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOflines = fr.readlines()
    for i in arrayOflines:
        if i==('\n'):
            arrayOflines.remove('\n')
    random.shuffle(arrayOflines)    #随机排列数据
    numOfLines = len(arrayOflines)
    returnMat = np.zeros((numOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOflines:
        line = line.strip()  # 去除换行符
        listFromline = line.split('\t')
        if listFromline==['']:
            continue
        returnMat[index, :] = listFromline[0:3]
        if listFromline[-1]=='didntLike':
            classLabelVector.append(0)
        elif listFromline[-1]=='smallDoses':
            classLabelVector.append(1)
        elif listFromline[-1]=='largeDoses':
            classLabelVector.append(2)
        index += 1
   
    #print(classLabelVector)
    #print(len(returnMat),len(classLabelVector))
    #print(returnMat)
    return returnMat, classLabelVector


def autoNorm(dataset):  # 归一化
    minVals = dataset.min(axis=0) #返回一个行向量 元素为每列最小值
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normDataset = dataset - np.tile(minVals, (m, 1))      
    normDataset = normDataset / np.tile(ranges, (m, 1))
    print(dataset)
    return normDataset, ranges, minVals


def datingclasstest():  # 测试准确度
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 9)
        #print('分类结果(预测值，真实值)，(%d,%d)' % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print('错误率:%f' % (errorCount / float(numTestVecs)))
    
def classifyperson(input_man):
    resultList=['不喜欢','有点喜欢','很喜欢']
    Mat,labels=file2matrix('datingTestSet.txt')
    normDataset, ranges, minVals=autoNorm(Mat)
    result_pro=classify0((input_man-minVals)/ranges,normDataset,labels,9)
    print('她可能对你%s'%(resultList[result_pro]))



if __name__ == '__main__':
    # group,labels=creatDataSet()
    # test=classify0([0.8,0.7],group,labels,3)
    # print(test)
    #datingclasstest()
    classifyperson([50000,23,11])
