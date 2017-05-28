from math import log
def calcShannonEnt(dataset):        #计算商值
    numEntries=len(dataset)
    labelCount={}
    for featVec in dataset:
        currentLabel=featVec[-1]        #当前类别
        # if currentLabel not in labelCount.keys():
        #     labelCount[currentLabel]=0
        # labelCount[currentLabel]+=1
        labelCount[currentLabel]=labelCount.get(currentLabel,0)+1  #统计每个类别数量
    shannoEnt=0
    for key in labelCount:
        prob=float(labelCount[key])/numEntries  #计算每个类别的概率  
        shannoEnt-=prob*log(prob,2)             #计算商值
    return shannoEnt
def createDataSet():
    dataset=[[1,1,'yes'],[1,0,'yes'],[0,1,'no'],[0,1,'no'],[0,1,'no']]
    labels={'no','yes'}
    return dataset,labels

def splitDataSet(dataset,axis,value):   #经过一个特征之后的数据
    retDataSet=[]
    for featVec in dataset:
        if featVec[axis]==value:    #某特征等于某值时 将其进行划分
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])#拼接list 
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataset):
    numFeatures=len(dataset[0])-1   #特征个数
    baseEntropy=calcShannonEnt(dataset)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataset] #取所有数据的第i个特征
        uniqueVals=set(featList)    #去除特征中重复的值 
        newEntropy=0.0          #新商值
        for value in uniqueVals:        #value
            subDataSet=splitDataSet(dataset,i,value)
            prob=len(subDataSet)/float(len(dataset))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
        return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        classCount[vote]=classCount.get(vote,0)+1
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    return sortedClassCount[0][0]   #返回个数最多的类别


def createTree(dataset,labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0])==len(classList):   #如果都为一个类别
        return classList[0]
    if len(dataset[0])==1:                             #如果只剩下一个特征
        return majorityCnt(classList)               #选择元素个数最多的那个类别
    bestFeat=chooseBestFeatureToSplit(dataset)  #得到最好的划分特征i
    bestFeatLabel = labels[bestFeat]            #这个最好特征的名字
    myTree={bestFeatLabel:{}}                   #构造树
    del(labels[bestFeat])                       #删除这个特征的名字
    featValues = [example[bestFeat] for example in dataset]     #这个特征的可选值
    uniqueVals = set(featValues)                    #删除重复的可选值
    for value in uniqueVals:            
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value),subLabels)  #递归
    return myTree          



if __name__ == '__main__':
    dataset,labels=createDataSet()
    print(calcShannonEnt(dataset))