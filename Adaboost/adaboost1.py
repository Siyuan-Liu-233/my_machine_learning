import numpy as np
def loadSimData():
    datMat=np.matrix([[1.0,2.1],[2.0,1.1],[1.3,1.0],[1.0,1.0],[2.0,1.0]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

def stumpClassfy(dataMatrix,dimen,threshVal,threshIneq):
    retArray=np.ones((np.shape(dataMatrix)[0],1))
    #把左边赋值为-1或者右边赋值为-1
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0   
    return retArray


def buildStump(dataArr,classLabels,D):     #D表示权重
    dataMatrix=np.mat(dataArr)
    labelMat=np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps=10.0               #步数
    bestStump={}
    bestClasEst=np.mat(np.zeros((m,1)))
    minError=np.inf             #np.inf 表示无穷大
    for i in range(n):          #第i个特征
        rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps #步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']: 
                    threshVal=(rangeMin+float(j)*stepSize)
                    predictedVals=stumpClassfy(dataMatrix,i,threshVal,inequal)
                    errArr=np.mat(np.ones((m,1)))
                    errArr[predictedVals==labelMat]=0   #预测和实际相等的地方赋值为0
                    weightedError=D.T*errArr            #计算分类误差率
                    if weightedError<minError:          #得到最小的分类误差率
                        minError=weightedError
                        bestClasEst=predictedVals.copy()
                        bestStump['dim']=i              #选取的特征编号
                        bestStump['stresh']=threshVal   #选取的划分阈值
                        bestStump['ineq']=inequal       #记录阈值哪边取-1
    return bestStump,minError,bestClasEst  

def adaBoostTrainDs(dataArr,classLabels,numIt=40):
    werkClassArr=[]  #存放弱分类器信息
    m=np.shape(dataArr)[0]
    D=np.mat(np.ones((m,1))/m)
    aggClassEst=np.mat(np.zeros((m,1)))    #用于记录每次迭代后样本的值
    for i in range(numIt):
        bestStump,minError,bestClasEst=buildStump(dataMat,classLabels,D)
        alpha=float(0.5*np.log((1-minError)/np.max(minError,1e-16)))
        print('D:',D.T)
        bestStump['alpha']=alpha  #alpha存入弱分类器信息
        werkClassArr.append(bestStump)
        print('bestClasEst:',bestClasEst.T)
        expon=np.multiply(-1*alpha*np.mat(classLabels).T,bestClasEst)#np.multiply表示对应相乘 类似数组的*
        D=np.multiply(D,np.exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*bestClasEst #记录累计值
        print("arrClassEst:",aggClassEst.T)
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))#得到错误的个数
        errorRate=aggErrors.sum()/m         #计算总的错误率
        print('total error:',errorRate)
        if errorRate==0: 
            print('最终分类结果:\n',np.sign(aggClassEst))
            break
    return werkClassArr







if __name__ == '__main__':
    # D=np.mat(np.ones((5,1))/5)
    dataMat,classLabels=loadSimData()
    # bestStump,minError,bestClasEst=buildStump(dataMat,classLabels,D)
    # print(bestStump,'\n最小误差率:',minError,'\n分类结果:\n',bestClasEst)
    classifierArray=adaBoostTrainDs(dataMat,classLabels)
    #print(classifierArray)




