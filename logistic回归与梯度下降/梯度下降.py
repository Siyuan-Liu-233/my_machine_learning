import numpy as np
import matplotlib.pyplot as plt
import random
def loadData():
    dataMat,labelMat=[],[]
    f=open('testSet.txt')
    for line in f.readlines():
        linenew=line.strip().split()
        dataMat.append([1.0,float(linenew[0]),float(linenew[1])])  #添加一组值1
        labelMat.append([int(linenew[2])])
    return dataMat,labelMat              #得到数据和类别

def sigmoid(inX):           
    return 1.0/(1+np.exp(-inX))


def gradescent(dataIn,dataLabel):           #整体梯度算法
    dataget=np.array(dataIn)             #转化为行列式
    dataLabelget=np.array(dataLabel)     #转化为行列式
    m,n=np.shape(dataget)
    #print(dataLabelget)
    alpha=0.01                          #步长
    maxCycles=6000                      #最大迭代次数
    weigh=np.ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataget.dot(weigh))       #所有样本对应的预测概率
        error=dataLabel-h                   #所有样本的误差值
        weigh=weigh+alpha*dataget.T.dot(error)      #计算每个特征的权重
    print(weigh)
    return weigh,n
def predict(inX,w):
    dataMat,labelMat=loadData()
    np.array(inX)
    inX=np.insert(inX,0,1)          #第0列 插入1
    inX=inX[np.newaxis,:]           #增加维度 变为1*n
    res=sigmoid(inX.dot(w))
    print('为1的概率为：',res)

def plotFit(w):          
    dataMat,labelMat=loadData()
    dataget=np.array(dataMat)
    n=np.shape(dataget)[0]
    labelget=np.array(labelMat)
    x1,x2,y1,y2=[],[],[],[]
    for i in range(n):
        #print(labelget[i,0])
        if int(labelget[i,0])==1:
            x1.append(dataget[i,1])
            y1.append(dataget[i,2])
        else:
            x2.append(dataget[i,1])
            y2.append(dataget[i,2])
    plt.subplot(1,1,1)
    plt.scatter(x1,y1,s=30,c='red',marker='s') #s表示点的大小
    plt.scatter(x2,y2,s=30,c='green')
    x=np.arange(-3,3,0.1)
    y=(-w[0]-w[1]*x)/w[2]           #sigmoid函数 wx=0时p为0.5 此线作为分界线
    plt.plot(x,y)

def tidu_suiji(dataM,LabelM):       #随机梯度 减少复杂度 仅用一个数据更新回归系数
    m,n=np.shape(dataM)
    #print(m,n)
    alpha=0.01
    w=np.ones(n)
    for i in range(m):
        h=sigmoid(np.sum(dataM[i]*w)) #计算某个样本对应的预测概率
        error=LabelM[i]-h      #计算误差
        #print(dataMat[i])
        w=w+alpha*error*dataM[i]
    #print(w)
    return w
def tidu_suiji_youhua(dataM,LabelM,numIter=550):       #随机梯度 减少复杂度 仅用一个数据更新回归系数
    m,n=np.shape(dataM)
    #print(m,n)
    w=np.ones(n)
    #print(dataIndex)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
                #不断更新步长，一开始步长大下降快，后面步长小 并且不是严格下降
            randindex=random.choice(dataIndex)
            #print(dataIndex,i,randindex)
            alpha=4/(i+1+j)+0.01
            h=sigmoid(np.sum(dataM[randindex]*w)) #计算某个样本对应的预测概率
            error=LabelM[randindex]-h      #计算误差
            
            w=w+alpha*error*dataM[randindex] #更新参数
            dataIndex.remove(randindex) #删除已使用的索引
        dataIndex=list(range(m))
    #print(w)
    return w





if __name__ == '__main__':
    
    dataMat,labelMat=loadData()
    w,n=gradescent(dataMat,labelMat)
    plotFit(w)
    w_suiji=tidu_suiji(dataMat,labelMat)
    plotFit(w_suiji)
    w_suiji_youhua=tidu_suiji_youhua(dataMat,labelMat)
    plotFit(w_suiji_youhua)
    predict([1.17,3.17],w_suiji_youhua)
    plt.show()