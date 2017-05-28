import os 
import numpy as np
from my_knn import *
def image_to_vector(filename):               #二维图像转化为1*1024的矩阵
    returnVector=np.zeros((1,1024))
    f=open(filename)
    line=f.readlines()                      #每行变为一个list的元素 存入list中
    for i in range(32):
        line.remove('\n')
        for j in range(32):
            returnVector[0,32*i+j]=int(line[i][j])  #第32*i+j个数 为第I行第J列的数
    return returnVector


def classify_write():
    labels=[]
    trainingfile=os.listdir('trainingDigits')   #读取文件名
    m=len(trainingfile)                         #读取文件个数                        
    trainMat=np.zeros((m,1024))                 #预处理
    for i in range(m):
        num=trainingfile[i][0]
        labels.append(int(num))                 #由文件名得到每个图所属类别
        trainMat[i,:]=image_to_vector('trainingDigits/%s'%(trainingfile[i])) 
    testFile=os.listdir('testDigits')
    error=0.0
    mtest=len(testFile)
    for i in range(mtest):
        num=testFile[i][0]
        testMat=image_to_vector('testDigits/%s'%(testFile[i]))
        result=classify0(testMat,trainMat,labels,5)   
        print("预测结果为：%d,实际结果为：%d"%(result,int(num)))
        if result!=int(num):
            error+=1
    print('总错误个数为：%d'%error)
    print('错误率为：%f'%(error/mtest))




if __name__ == '__main__':
    classify_write()

