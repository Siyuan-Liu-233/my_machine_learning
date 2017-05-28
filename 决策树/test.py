import numpy as np
from tree import *
import sklearn 

def storeTree(inputTree,filename):
    import pickle                   #以2进制存储字典型数据
    fw=open(filename)           
    pickle.dump(inputTree,fw)   #存储
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)  #读取字典


if __name__ == '__main__':
    fr=open('lenses.txt')
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    labels=['age','prescript','astigmatic','tearRate']
    lensesTree=createTree(lenses,labels)
    print(lensesTree)
