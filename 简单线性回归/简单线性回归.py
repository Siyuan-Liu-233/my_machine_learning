import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class LinearRegression():  # 线性回归类

    def __init__(self):
        self.w = None

    def fit(self, X, y):  # 计算w
        #print(X.shape)
        X = np.insert(X, 0, 1, axis=1)  # 在第0列插入一个全为1的列 0 表示位置 1表示值 axis表示列
        #print(X.shape)
        #X_= np.linalg.inv(X.T.dot(X))  # 得到(X^T×X)^-1  还可以X_=np.mat(X.T.dot(X)) X_=X_.I
        X_=np.mat(X.T.dot(X)) 
        X_=np.array(X_.I)
        print(X_)
        self.w = X_.dot(X.T).dot(y)

    def predict(self, X):  # 预测y
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        #print('w is',self.w)
        return y_pred


def mean_squared_error(y_true, y_pred):
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def main():
    diabetes = datasets.load_diabetes()
    X=np.zeros((diabetes.data.shape[0],1))
    X[:,0]= diabetes.data[:, 2] # 方法2：X = diabetes.data[:,2,np.newaxis]   np.newaxis建立一个新的维度
    #print(X.shape)
    #print(X)
    x_train, x_test = X[:-20], X[-20:]  # 从第0个到倒数-20个为训练集，最后20个为测试集
    y_train, y_test = diabetes.target[:-20], diabetes.target[-20:]
    clf = LinearRegression()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print('错误方差：',mean_squared_error(y_test,y_pred))
    plt.scatter(x_test[:,0],y_test,c='black')
    plt.plot(x_test,y_pred,c='blue',linewidth=3)
    value=clf.predict([[0.345]])
    print('预测值：',value)
    plt.show()

main()


