import pandas
import matplotlib.pyplot as plt
import numpy as np
pga=pandas.read_csv('pga.csv')

pga['开球距离']=(pga['开球距离']-pga['开球距离'].mean())/pga['开球距离'].std()
#print(pga['开球距离'])
pga.精确度=(pga.精确度-pga.精确度.mean())/pga.精确度.std()

plt.subplot(1,2,1)
plt.scatter(pga.开球距离,pga.精确度)
plt.xlabel('开球距离')
plt.ylabel('精确度')


def cost(w0,w1,x,y):
    J=0
    m=len(x)
    for i in range(m):
        h=w1*x[i]+w0
        J+=(h-y[i])**2
    J/=(2*m)
    return J

print(cost(0,1,pga.开球距离,pga.精确度))
w0=100
www=np.linspace(-3,2,100)
costs=[]
for ww in www:
    costs.append(cost(w0,ww,pga.开球距离,pga.精确度))
plt.subplot(1,2,2)
plt.plot(www,costs)
print(costs)
plt.show()

