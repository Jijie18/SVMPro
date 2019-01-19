import numpy as np
import random

class SVM:
    def __init__(self,x,y,learning_rate=0.01):
        self.x=x
        self.y=y
        self.learning_rate=learning_rate
        self.w=np.random.uniform(size=len(x))

    def get_lost(self,x,y):
        loss=max(0,1-y*np.dot(x,self.w))
        return loss

    def cal_sgd(self,x,y):
        if y*np.dot(x,self.w)<1:
            self.w=self.w+self.learning_rate*y*x
        else:
            self.w=self.w
        #return self.w

f=open("C:/Users/16972/Desktop/AI/train_data.txt","r")
line=f.readline()
list=line.split()
x=np.zeros(10)
for i in range(10):
    x[i]=float(list[i])
y=float(list[-1])
svm=SVM(x,y)
print(svm.w)
svm.cal_sgd(x,y)
line=f.readline()
z=1
while z<=1200:
    list=line.split()
    x=np.zeros(10)
    for i in range(10):
        x[i]=float(list[i])
    y=float(list[-1])
    svm.cal_sgd(x,y)
    print(svm.w)
    line=f.readline()
    z=z+1
