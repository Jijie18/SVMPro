import numpy as np
import random
import sys

class SVM:
    def __init__(self,x,y,epochs=200,learning_rate=0.01):
        self.x=np.c_[np.ones((x.shape[0])),x]
        self.y=y
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.w=np.random.uniform(size=np.shape(self.x)[1],)

    def get_lost(self,x,y):
        loss=max(0,1-y*np.dot(x,self.w))
        return loss

    def cal_sgd(self,x,y,w):
        if y*np.dot(x,self.w)<1:
            w=w+self.learning_rate*y*x
        else:
            w=w
        return w

    def train(self):
        for epochs in range(self.epochs):
            randomize=np.arange(len(self.x))
            np.random.shuffle(randomize)
            x=self.x[randomize]
            y=self.y[randomize]
            loss=0
            for xi,yi in zip(x,y):
                loss+=self.get_lost(xi,yi)
                self.w=self.cal_sgd(xi,yi,self.w)
            #print("epochs:{0} loss:{1}".format(epochs,loss))


    def predict(self,x):
        x_test=np.c_[np.ones((x.shape[0])),x]
        return np.sign(np.dot(x_test,self.w))

if len(sys.argv) == 5:
    train_data=sys.argv[1]
    test_data=sys.argv[2]
    time_bu=sys.argv[4]


f=open(train_data,"r")

line=f.readline()
t=0
while line:
    t=t+1
    line=f.readline()
xx=np.zeros((t,10))
yy=np.zeros(t)

f.close()
f=open(train_data,"r")
for i in range(t):
    line=f.readline().split()
    for j in range(10):
        xx[i][j]=line[j]
    yy[i]=line[10]
svm=SVM(xx,yy)
svm.train()
f.close()

f=open(test_data,"r")
line=f.readline()
t=0
while line:
    t=t+1
    line=f.readline()
xx=np.zeros((t,10))


f.close()
f=open(test_data,"r")
for i in range(t):
    line=f.readline().split()
    for j in range(10):
        xx[i][j]=line[j]


y=svm.predict(xx)
for i in y:
    print(int(i))


