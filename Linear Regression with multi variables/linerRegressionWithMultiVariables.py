import numpy as np
#import numpy.linalg as inv
import pandas as pd
import matplotlib.pyplot as plt
def normalization(x):
    max=np.amax(x)
    min=np.amin(x)
    mean=np.mean(x)
    print(x[0,0])
    for i in range(len(x)):
        x[i,0]=((x[i,0]-mean)/(max-min))
    return x
##############################################
def costError(X,Y,theta): 
    return (1/(2*len(X)))*np.sum(np.power(((X*theta)-Y),2))

##############################################
def gridentDecent(X,Y,theta,alpha,iters):
    m=X.shape[1]
   
    temp=np.zeros((m,1))
    for i in range(iters):
        hx=X*theta
        for j in range(m):
            dot=np.multiply((hx-Y),X[:,j])
            temp[j,0]=theta[j,0]-((alpha/len(X))*np.sum(dot))
        theta=temp
        
    return theta
#################################################################### 
def normalEquation(X,Y):
    theta=np.linalg.inv(X.T*X)*X.T*Y
    return theta
###################################################################   
path="Admission_Predict.csv"
datarp=pd.read_csv(path,header=None)
print(datarp.head(10))
#datarp=(datarp-datarp.mean())/datarp.std()
#datarp.insert(0,'Ones',1)
#cols=datarp.shape[1]
#X=np.matrix(datarp.iloc[:,:cols-1])
#Y=np.matrix(datarp.iloc[:,cols-1:cols])
#theta=np.zeros((X.shape[1],1))
#theta=gridentDecent(X,Y,theta,0.1,1000)
#draw=plt.scatter(datarp["firstInput"],datarp["result"])
#draw=plt.scatter(datarp["secondInput"],datarp["result"])
#hx=X*theta
#theta=normalEquation(X,Y)
##f=theta[0,0]+X[:,1]*theta[1,0]
##plt.plot(X[:,1],f,"r")
#print(costError(X,Y,theta))