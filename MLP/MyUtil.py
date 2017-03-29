'''
Created on Mar 17, 2017

@author: Owner
'''
import numpy as np
import matplotlib.pyplot as plt

def LogtisticFunction(x):
    '''
    x: a float
    '''
    return 1/(1+np.exp(-1*x))

def DerivationLogtisticFunction(x):
    return LogtisticFunction(x)*(1-LogtisticFunction(x))


def SquareError(x,y):
    '''
    x: a vector
    y: a vector
    '''
    if len(x)!=len(y):
        print 'Length different!'
        return 0.0
    
    err = np.subtract(x,y)
    totalerror = 0.0
    for anitem in err:
        totalerror += np.square(anitem)
    return totalerror/2
    

if __name__=='__main__':
    x = [-10,-5,-3,-2,-1,-0.5,0,0.5,1,2,3,5,10]
    a =[]
    for anitem in x:
        b = LogtisticFunction(anitem)
        a.append(b)
    plt.plot(x,a)
    plt.show()