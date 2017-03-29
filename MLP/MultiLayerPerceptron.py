'''
Created on Mar 17, 2017

@author: Owner
'''
import numpy as np
import MyUtil
import pickle

class MLP:
    m_LayerDimesions = []
    m_ActiveFunctions = []
    m_LearningRate = 0.01
    m_EchoNumber = 100
    m_ModelPath = ''
    
    def __init__(self,LayerDimesions,ActiveFunctions,LearningRate,ModelPath):
        '''
        LayerDimesions: an array, 
        LayerDimesions[0]: inputDim
        LayerDimension[-1]: outputDim
        In the middle, the dimensions of hidden layers
        ActiveFunctions: log
        LearningRate: current every layer has same learning rate
        '''
        self.m_Weights = []
        if len(LayerDimesions)<=1:
            print 'You must have input and output'
            return
        if LayerDimesions[0]<=0 or LayerDimesions[-1]<=0:
            print 'Input and output dimension must be 1 or above'
            return
        
        if len(LayerDimesions)!=len(ActiveFunctions)+1:
            print 'Every hidden layer should have a active function. '
            return
        
        self.m_ModelPath = ModelPath
        self.m_LearningRate = LearningRate
        self.m_LayerDimesions = LayerDimesions
        self.m_ActiveFunctions = ActiveFunctions
        for i in range(len(self.m_LayerDimesions)-1):
            aweight = np.random.rand(self.m_LayerDimesions[i+1],self.m_LayerDimesions[i])
            b = np.random.rand(1)
            self.m_Weights.append([aweight,b])
    
    def Train(self,X,Y):
        '''
        X: input, 2d array, each one is a sample [[1,2,3],[-1,-2,-3]]
        Y: label, each one is an output
        '''
           
    

    def OneLayerForwardForOneSample(self,X,W,b):
        '''
        X: a sample
        W: weight matrix
        b: bias
        '''
        if len(X)==0:
            return []
        InputDim = len(X)
        
        if len(W)==0:
            return [] 
        OutputDim = len(W)
        
        
        output = []
        for j in range(OutputDim):
            output.append(np.dot(X,W[j])+b)
        
        return output
    
    def ForwardForOneSample(self,X):
        Input = X
        Output = [X]
        PhiOutput = [X]
        for i in range(len(self.m_Weights)):
            anoutput = self.OneLayerForwardForOneSample(Input, self.m_Weights[i][0], self.m_Weights[i][1][0])
            Output.append(anoutput)
            if self.m_ActiveFunctions[i]=='log':
                phiout = []
                for anitem in anoutput:
                    phiout.append(MyUtil.LogtisticFunction(anitem))
                Input = phiout
            else:
                Input = anoutput
            PhiOutput.append(phiout)
        return [Output,PhiOutput]
    
    def UpdateWeightForOneSample(self,X,Y):
        '''
        X: a sample
        Y: a label
        '''
        [MiddleRawOutput,MiddleFinalOutput] = self.ForwardForOneSample(X)
        #output layer
        delta = []
        for i in range(len(self.m_Weights)):
            delta.append([])
        temp = []
        for anitem in MiddleRawOutput[-1]:
            temp.append(MyUtil.DerivationLogtisticFunction(anitem))
        delta[-1] = np.multiply((np.subtract(Y,MiddleFinalOutput[-1])),temp)
        print delta[-1]
        currentWeight = self.m_Weights[-1][0]
        #print 'before update'
        #print currentWeight
        for j in range(len(Y)):
            for i in range(self.m_LayerDimesions[-2]):
                currentWeight[j,i] =currentWeight[j,i] + self.m_LearningRate*delta[-1][j]*MiddleFinalOutput[-2][i]
        #print 'after update'
        #print currentWeight
        self.m_Weights[-1][0] = currentWeight  
        # self.m_Weights is a list
        # each item is an array with two items. 
        # The first item is the weight matrix
        # The second item is the bias.
        for i in range(-2,-1*len(self.m_LayerDimesions),-1):
            temp = []
            currentLayerDimension = len(self.m_Weights[i][0][0])
            nextLayerDimension = len(self.m_Weights[i][0])
            nextnextLayerDimension = self.m_LayerDimesions[i+1]
            for anitem in MiddleRawOutput[i]:
                temp.append(MyUtil.DerivationLogtisticFunction(anitem))
            
            for j in range(nextLayerDimension):
                atemp = 0
                for k in range(nextnextLayerDimension):
                    atemp +=np.multiply(delta[i+1][k],self.m_Weights[i+1][0][k,j])
                delta[i].append(atemp*temp[j])
            
            currentWeight = self.m_Weights[i][0]
            #print 'before update'
            #print currentWeight
            for j in range(nextLayerDimension):
                for k in range(currentLayerDimension):
                    currentWeight[j,k] =currentWeight[j,k] + self.m_LearningRate*delta[i][j]*MiddleFinalOutput[i-1][k]
            #print 'after update'
            #print currentWeight
            self.m_Weights[i][0] = currentWeight
            
        [MiddleRawOutput,MiddleFinalOutput] = self.ForwardForOneSample(X)
        error = MyUtil.SquareError(MiddleFinalOutput[-1], Y)
        print 'output is '+str(MiddleFinalOutput[-1])
        print 'Y is '+str(Y)
        return error
        
    def UpdateForASet(self,X,Y):
        '''
        X: a matrix, a row is a sample
        Y: a matrix, a row is a label
        '''
        if len(X)!=len(Y):
            print "Length different!"
            return
        totalError = float('inf')
        lastTotalError =float('inf')
        threshold = 1e-5
        iCurrentEcho = 0
        
        while True:
            temperror = 0.
            for i in range(len(X)):
                temperror += self.UpdateWeightForOneSample(X[i],Y[i])
            lastTotalError = totalError
            totalError = temperror
            iCurrentEcho+=1
            if np.abs(lastTotalError-totalError)<threshold or iCurrentEcho>self.m_EchoNumber:
                break 
        
        pickle.dump(self.m_Weights, open(self.m_ModelPath,'w'))
        
        output = []
        for i in range(len(X)):
            [a,b] = self.ForwardForOneSample(X[i])
            output.append(b[-1])
        np.savetxt('./ouput.txt',output)
        return totalError
    

if __name__=='__main__':
    X = [[1,2,3],[2,4,6],[-1,-2,-3]]
    b = 0.5
    W = [[2,2,2],[4,4,4]]
    Y = [[1],[1],[-1]]
    m = MLP([3,2,1],['log','log'],0.01,'./model')
    #out = m.ForwardForOneSample(X[0])
    #print out 
    err = m.UpdateForASet(X, Y)
    print err
    
        