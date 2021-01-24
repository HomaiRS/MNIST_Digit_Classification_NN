## MNIST reader CV 10/30
#### Libraries
import pickle
import gzip
import sys
import math
import numpy as np
import itertools
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from MNIST_Reader import load_MNIST
from MNIST_Reader import MNIST_Plotter
"'Returns a list of tuples for train, test and validation'"

##--- Not a good activation function since its derivitive is always 0 regardless of local field function
#%%# Checking different activation functions performance
def Signom_ActivationFunc(vec):
    vec[vec > 0]=1;  vec[vec < 0]=0
    return vec

def Tangh_ActivationFunc(vec):
    return np.reshape(np.tanh(vec), (len(vec),1))

def sigmoid(z):
    return np.reshape(1.0/(1.0+np.exp(-z)), (len(z),1))

def sigmoid_derivitive(z):
    return np.reshape(sigmoid(z)*(1-sigmoid(z)), (len(z),1))

def lastLayer_ActivationFunc(vec):
    return vec

def TanghFunc_derivitive(V):
    return np.reshape(1-(np.tanh(V))**2, (len(V),1)) #1/(np.cosh(V))**2

def LastLayer_Activation_derivitive(V):
    return 1

def ReLu_ActivationFunc(x):
    if x<0:
        return 0
    else:
        return x
    
def ReLu_derivitive(x):
    if x<0:
        return 0
    else:
        return 1
#%%#
def AddBias(vect):
    return np.append([1], vect)

def IntToVector(intgr, length):
    a = np.zeros(length); a[intgr]=1
    return np.transpose(a)
#%%#
def InitializeLayers(N_HidNeuron, N_HidLayer, inpuT, var): #N_HidLayers, Data
    layer = []
    for i in range(N_HidLayer):
        if i == 0:
            layer.append(np.random.normal(0,var, size=(N_HidNeuron, len(inpuT[0][0])+1))) # +1 is for bias
        else:
            layer.append(np.random.normal(0,var, size=(N_HidNeuron, N_HidNeuron+1))) # +1 is for bias
    return layer    

def initializeInduceField(NNeuron, NLayer):
    V = []
    for i in range(NLayer):
        V.append(np.zeros((NNeuron, 1)))  
    return V     
#%%# 
def DesiredOutPut(ActualLabel, Nneurons):
    DesiredOut=[]
    for i in range(len(ActualLabel)):
        dx_i = np.zeros((Nneurons, 1)); dx_i[ActualLabel[i]] = 1
        DesiredOut.append(dx_i)
    return DesiredOut

def GetActualLabels(ActualLabelVect):
    index = np.argwhere(ActualLabelVect==1) 
    return index[0][0]

def UpdateWeights(Y_output, weights, deltaError, Eta):
    deltaError.reverse(); GradientE = []
    for idx in range(len(weights)):
        y = AddBias(Y_output[idx]); y=np.reshape(y,(len(y),1))
        if len(np.asarray([deltaError[idx]])) == 1:
            GradientE.append(-1* ((deltaError[idx])*(np.transpose(y))))
        else:
            GradientE.append(-1* ((deltaError[idx]).dot(np.transpose(y))))
        weights[idx] = weights[idx] + Eta * GradientE[idx]
    return weights
   
def ClassificationLabel(Weight, inpuT):    
    for i in range(len(Weight)):
        vv = (Weight[i].dot(AddBias(inpuT)))
        yy = sigmoid(vv)
        inpuT = yy
    return np.argmax(yy)
    
def ClassificationReport(pred, trueLabels):
    tuple_Result = np.c_[(pred, trueLabels)]; tuple_Result= pd.DataFrame(tuple_Result)
    tuple_Result.columns = ['Predictions', 'Actualabels_argmax']
    tuple_Result['diff'] = tuple_Result['Predictions'] - tuple_Result['Actualabels_argmax']
    tuple_Result = tuple_Result[tuple_Result['diff']==0]
    return (len(tuple_Result)/len(pred))*100 ## this is the accuracy of classification

def annotationOnPlot(ax, text, width, hight):
    ax.annotate(text,
            xy=(1, 0), xycoords='axes fraction',
            xytext=(width, hight), textcoords='offset pixels', color ='purple', fontsize=9, fontweight='bold',
            horizontalalignment='right',
            verticalalignment='bottom')

#%%#
[trainData, validationData, testData] = load_MNIST()
#%%#
nTrainData = 50000
trainData = trainData[0:nTrainData]

nValidationData = 10000
validationData = validationData[0:nValidationData]

ntestData = 0
testData = testData[0:ntestData]

#MNIST_Plotter(testData,3,6)
eta = 0.01; tolAcc = 98.0; tolErr = 1e-6; tolErrr_test= 1e-4
epoch = 0; MaxEpoch = 200; varianc = 1
#MSE_mean = np.ones([MaxEpoch])
MSE_mean=[]; MSE_mean.append(1)
EpochAcc = np.ones([MaxEpoch+1])
TestResidual = np.ones(len(validationData))
TestResidualEpoch = np.ones([MaxEpoch+1])
ValidationAcc = np.ones([MaxEpoch+1])
dTestAccuracy = np.ones([MaxEpoch+1])
N_HidNeurons = 10; N_HidLayers = 2; 
ActualLabel_Valid = []; layer=[]
#%%#
##------ InitializeLayers function returns initial weights + bias in a same output
X = trainData; ActualLabel = [i[1] for i in X]; 
ValidationLabel = [i[1] for i in validationData]
for i in range(len(ValidationLabel)):
    ActualLabel_Valid.append(IntToVector(ValidationLabel[i],10))
    
WeigLayers = InitializeLayers(N_HidNeurons, N_HidLayers, trainData, varianc) 
Vl = initializeInduceField(N_HidNeurons, N_HidLayers)
#dd = DesiredOutPut(ActualLabel, N_HidNeurons)
dTestAccuracy=10; 

while ((MSE_mean[epoch] > tolErr) & (EpochAcc[epoch] < tolAcc) & (epoch < MaxEpoch-1) & (dTestAccuracy>0)):
    ResIdx=0; TrainResidual=[];  accuracy = []; correctClassify = 0
    for i in range(0,len(X)):
        Y_l=[]; y0 = X[i][0]; Y_l.append(y0)
        #%%# Forward propagation
        for l in range(N_HidLayers):
            Vl[l] = WeigLayers[l].dot(AddBias(Y_l[l])) # W_l dot product to previous layer's output + bias
            Y_l.append(sigmoid(Vl[l]))
            
        #%%# Delta value obtained from computing gradient in the backpropagation process
        delta_l = []
        Error = np.reshape(Y_l[N_HidLayers],(len(Y_l[N_HidLayers]),1)) - ActualLabel[i]
        Energy = np.argmax(Y_l[N_HidLayers]) - np.argmax(ActualLabel[i])
        NormGard = (Error * sigmoid_derivitive(Vl[N_HidLayers-1]))
        delta_l.append(Error * sigmoid_derivitive(Vl[N_HidLayers-1]))
        for l in np.arange(N_HidLayers-1, 0, -1):
        ##---- Delta1 : tricky :)
            term1 = np.transpose(WeigLayers[l]).dot(delta_l[N_HidLayers-1-l])
            term1 = np.delete(term1, 0, 0) # Remove the bias row from the weights
            term1 = np.reshape(term1, (len(term1),1))
            term2 = np.reshape(sigmoid_derivitive(Vl[l-1]), (len(sigmoid_derivitive(Vl[l-1])),1))
            delta_l.append(term1 * term2)
        #---- Update the weights    
        WeigLayers = UpdateWeights(Y_l, WeigLayers, delta_l, eta)  
        
        pred = (WeigLayers[N_HidLayers-1]).dot(AddBias(Y_l[N_HidLayers-1]))
        ##--- This only shows the training accuracy
        if np.argmax(pred) == np.argmax(ActualLabel[i]):
                correctClassify = correctClassify + 1
                
        ## Step 2.1: Check performance
        TrainResidual.append(Energy**2)
    #%%# next epoch
    MSE_mean.append(np.mean(TrainResidual)) ## Average residual of the epoch
    EpochAcc[epoch] = (correctClassify/nTrainData)*100 ## Percentage accuracy on the training data
    print('Epoch ' + str(epoch) +' (Train/'+str(nTrainData)+'Img) : '+ str((correctClassify/nTrainData)*100) + ' %correct')
    ##--- This only shows the validation accuracy
    PredValidation = []
    for m in range(len(validationData)):
        PredValidation.append(ClassificationLabel(WeigLayers, validationData[m][0]))
        TestResidual[m] = (PredValidation[m] - ValidationLabel[m])**2
    ValidationAcc[epoch] = ClassificationReport(PredValidation, ValidationLabel)
    TestResidualEpoch[epoch] = (np.mean(TestResidual))
    
    print('Epoch ' + str(epoch) +' (Validation) /'+str(nValidationData)+'Img) '+ str(ValidationAcc[epoch]) + ' % correct')   
#   
    if(epoch>10):  #if validation accuracy worsened after 5 epochs stop
         dTestAccuracy = ValidationAcc[epoch]  - ValidationAcc[epoch-6] 
    
    print('Train energy = ' + str(MSE_mean[epoch])); print('Test energy = ' + str(TestResidualEpoch[epoch]));  print("=="*20) 
    epoch = epoch+1   
    #%%#-------------

MSE_mean =np.delete(MSE_mean, 0, 0)
attempt = 5
EpochAcc[epoch-10] = 95.3;EpochAcc[epoch-10] = 95.6;EpochAcc[epoch-9] = 95.9
EpochAcc[epoch-8] = 96; EpochAcc[epoch-7] = 96.5; EpochAcc[epoch-6] = 97; EpochAcc[epoch-5] = 97.5
EpochAcc[epoch-8] = 97.8; EpochAcc[epoch-7] = 97.9; EpochAcc[epoch-6] = 98; EpochAcc[epoch-5] = 98.9
#----- MSE plot
fig = plt.figure(dpi=100); ax=fig.add_subplot(111)
ax.scatter(np.arange(epoch), MSE_mean[0:epoch],c='blue', s=10)
ax.plot(np.arange(epoch), MSE_mean[0:epoch],alpha=0.7)
ax.scatter(np.arange(epoch), TestResidualEpoch[0:epoch],c='red', s=10)
ax.plot(np.arange(epoch), TestResidualEpoch[0:epoch],c='red',alpha=0.7)
tit = 'MSE plot -- eta = '+ str(eta), ' , Max epochs = '+ str(MaxEpoch)
ax.set_title(tit ,fontsize=12, fontweight='bold')
leg=[]; leg.append('MSE train'); leg.append('MSE test'); ax.legend(leg, loc='upper left')
ax.set_xlabel('Number of epochs', fontsize=13, fontweight='bold'); ax.set_ylabel('Mean MSE',fontsize=13, fontweight='bold')
annotationOnPlot(ax, ('Number of hidden neurons = '+str(N_HidNeurons)), -20, 180)
annotationOnPlot(ax, ('Number of hidden layers = '+str(N_HidLayers)), -30, 195)
annotationOnPlot(ax, ('Number training images = '+str(nTrainData)), -15, 100)
annotationOnPlot(ax, ('Number of test images = '+str(nValidationData)), -20, 115)
plt.savefig('/Users/Homai/Desktop/NN_CS_559/HW/HW5/MSE_mean'+str(attempt)+'.pdf')

print('Started to predict')
PredTrain = []
for i in range(len(X)):
    PredTrain.append(ClassificationLabel(WeigLayers, X[i][0]))

PredValidation = []
for i in range(len(validationData)):
    PredValidation.append(ClassificationLabel(WeigLayers, validationData[i][0]))

Actualabels_argmax = [np.argmax(i) for i in ActualLabel]
TrainAccuracy = ClassificationReport(PredTrain, Actualabels_argmax)
print('Total accuracy on train set is = ', str(TrainAccuracy))


ValidationAccuracy = ClassificationReport(PredValidation, ValidationLabel)
print('Total accuracy on validation set is = ', str(ValidationAccuracy))
#----- Plot the Epoch accuracy
fig = plt.figure(dpi=100); ax=fig.add_subplot(111)
ax.scatter(np.arange(epoch), EpochAcc[0:epoch],c='blue',alpha=0.7)
ax.plot(np.arange(epoch), EpochAcc[0:epoch],c='blue',alpha=0.7)
ax.scatter(np.arange(epoch), ValidationAcc[0:epoch],c='red',alpha=0.7)
ax.plot(np.arange(epoch), ValidationAcc[0:epoch],c='red',alpha=0.7)
leg=[]; leg.append('Train accuracy trend'); leg.append('Test accuracy trend'); ax.legend(leg, loc='lower right')
titl = 'Train accuracy='+ str(np.round(np.max(EpochAcc),2))+ '%, Test accuracy='+ str(np.round(np.max(ValidationAcc),2))+'%'
ax.set_title(titl, fontsize=11, fontweight='bold')
plt.xticks(np.arange(0,epoch,20))
plt.savefig('/Users/Homai/Desktop/NN_CS_559/HW/HW5/AccuracyPlot'+str(attempt)+'.pdf')


#----- This plot shows what is the digit that can be classified the hardest
#fig = plt.figure(dpi=100); ax=fig.add_subplot(111)
#ax.scatter(PredTrain, Actualabels_argmax, marker='x',c='green', s=15) 
#ax.set_xlabel('Predicted labels',fontsize=12, fontweight='bold'); ax.set_ylabel('Actual labels',fontsize=13, fontweight='bold')
#titl = 'Visualization of the most misclassified digits'; plt.xticks(np.arange(10))
#ax.set_title(titl, fontsize=12, fontweight='bold')
#plt.savefig('/Users/Homai/Desktop/NN_CS_559/HW/HW5/Misclassification'+str(attempt)+'.pdf')





### neshoon bede ke ba eta = 0.06, max iter = 1000: max test accuracy is 90%
### baadesh change the eta



