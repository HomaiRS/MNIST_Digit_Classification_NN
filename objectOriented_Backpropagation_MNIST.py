"'Construct NN for MNIST dataset classification'"

#%%# read in data
import numpy as np
import MNIST_Reader
import random
import matplotlib.pyplot as plt

## Constructing Object Network:
class Network:
    def __init__(self, topology):
        self.topology = topology
        self.nLayers = len(topology)
        self.biasMx = [np.random.normal(0,0.1,(b,1)) for b in topology[1:]] #list arrays biases(each inner layer -w/o input)
        self.weightMx = [np.random.normal(0,0.1,(j,k)) for k,j in zip(topology[:-1],topology[1:])]  #paired (L,L-1) weight matrices 

def forwardStep(myNetwork,x):
    ## Feedforward Step: 
    activation = x    # First activation will be x (a_0)
    aLayer = [x]      # initialize list activations/layer
    zLayer = []       # initialize list z/layer
    for b,w in zip(myNetwork.biasMx,myNetwork.weightMx): #get lists of activations and zs per layer without last layer
        z = np.dot(w,activation) + b # z_L = w_L * a_L-1 + b_L
        zLayer.append(z)
        activation = sigmoid(z) # a_L = activation'(z_L)
        aLayer.append(activation)
    return (aLayer,zLayer)

def backpropagation(myNetwork,x,d):
    grad_b = [np.zeros(bVec.shape) for bVec in myNetwork.biasMx] #NablaB = [dC/db_L, dC/db_L-1, dC/db_L-2...]
    grad_w = [np.zeros(wVec.shape) for wVec in myNetwork.weightMx] #NablaW = [dC/dw_L, dC/dw_L-1, dC/dw_L-2...]
    
    ## ForwardStep
    aLayer,zLayer = forwardStep(myNetwork,x)

    ## Backprop Step - Last layer only (-1):  
    delta = dCdA(aLayer[-1],d) * d_sigmoid(zLayer[-1]) #delta_L = dC/dA_L * dA_L/dz_L = dC/dA_L * dA_L/tanh'(z_L); ##Note: for Hw4 d/a_2/d_z2 = 1    
    grad_b[-1] = delta #dC/db_L = delta
    grad_w[-1] = np.dot(delta,aLayer[-2].transpose()) #dc/dw_L = delta_L * activation_(L-1).transpose()
    
    ## Backprop Step - second to last layer etc:
    for layer in range(2,myNetwork.nLayers):
       z = zLayer[-layer] 
       delta = np.dot(myNetwork.weightMx[-layer+1].transpose(),delta)*d_sigmoid(z) #delta_L-1 = W.trans_L*delta_L(*)tanh'(z_L-1) 
       grad_b[-layer] = delta
       grad_w[-layer] = np.dot(delta,aLayer[-layer-1].transpose())
    return (grad_b,grad_w)

def dCdA(aL,d):
    return (aL-d)
def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def check_Accuracy(myNetwork,listData): #checking accuracy in validation set for current weights
    correctImg = 0;
    for x,d in listData:
        aLayer,zLayer = forwardStep(myNetwork,x)
        if(np.argmax(aLayer[-1]) == d):
            correctImg = correctImg + 1
    return correctImg      
  
#%%# Read in MNIST data
## Train data
trainingData, validationData, testData = MNIST_Reader.load_MNIST()
nTrainImg = 1000
trainingData = trainingData[0:nTrainImg]

## Validation data
nValidationImg = 100
validationData = validationData[0:nValidationImg]

## Test data
nTestImg = 50
testData = testData[0:nTestImg]

#%%# Define network topology and training parameters
topology = [784,3,10]
myNetwork = Network(topology)
#%%#
## Training parameters
maxEpoch = 500; eta = 0.1;

#%%# Training Data:
epoch = 0;
errorEpoch = np.ones([maxEpoch+1]); percValidation = np.ones([maxEpoch+1])
threshold = 95.0
sizeBatch = 1 #nImages/batch to train
dTestAccuracy = 10;
while ((epoch < maxEpoch) & (errorEpoch[epoch] < threshold) & (dTestAccuracy>0)):
    ## Step 1: Construct minibatches MNIST
    random.shuffle(trainingData)
    listBatches = [trainingData[k:k+sizeBatch] for k in range(0,len(trainingData),sizeBatch)]
    bIdx = 0; errorBatch = np.ones(len(listBatches))
    for batch in listBatches: ## update weights per batch
        imgCorrect = 0;
        ## Initialize grad biases and weights (same size as each Layer biases/weights)
        grad_b = [np.zeros(bVec.shape) for bVec in myNetwork.biasMx]
        grad_w = [np.zeros(wVec.shape) for wVec in myNetwork.weightMx]

        ## Step 2: Train on each image in batch: Get del_weight and del_bias (change from backprop)
        for x,d in batch:
            dgrad_b, dgrad_w = backpropagation(myNetwork,x,d)
            grad_b = [b+db for b,db in zip(grad_b,dgrad_b)]
            grad_w = [w+dw for w,dw in zip(grad_w,dgrad_w)]          
            
            ## check performance per image in batch
            aLayer,zLayer = forwardStep(myNetwork,x)
            if(np.argmax(aLayer[-1]) == np.argmax(d)):
                imgCorrect = imgCorrect + 1
        
        ## Step 2.1: Check performance in batch
        errorBatch[bIdx] = (imgCorrect/len(batch))*100
        
#        print('Batch ' + str(bIdx) +': '+ str(imgCorrect) + ' correct out of ' + str(len(batch)))
        
        ## Step 3: Update weights after each batch
        myNetwork.weightMx = [old_W-(eta)*new_W for old_W, new_W in zip(myNetwork.weightMx,grad_w)]    
        myNetwork.biasMx = [old_B-(eta)*new_B for old_B, new_B in zip(myNetwork.biasMx,grad_b)]
        bIdx = bIdx+1
              
    ## Step 4: Check Train error after every epoch:
    errorEpoch[epoch] = np.mean(errorBatch)
    print('Epoch ' + str(epoch) +' (Train/'+str(nTrainImg)+'Img) : '+ str(errorEpoch[epoch]) + ' %correct')
    
    ## Step 5: Check Test (validation) accuracy after each epoch
    accValidation = check_Accuracy(myNetwork, validationData)
    percValidation[epoch] = (accValidation/nValidationImg)*100
    print('Epoch ' + str(epoch) +' (Validation) /'+str(nValidationImg)+'Img) '+ str(percValidation[epoch]) + ' %correct')
    print("=="*20)
    if(epoch>8):  #if validation accuracy worsened after 5 epochs stop
        dTestAccuracy = percValidation[epoch]-percValidation[epoch-5] 
    epoch = epoch + 1

print('Finished training -- moving on to test data')
#%%# Plotting Error Terms (Train/Test)
width = 8; height = 5; resDPI = 130; fontSizeTicks = 18
labelSize = 16
fig, ax1 = plt.subplots(1, figsize=(width, height), dpi=resDPI)
ax1.plot(np.arange(0,epoch,1),errorEpoch[0:epoch], 'o-',color = 'black',ms = 3, linewidth = 0.6, label = 'Train Data ('+str(nTrainImg)+' Images)')
ax1.plot(np.arange(0,epoch,1),percValidation[0:epoch], 'o-',color = 'red',ms = 3, linewidth = 0.6, label = 'Validation Data (' +str(nValidationImg)+' Images)')

ax1.set_xlabel('Epochs',fontsize = labelSize)
ax1.set_ylabel('%Accuracy',fontsize = labelSize)
ax1.set_title('Network topology: ' + str(topology) + ', eta = ' + str(eta),fontsize = labelSize)
ax1.tick_params(axis="y", labelsize=fontSizeTicks)
ax1.tick_params(axis="x", labelsize=fontSizeTicks)

ax1.legend(loc='best', bbox_to_anchor=(0.8, 0.5),prop={'size': 16}, ncol = 2,title = '')
plt.legend()
plt.show()

print('maxEpochs Trained: ' + str(epoch))
print('N_train images: ' + str(nTrainImg))
print('N_Validation images: ' + str(nValidationImg))
print('Max Train Accuracy: ' + str(np.max(errorEpoch)) + '%  at epoch ' + str(np.argmax(errorEpoch)))
print('Max Validation Accuracy: ' + str(np.max(percValidation))+ '%  at epoch ' + str(np.argmax(percValidation)))

#%%# Evaluating weights against Validation data:

#%%#





