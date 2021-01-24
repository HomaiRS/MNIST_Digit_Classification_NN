#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:57:59 2019

@author: Homai
"""
import pickle
import itertools
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def load_MNIST():
    ## Read Data (50,000 train images, 10,000 validation images, 10,000 test images)
    f = open('mnist.pkl', 'rb')
    train, validation, test = pickle.load(f, encoding="latin1")
    f.close()
    ## Process input data: 
    trainImg = [np.reshape(x, (784, 1)) for x in train[0]]
    validationImgs = [np.reshape(x, (784, 1)) for x in validation[0]]
    testImgs = [np.reshape(x, (784, 1)) for x in test[0]]
    ## Vectorize the result: 
    trainLabelVec = []
    for d in train[1]:
        vec = np.zeros((10, 1));  vec[d] = 1.0
        trainLabelVec.append(vec)
    
    ## Make them into a tuple: 
    train = list(zip(trainImg, trainLabelVec))
    validation = list(zip(validationImgs,validation[1]))
    test = list(zip(testImgs,test[1]))
    return train, validation, test


def MNIST_Plotter(Data,N,M):
    f , ax = plt.subplots(N,M, figsize=(11,6.5)); u=-1
    for i, j in itertools.product(range(N), range(M)):
        u = u+1
        ax[i,j].imshow(Data[u][0].reshape((28, 28)), cmap=cm.Greys_r)
        print(Data[u][1])
        ax[i,j].set_title('True label: '+str(Data[u][1]), fontweight='bold', fontsize=13)
        ax[i,j].set_yticklabels([])
        ax[i,j].set_xticklabels([])
    plt.savefig('MNIST_figs.pdf', dpi=300)
