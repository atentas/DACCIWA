#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:20:37 2019

@author: lheurvin
"""

import numpy as np
import pandas as pd 

from averaging import *
include_data = ['org','so4','no3','nh4','chl','o3','nox','no','no2','so2','co','cpc']

def Kmeans(data,n_clusters):
    """
    inputs: data,n_clusters
    purpose: apply kmeans algorithm to data with n_clusters
    output: prediction of cluster labels 
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++',max_iter=1000, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(data)
    return pred_y

def covariance (X): # function for matrix X
    """
    input:  matrix X 
    purpose: get covariance matrix 
    output: covariance matrix
    """
    [nX,mX]=X.shape # get the size of the matrix X
    meanX = np.divide(np.sum(X, axis=0),nX)  # mean row of matrix X
    zX = X - np.kron(np.ones((nX,1)),meanX) # zX = [X - meanX]
    covX = np.divide(np.dot(zX.T,zX),nX-1) # covariance matrix
    return covX

def PCA(X):
    """
    input:  matrix X 
    purpose: transform matrix 
    output: V = eigenvectors
            Yn = transformed matrix
            D = eigenvalues
            
    """
    XCov=covariance(X) # this is the same as XCov=np.cov(Xn.T)
#    XCov = np.cov(X.T)
    D, V = np.linalg.eig(XCov) # D is eigval and V is eigvec
    Yn=np.dot(X,V)             # perform the linear transformation
                               # by multiplying the original matrix
                               # with eigenvector
    return V,Yn,D

def zscore(X): # z-score uses to normalise the data.
    """
    input:  matrix X 
    purpose: normalise the matrix 
    output: normalized matrix 
    """
    [nX,mX]=X.shape              # X has NxD
    XMean=np.mean(X,axis=0)      # take the mean of every row X
    XStd=np.std(X,axis=0,ddof=1) # take the std of every row X
    zX = X - np.kron(np.ones((nX,1)),XMean) # Z=[X - mX]
    Zscore = np.divide(zX,XStd)             # Zscore = Z/Xstd
    return Zscore

def correlation_matrix(n_flight):
    """
    input: number of flight
    purpose: see the link between parameters
    output: None
    """
    data = create_database(n_flight)
    data = data[include_data]
    corr = data.corr()
    fig = plt.figure('Correlation matrix')
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.title('Correlation between variables')
#    plt.savefig(fname='CorrelationMatrix'+str(n_flight)+'.png')
#    plt.savefig(fname='CorrelationMatrix.png')
    plt.show()