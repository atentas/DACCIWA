#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:38:53 2019

@author: lheurvin
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.ticker import AutoLocator, AutoMinorLocator

from mpl_toolkits.basemap import Basemap

import statistics
import sklearn

import silouhette_kmeans 
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score,davies_bouldin_score

from averaging import *
from read_apitof import *
from functions import *

import itertools # to flatten list 

import sklearn.decomposition

### These are 2 ways to create the input for the database creation
#exclude_data = ['time','latitude','altitude','longitude','pressure','temperature','wind_dir','wind_speed',
#                'rapport_u/d_long','rapport_u/d_short']
include_data = ['org','so4','no3','nh4','chl','o3','nox','no','no2','so2','co','cpc']

L_separation_for_plotting = [[i,'label'] for i in include_data]
L_separation_for_plotting.append(['altitude','label'])
L_separation_for_plotting.append(['rapport_u/d_short','label'])

""" 
    LABEL_COLOR_MAP used to color the different labels : if you try algorithms with moer that 11 
    clusters, it won't work. Just add other colors in the dictionary. 
"""
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2 : 'g',
                   3 : 'c',
                   4 : 'y', 
                   5 : 'b',
                   6 : 'maroon',
                   7 : 'springgreen',
                   8 : 'white',
                   9 : 'orange',
                   10 : 'grey'}


def validation_nb_cluster(n_flight):
    """
    input : n_flight : number of the flight 
    purpose : find the ideal number of clusters for Kmeans algorithms for the current flight n_flight
    output : None - display of different methods results
    
    remark : if we want to include all the database and not only one flight : n_flight = 0
    """
    database = create_database(n_flight)
    database = database[include_data]
    normalised = zscore(database.copy())
    #Dont take altitude for k-means clustering
    kmeans = normalised
    Lx = [3,4,5,6,7,8,9,10,11,12,13,14,15]
    dbi=[] #dbi_method
    silou=[] #silouhette method
    elbow = []
    for n_clusters in Lx:
        km = KMeans(n_clusters=n_clusters, init='k-means++',max_iter=1000, n_init=10, random_state=0)
        cluster_labels = km.fit_predict(kmeans)  
        dbi.append(davies_bouldin_score(kmeans,cluster_labels))
        silou.append(silhouette_score(kmeans, cluster_labels))
        elbow.append(km.inertia_)
    plt.figure('Ideal number of clusters with diffrent methods')
    plt.subplot(2,2,1)
    plt.plot(Lx,dbi,'r',label='DBI method')
    plt.xlabel('number of clusters')
    plt.ylabel('dbi score')
    plt.subplot(2,2,2)
    plt.plot(Lx,silou,'b',label='silouhette')
    plt.xlabel('number of clusters')
    plt.ylabel('silouhette score')
    plt.subplot(2,1,2)
    plt.plot(Lx,elbow,'g',label='elbow method')
    plt.xlabel('number of clusters')
    plt.ylabel('sum squaerd distance')
    plt.title('flight '+ str(n_flight))
    
def validation_nb_cluster_all_database(database):
    """
    input : database : database already normalized
    purpose : find the ideal number of clusters for Kmeans algorithms for the current database
    output : None - display of different methods results
    """
    kmeans = database
    Lx = [3,4,5,6,7,8,9,10,11,12,13,14,15]
    dbi=[] #dbi_method
    silou=[] #silouhette method
    elbow = []
    for n_clusters in Lx:
        km = KMeans(n_clusters=n_clusters, init='k-means++',max_iter=1000, n_init=10, random_state=0)
        cluster_labels = km.fit_predict(kmeans)  
        dbi.append(davies_bouldin_score(kmeans,cluster_labels))
        silou.append(silhouette_score(kmeans, cluster_labels))
        elbow.append(km.inertia_)
    plt.figure('Ideal number of clusters with diffrent methods dtb')
    plt.subplot(2,2,1)
    plt.plot(Lx,dbi,'r',label='DBI method')
    plt.xlabel('number of clusters')
    plt.ylabel('dbi score')
    plt.subplot(2,2,2)
    plt.plot(Lx,silou,'b',label='silouhette')
    plt.xlabel('number of clusters')
    plt.ylabel('silouhette score')
    plt.subplot(2,1,2)
    plt.plot(Lx,elbow,'g',label='elbow method')
    plt.xlabel('number of clusters')
    plt.ylabel('sum squaerd distance')
#    plt.title('flight '+ str(n_flight))


def cluster_database(n_flight,n_clusters):
    """
    input : n_flight : number of the flight
            n_clusters : number of clusters wanted
    purpose : add the label of each data point in the database
    output : original database with the colomn 'label' added
    """
    database = create_database(n_flight)
    kmeans = zscore(database[include_data])
    final_database = database.copy()
    cluster_labels = Kmeans(kmeans,n_clusters)
    final_database['label'] = cluster_labels
    return final_database

def pca_cluster_database(n_flight,n_clusters,n_columns_to_keep):
    """
    input : n_flight : number of the flight
            n_clusters : number of clusters wanted
            n_columns_to_keep : number of columns to keep for the pca database
    purpose : same as cluster_database ; but we use the PCA matrix as database
    output : original database with the colomn 'label' added
    """
    database = create_database(n_flight)
    database2 = zscore(database)
    database2 = database2[include_data]
    V,Ypca, D = PCA(database2)
    assignment1 = Kmeans(Ypca[:,0:n_columns_to_keep],n_clusters)
    database['label'] = assignment1
    return database


"""
PCA
"""

def number_of_columns_to_keep(V,Ypca,D):
    """
    input : V,Ypca,D as result of PCA function 
    purpose : Find the number of columns to reach 90%  of cumulative variance
    output : None
    """
    idc = np.divide(np.cumsum(D),np.sum(D)) 
    # We plot normalized cumulative sum to understand the contributions of the obtained PCs
    plt.figure('Variance')
    plt.title('Cumulative Pourcentage of the PC')
    plt.xlabel('Principal Components')
    plt.ylabel('Pourcentage %')
    #plt.plot(xrange(1,len(D)+1),idc,'bo') # re-plot the data
    plt.plot(range(1,len(D)+1),idc*100,'bo') # re-plot the data
    plt.grid()
    plt.show()
    
    
def pca3D(V,Ypca,D,n_clusters):
    """
    input : V,Ypca,D as result of PCA function  ; n_clusters = number of clusters 
    purpose : Plot the datapoints in the first 3 Principal Components dimension of the PCA matrix
    output : None
    """
    assignment1 = Ypca['label']
    fig2 = plt.figure('PCA3D')
    ax = Axes3D(fig2)
    ax.set_xlabel('Principal Compound 1')
    ax.set_ylabel('Principal Compound 2')
    ax.set_zlabel('Principal Compound 3')
    ax.set_title('Observation of the dataset in 3D after PCA algorithm')
    for i in range (n_clusters):
        ax.scatter(Ypca[Ypca['label']==i][0],Ypca[Ypca['label']==i][1],Ypca[Ypca['label']==i][2],c=LABEL_COLOR_MAP[i])

def pca1D(clustered_database,V,Ypca,D,n_clusters):
    """
    same as pca3D but we plot the datapoints in all the dimensions, one by one, to understand the sense of the PCA columns  
    """
    assignment1 = Ypca['label']
    fig, axi = plt.subplots(len(Ypca.T)//2,len(Ypca.T)//2, figsize=(90, 50),num='1d')
    axi=axi.flatten()
    for j in range(0,len(Ypca.T)-1):
        for i in range (n_clusters):
            axi[j].scatter(clustered_database[clustered_database['label']==i]['time'],Ypca[Ypca['label']==i][j].values,c=LABEL_COLOR_MAP[i],s=3.5)
        axi[j].set_ylabel('normalized value PC ' + str(j))
        axi[j].set_xlabel('time in the day(s)')

def components_evolution(clustered_database,n_clusters):
    """
    input : clustered database  ; n_clusters = number of clusters 
    purpose : Plot the evolution of the value of each input with the labeled color of the datapoint
    output : None
    """
    inte = clustered_database[include_data+['altitude','rapport_u/d_short']]
    inte = inte.apply(lambda x : [np.log10(i) if i>0 else -2.5 for i in x])
    inte['label']=clustered_database['label']
    print(inte)
    print(clustered_database)
    fig, axi = plt.subplots(4,5, figsize=(90, 50),num='COMPONENTS EVOLUTION')
    axi=axi.flatten()
    k=0
    for j in list(inte.drop(['label'],axis=1).columns):
        for i in range (n_clusters):
            axi[k].scatter(clustered_database[clustered_database['label']==i]['time'],inte[inte['label']==i][j].values,c=LABEL_COLOR_MAP[i],s=3.5)
        axi[k].set_ylabel('log('+j+')')
        axi[k].set_xlabel('time in the day(s)')
        k+=1
    fig.tight_layout(pad=15,h_pad=20,w_pad=20)
    
def understand_pca(n_flight):
    """
    imput : n_flight
    output : return V,Y,D fora given flight
    """
    database = zscore(create_database(n_flight))
    database = database[include_data]
    V,Ypca, D = PCA(database)
    Vpca = pd.DataFrame(data=V,    # values
                    index=include_data,    # 1st column as index
                    columns=[i for i in range(0,len(include_data))])

    return Vpca,Ypca,D

"""
DISPLAY
"""

def display_clustering(clustered_database,n_clusters,title='',save=False,fname=''):
    """
    input : n_clusters = number of clusters 
            clustered_database
            
    purpose : display the clustering on a map wit labels colored 
    output : None
    """
    database = clustered_database
    lat_min = min(database.latitude)
    lat_max = max(database.latitude)
    long_min = min(database.longitude)    
    long_max = max(database.longitude)
    
    fig = plt.figure(figsize=(40,20))
    ax = fig.add_subplot(111)
    m = Basemap(projection='merc',
               llcrnrlat = lat_min-0.5,
               urcrnrlat = lat_max+0.5,
               llcrnrlon = long_min-0.5,
               urcrnrlon = long_max+0.5, 
               resolution = 'f')
    
    m.drawcoastlines()
    m.drawcountries(color='black')
    m.drawlsmask(land_color='cornsilk',ocean_color='lightblue',lakes=True)
    #m.fillcontinents(color='white', lake_color='aqua')
    #m.etopo()
    for i in range(0, n_clusters):
        m.scatter(x=database[database['label'] == i]['longitude'].values, 
                  y=database[database['label'] == i]['latitude'].values,
                  c=LABEL_COLOR_MAP[i], latlon=True, s=130, label = 'cluster' + str(i))
    plt.legend(scatterpoints=1)
    m.drawparallels(np.arange(-90.,91.,2.),labels=[True,True,True,True],dashes=[2,2], fontsize = 'larger')
    m.drawmeridians(np.arange(-180.,181.,3.),labels=[True,True,True,True],dashes=[2,2],fontsize = 'larger')
    ax.set_xlabel('longitude', fontsize='larger')
    ax.set_ylabel('latitude', fontsize='larger')
    ax.set_title(title,fontsize='larger')
    if save:
        plt.savefig(fname +'.png',format='png',dpi='1000')
    
"""
BOX PLOT 
"""
            
def box_plot_from_database(database): #we dont apply the label to log10
    """
    input: clustered database
    purpose: plot boxes to see the repartitions of the different parameters (param concentrations) for different clusters
             different scales used for plotting
    output: None
    """
    log10_database = database[include_data + ['altitude','rapport_u/d_short'] ].copy()
    
    
    log10_database = log10_database.apply(lambda x : [np.log10(i) if i>0 else -2.5 for i in x])
    log10_database['label'] = database['label']

    fig, axi = plt.subplots(4,4, figsize=(90, 50))
    axi=axi.flatten()
    for i in range(len(L_separation_for_plotting)):
        current_df = log10_database[L_separation_for_plotting[i]]    
        box_i = current_df.boxplot(by=['label'],ax=axi[i])
        fig.tight_layout(pad=6,h_pad=9,w_pad=5)
    # get rid of the automatic 'Boxplot grouped by group_by_column_name' title
    plt.suptitle("")
    
def boxplot_final(database,n_clusters):
    """
    input: clustered database
    purpose: plot boxes to see the repartitions of the different parameters (param concentrations) for different clusters
             different scales used for plotting
    output: None
    """
    log10_database = database[include_data+ ['altitude','rapport_u/d_short']]
    log10_database = log10_database.apply(lambda x : [np.log10(i) if i>0 else -2.5 for i in x])
    log10_database['label'] = database['label']

# OPTION 2 
    A = np.zeros((len(L_separation_for_plotting),n_clusters))
    for i in range(len(L_separation_for_plotting)):
        current_df = log10_database[L_separation_for_plotting[i]]    
        box_i = current_df.boxplot(by=['label'])
        for j in range (n_clusters):
            A[i,j]=statistics.median(current_df[current_df['label'] == j][L_separation_for_plotting[i][0]])
        plt.suptitle("")
#        plt.savefig(str(i),optimize=True)
        plt.close()
    df = pd.DataFrame(A,index =log10_database.drop(['label'],axis=1).columns,
                        columns = [k for k in range (n_clusters)])
    return df