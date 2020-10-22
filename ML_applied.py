#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:46:20 2019

@author: lheurvin
"""
""" FONCTIONS IMPORT """ 

#Basics
import numpy as np
import pandas as pd 

import statistics

import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.ticker import AutoLocator, AutoMinorLocator

#Basemap: plot maps behind figures 
from mpl_toolkits.basemap import Basemap

#ML
import silouhette_kmeans 
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score,davies_bouldin_score

#import
from averaging import *
from read_apitof import *
from functions import *
from applyML_algorithms import *

### Choose n_flight = 0 to get all flights combined
n_flight = 24
"""
Step 1.1 :
    WHICH NUMBER OF CLUSTERS
    Test what is the best number of cluster (original database)
"""
#validation_nb_cluster(n_flight)
##6 clusters for F24
"""
Step 1.2 :
    WHICH NUMBER OF CLUSTER AFTER PCA 
    We check the number of colums to keep in the new matrix
    Apply PCA and check the number of cluster to keep ; Here Ypca reprensents the 
    dataframe in the new dimension of Principal Components; 
"""
Vpca,Ypca,Dpca = understand_pca(n_flight);
number_of_columns_to_keep(Vpca,Ypca,Dpca)
n_columns_to_keep = 4   ### we keep the 4 first columns 
Ypca = Ypca[:,0:n_columns_to_keep]
Ypca = pd.DataFrame(data=Ypca,
                    columns = [i for i in range(n_columns_to_keep)])
##
validation_nb_cluster_all_database(Ypca)
#6 clusters if we keep all the columns
n_clusters = 5 #with n_columns_to_keep = 4 ~ 90% of the information

"""
#Step 2 : 
    CORRELATION MATRIX
    Check if there are correlations between the variables/inputs 
    So we create the correlation matrix, either only for one flight or for all flights
    
    Conclusion : maybe we could remove org or no3 and nox or no2 from the Kmeans database
""" 
#correlation_matrix = correlation_matrix(n_flight)

"""
Step 3 : 
    VISUALISATION MAP with clusters : 
        Create the clustered database
        first without pca 
        then with pca 
        
    *You can change the resolution while definiting the basemap : 'f' for finest
                                                                  'h' for high                                              
                                                                  'c' for classic
    Parameters : 
        (n_flight,n_clusters,title of the image, save (boolean), fname (str) only if save == True)
        -> save is True if you directly want to save the figure, and so you give a fname after.
        But the best thing is to not save and to save from the figure 
        
#display_clustering(clustered_db1,n_clusters,title='K means clustering with' + str(n_clusters) +' clusters',
#                  save=True,fname=str(n_clusters) + '_clusters')
#display_clustering(clustered_db2,n_clusters,title='K means clustering with' + str(n_clusters) +' clusters - after pca',
#                  save=True,fname=str(n_clusters) + '_clusters_after_pca')    
                
"""
#clustered_db1 = cluster_database(n_flight,n_clusters) #clustered database without PCA applied
clustered_db2 = pca_cluster_database(n_flight,n_clusters,n_columns_to_keep) #Clustered database with PCA applied
#
#
"""
Step 4.1 : Look at the evolution of the different compounds through time 
         For this we consider the clustering made with the pca database (ie clustered_db2)    
"""
#components_evolution(clustered_db2,n_clusters)
"""
Step 4.2 : Look at the evolution of the different Principal Components in 1D 
           And also plot the datapoints in the dimension of the 3 first Principal Components
        
"""
#Ypca['label']=clustered_db2['label']
#pca1D(clustered_db2, Vpca, Ypca, Dpca, n_clusters)
#pca3D(Vpca, Ypca, Dpca, n_clusters)
"""
Step 5
DISPLAY THE MAP 
        - with the clustering from original database
        - with the clustering from the PCA database
"""
#display_clustering(clustered_db1,n_clusters,title='K means clustering with ' + str(n_clusters) +' clusters')
#display_clustering(clustered_db2,n_clusters,title='K means clustering with ' + str(n_clusters) +' clusters - after pca')
#

"""
Step 6 :
    BOXPLOT
    first : fast vizualisation in Python : good for test ! We can also directly save it from the figure
    second : plot each input one by one and save it in the current folder ; 
             also return a database of the median values for each parameters in 
             the clusters  
"""
box_plot_from_database(clustered_db2)
#median_values1 = boxplot_final(clustered_db1,n_clusters).T
#median_values2 = boxplot_final(clustered_db2,n_clusters).T