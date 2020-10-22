#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:24:22 2019

@author: lheurvin
"""

from applyML_algorithms import *
from fonctions import *

"""
This file is made for fonctions which will generate and save figures, plots for a lot of flights
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
def display_Clusters(database,n_clusters,flight_number,title,fname):
    """
    input:  original dataabse,number of clusters, number of flight, title and file name of the resulted png
    purpose: plot the different clusters on a map with x,y = latitude,longitude
    output: None
    """
    lat_min = min(database.latitude)
    lat_max = max(database.latitude)
    long_min = min(database.longitude)    
    long_max = max(database.longitude)
    
    fig = plt.figure(figsize=(40,20))
    ax = fig.add_subplot(111)
    m = Basemap(projection='merc',
               llcrnrlat = lat_min-1,
               urcrnrlat = lat_max+1,
               llcrnrlon = long_min-1,
               urcrnrlon = long_max+1, 
               resolution = 'c')
    
    m.drawcoastlines()
    m.drawcountries(color='black')
    m.drawlsmask(land_color='cornsilk',ocean_color='lightblue',lakes=True)
    #m.fillcontinents(color='white', lake_color='aqua')
    #m.etopo()

    label_color = [LABEL_COLOR_MAP[l] for l in database[str(n_clusters) + 'clusters']]
    m.scatter(x=database['longitude'].values, 
                  y=database['latitude'].values, 
                  c=label_color,    
                  latlon=True,
                  s = 25) 
    m.drawparallels(np.arange(-90.,91.,2.),labels=[True,True,True,True],dashes=[2,2])
    m.drawmeridians(np.arange(-180.,181.,3.),labels=[True,True,True,True],dashes=[2,2])

    ax.set_xlabel('longitude', fontsize='xx-large')
    ax.set_ylabel('latitude', fontsize='xx-large')
    ax.set_title(title, fontsize='xx-large')   
#    plt.savefig(fname=fname)
#    plt.close('all')
    
    
def plot_all_flights():
    """
    input: 
    purpose: plot the clustering flight per flight, and save the png file in cleaned_datas folder
             with 'exclusion', you choose which datas are not in the Kmeans algorithm
             The best option looks to not take altitude in account : exclusion = [['time','latitude','altitude','longitude']]
             
             You can choose the flights you want to plot and also the number of clusters to try for Kmeans
             If you don'y know which number of cluster to choose, have a look at this function : 
                 validation_nb_cluster(n_flight);
    output: None
    """
    exclusion = [['time','latitude','longitude'],['time','latitude','longitude','altitude','cpc'],['time','latitude','longitude','altitude'],['time','latitude','longitude','cpc']]
#    exclusion = [['time','latitude','altitude','longitude','no','no2']]
    exclusion = [['time','latitude','altitude','longitude','org','no2']]
    for i in range(24,25):
        for exclude_datas in exclusion:
            if i not in [19,20,21,25,32,36]: 
                database = create_database(i)
                kmeans = zscore(database.copy().drop(exclude_datas,axis=1))
                final_database = database.copy()
                for n_clusters in range(6,7):
                    cluster_labels = Kmeans(kmeans,n_clusters)
#                    silhouette_avg = silhouette_score(kmeans, cluster_labels)
                    final_database[str(n_clusters)+'clusters'] = cluster_labels
                    kmeans[str(n_clusters)+'clusters'] = cluster_labels
#                    fname = 'datas/cleaned_data/'+str(i)+'/withoutnono2/' +str(n_clusters) + 'clusters ' + 'without altitude,no,no2 ' + 'with cpc'
#                    title = 'Kmeans algorithm with ' + str(n_clusters) + ' clusters.'+'\n' + 'the average silhouette_score is : ' + str(silhouette_avg) +'\n' + 'without altitude,no,no2 ' + 'with cpc'
                    if 'altitude' in exclude_datas:
                        if 'cpc' in exclude_datas:
                            fname = 'datas/cleaned_data/'+str(i)+'/F'+str(i)+' ' +str(n_clusters) + 'clusters ' + 'without altitude ' + 'without cpc'
                            title = 'Kmeans algorithm with ' + str(n_clusters) + ' clusters.'+'\n' + 'without altitude ' + 'without cpc' 
                        else:
                            fname = 'datas/cleaned_data/'+str(i)+'/F'+str(i)+' ' +str(n_clusters) + 'clusters ' + 'without altitude ' + 'with cpc'
                            title = 'Kmeans algorithm with ' + str(n_clusters) + ' clusters.'+'\n' + 'without altitude ' + 'with cpc' 
                    else : 
                        if 'cpc' in exclude_datas:
                            fname = 'datas/cleaned_data/'+str(i)+'/F'+str(i)+' ' +str(n_clusters) + 'clusters  ' + 'with altitude ' + 'without cpc'
                            title = 'Kmeans algorithm with ' + str(n_clusters) + ' clusters.'+'\n' + '\n' + 'with altitude ' + 'without cpc' 
                        else:
                            fname = 'datas/cleaned_data/'+str(i)+'/F'+str(i)+' ' +str(n_clusters) + 'clusters ' + 'with altitude ' + 'with cpc'
                            title = 'Kmeans algorithm with ' + str(n_clusters) + ' clusters.'+'\n' + 'with altitude ' + 'with cpc' 
#                    
                    display_Clusters(final_database,n_clusters,i,title,fname)
                    
                    
def plot_all_map():
    """
    input: 
    purpose: plot the clustering for every flight, and save the png file in cleaned_datas/0 folder
             with 'exclusion', you choose which datas are not in the Kmeans algorithm
             The best option looks to not take altitude in account : exclusion = [['time','latitude','altitude','longitude']]
             
             You can choose the number of clusters to try for Kmeans
             If you don'y know which number of cluster to choose, have a look at this function : 
                 validation_nb_cluster(n_flight) and chose as database create_database_all_flights()
    output: None
    """
    exclusion = ['time','latitude','altitude','longitude']
    all_flights_database = create_database_all_flights()
    all_flights_normalised = zscore(all_flights_database)
    all_flights_final_database = all_flights_database.copy()
    kmeans= all_flights_normalised.drop(exclusion, axis=1) 
    for n_clusters in range(3,7):
                #without altitude in acount
        cluster_labels = Kmeans(kmeans,n_clusters)
        silhouette_avg = silhouette_score(kmeans, cluster_labels)
        
        all_flights_final_database[str(n_clusters)+'clusters'] = cluster_labels
        display_Clusters(all_flights_final_database,n_clusters,'',0,'')

def box_plot(n_flight,n_clusters):
    """
    input: number of flight, number of clusters
    purpose: plot boxes to see the repartitions of the different parameters (param concentrations) for different clusters
             different scales used for plotting
    output: None
    """
    exclude_datas = ['time','latitude','longitude']
    database = create_database(n_flight)
    normalised = zscore(database.copy().drop(exclude_datas,axis=1))
    #Dont take altitude for k-means clustering
    kmeans = normalised.drop(['altitude'],axis=1)
    cluster_labels = Kmeans(kmeans,n_clusters)
    
#    log2_database = database.apply(lambda x : [np.log2(i) if i>0 else -3 for i in x])
    log10_database = database.copy().drop(exclude_datas,axis=1)
    log10_database = database.apply(lambda x : [np.log10(i) if i>0 else -2.5 for i in x])
    
    database['num of cluster'] = cluster_labels
#    normalised[str(n_clusters)+'clusters'] = cluster_labels
#    log2_database[str(n_clusters)+'clusters'] = cluster_labels
    log10_database['num of cluster'] = cluster_labels
    
#    box_normalised = normalised.boxplot(by=[str(n_clusters)+'clusters'])
#    box_log2 = log2_database.boxplot(by=[str(n_clusters)+'clusters'])
    L_separation_for_plotting = [['altitude','num of cluster'],['cpc','num of cluster'],
                                  ['org','num of cluster'],
                                    ['so4','num of cluster'],['no3','num of cluster'],
                                    ['nh4','num of cluster'],['chl','num of cluster'],
                                    ['o3','num of cluster'],['nox','num of cluster'],
                                    ['no','num of cluster'],['no2','num of cluster'],
                                    ['so2','num of cluster']]
#    ['altitude',str(n_clusters)+'clusters']['org','so4','no3','nh4','chl',str(n_clusters)+'clusters'],['o3','nox','no','no2','so2',str(n_clusters)+'clusters']]
## OPTION 1 
    fig, axi = plt.subplots(3,4, figsize=(90, 50))
    axi=axi.flatten()
    for i in range(len(L_separation_for_plotting)):
        current_df = log10_database[L_separation_for_plotting[i]]    
        box_i = current_df.boxplot(by=['num of cluster'],ax=axi[i])
        fig.tight_layout(pad=5,h_pad=2.5,w_pad=4)
    # get rid of the automatic 'Boxplot grouped by group_by_column_name' title
    plt.suptitle("")
    plt.savefig('best_boxplot',optimize=True)
    return database

def box_plot2(n_flight,n_clusters):
    """
    
    output: None
    """
    exclude_datas = ['time','latitude','longitude']
    database = create_database(n_flight)
#    database = create_database_all_flights()
    database = database.drop(exclude_datas,axis=1)
      
    normalised = zscore(database.copy())
    #Dont take altitude for k-means clustering
    kmeans = normalised.drop(['altitude'],axis=1)
    cluster_labels = Kmeans(kmeans,n_clusters)
    
    log10_database = database.apply(lambda x : [np.log10(i) for i in x])
    log10_database[str(n_clusters)+'clusters'] = cluster_labels
    normalised[str(n_clusters)+'clusters'] = cluster_labels
    database[str(n_clusters)+'clusters'] = cluster_labels
    fig = plt.figure(figsize=(40,20))
    c=1
    for i in range(1):
        c_i = log10_database[log10_database[str(n_clusters)+'clusters'] == i]
        c_i = c_i.drop([str(n_clusters)+'clusters'],axis=1)
        j=0
        for col in c_i.columns:     
            j+=1
            fig.add_subplot(1,len(c_i.columns),c) 
            c_i.boxplot(column=col)
            c+=1
            fig.tight_layout(pad=5,h_pad=2.5,w_pad=4)
    plt.savefig('boxplotf246c')
#    fig2 = plt.figure(figsize=(40,20))
#    for i in range(n_clusters):
#        c_i = database[database[str(n_clusters)+'clusters'] == i]
#        c_i = c_i.drop([str(n_clusters)+'clusters','altitude','cpc'],axis=1)
#        ax = fig2.add_subplot(n_clusters,1,i+1) 
#        c_i.boxplot()
#    fig3 = plt.figure(figsize=(40,20))
#    for i in range(n_clusters):
#        c_i = normalised[normalised[str(n_clusters)+'clusters'] == i]
#        c_i = c_i.drop([str(n_clusters)+'clusters'],axis=1)
#        c_i = 3*c_i
#        ax = fig3.add_subplot(n_clusters,1,i+1) 
#        c_i.boxplot()


