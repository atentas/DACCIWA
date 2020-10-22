#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:04:46 2019

@author: lheurvin
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans 

def lecture(n_flight,time):
    """
    input:  number of flight to read, average time choosen 
    purpose: read and load parameters,data, keep useful columns and rows for which ams,gas and cpc data are available
    output: ams,gas,cpc
    """
    gas = pd.read_csv('data/cleaned_data/'+str(n_flight)+'/gas_phases.txt',sep=' ')
    gas = gas[['time','O3','NOx','NO','NO2','SO2']] 
    
    ams = pd.read_csv('data/cleaned_data/'+str(n_flight)+'/ams.txt',sep=' ')
    ams = ams[['time','latitude','longitude','altitude','org','so4','no3','nh4','chl']]
    
    cpc = pd.read_csv('data/cleaned_data/'+str(n_flight)+'/cpc.txt',sep=' ')
    cpc = cpc[['time','cpc']]
    
    co = pd.read_csv('data/cleaned_data/'+str(n_flight)+'/co.txt',sep=' ')
    co = co[['time','co']]
    
    core = pd.read_csv('data/cleaned_data/'+str(n_flight)+'/core.txt',sep=' ',dtype = np.float64)
    core = core[['time','pressure','temperature','wind_dir','wind_speed','rapport_u/d_long','rapport_u/d_short']]
    
    gas = gas[gas['time'] >= ams['time'][0]]
    gas = gas[gas['time'] <= ams['time'][len(ams)-1]+time]
    gas = gas.reset_index(drop=True)
    
    co = co[co['time'] >= ams['time'][0]]
    co = co[co['time'] <= ams['time'][len(ams)-1]+time]
    co = co.reset_index(drop=True)
       
    cpc = cpc[cpc['time'] >= ams['time'][0]]
    cpc = cpc[cpc['time'] <= ams['time'][len(ams)-1]+time]
    cpc = cpc.reset_index(drop=True)
    
    core = core[core['time'] >= ams['time'][0]]
    core = core[core['time'] <= ams['time'][len(ams)-1]+time]
    core = core.reset_index(drop=True)
    core = core.astype(np.float64)
    ti = ams['time'][0]
#    data_mz69 = pd.read_csv('cleaned_data/'+str(17)+'/voc_mz69.txt',sep=' ')
#    data_mz69 =  data_mz69[['time','mz69']]
#
#    data_mz137 = pd.read_csv('cleaned_data/'+str(n_flight)+'/voc_mz137.txt',sep=' ')
#    data_mz137 =  data_mz137[['time','mz137']]
#    return data, data_mz69,data_mz137
#    return core
    return ams,gas,co,cpc,core,ti

def create_database(n_flight):
    """
    input:  number of flight 
    purpose: merge parameters
    output: database 
    """
    if n_flight == 0:
        return load_complete_db()
    avrg_time = 10
    ams,gas,co,cpc,core,ti = lecture(n_flight,avrg_time)
    cpc = average_cpc_data(cpc,ti,avrg_time)
    gas = average_gas_phases_data(gas,ti,avrg_time)
    co = average_co_data(co,ti,avrg_time)
    core = average_core_data(core,ti,avrg_time)
    merge = pd.merge(ams,gas)
    merge2 = pd.merge(merge,co)
    merge3 = pd.merge(merge2,cpc)
    return pd.merge(merge3,core)


def load_complete_db():
    create_database_all_flights()
    db = pd.read_csv('complete_db',sep=' ').drop(['time','Unnamed: 0'],axis=1)
    db['time'] = db['t']
    db = db.drop('t',axis=1)
    return db

def create_database_all_flights():
    """
    input:  /
    purpose: same than create_database but with all the flights
    output: database 
    """
    final = pd.DataFrame()
    
    dates = [i*60*60*24 for i in range (17,37)]
    for i in range(17,37):
        if i not in [19,20,21,25,30,35]:
            print(i)
            current = create_database(i)
            current['t']=dates[i-17]+current['time']
            final = final.append(current)
    final.to_csv('complete_db',sep=' ')
    
def average_core_data(coree,ti,time): 
    """
    input:  core data, averaging time
    purpose: average time of gas data 
    output: averaged gas data
    """
    new_df = pd.DataFrame()
    columns = ['time','pressure','temperature','wind_dir','wind_speed','rapport_u/d_long','rapport_u/d_short']
    t=ti
    pressure = 0
    temp = 0
    wind_dir = 0
    wind_speed = 0 
    rapport_long = 0 
    rapport_short = 0
    c = 0
    
    for i in range(0,len(coree)):
        if coree['time'][i]<t+time :
            pressure += coree['pressure'][i]
            temp += coree['temperature'][i]
            wind_dir += coree['wind_dir'][i]
            wind_speed += coree['wind_dir'][i]
            rapport_long += coree['rapport_u/d_long'][i]
            rapport_short += coree['rapport_u/d_short'][i]
            c+=1
        else:
            if c != 0:
                current_dataframe = pd.DataFrame([[t,pressure/c,temp/c, wind_dir/c,
                                                   wind_speed/c,rapport_long/c,
                                                   rapport_short/c]],columns = columns)
                new_df = new_df.append(current_dataframe,ignore_index=True)

            pressure = 0
            temp =0
            wind_dir = 0
            wind_speed = 0 
            rapport_long = 0
            rapport_short = 0
           
            c = 0
            t += time
            
    return new_df

def average_gas_phases_data(gas,ti,time): 
    """
    input:  gas data, averaging time
    purpose: average time of gas data 
    output: averaged gas data
    """
    new_df = pd.DataFrame()
    columns = ['time','o3','nox','no','no2','so2']
    t=ti
    o3 = 0
    nox = 0
    no = 0
    no2 = 0
    so2 = 0
    c = 0
    for i in range(0,len(gas)):
        if gas['time'][i]<t+time :
            o3 += gas['O3'][i]
            nox += gas['NOx'][i]
            no2 += gas['NO2'][i]
            no += gas['NO'][i]   
            so2 += gas['SO2'][i]
            c+=1
        else:
            if c != 0:
                current_dataframe = pd.DataFrame([[t,o3/c,nox/c,no/c,no2/c,so2/c]],columns = columns)
                new_df = new_df.append(current_dataframe,ignore_index=True)
            o3 = 0
            nox = 0
            no = 0
            no2 = 0
            so2 = 0
            c = 0
            t += time
    return new_df

def average_cpc_data(cpcc,ti,time): 
    """
    input: cpc data, averaging time, ti
    purpose: average time of cpc data
    output: averaged cpc data
    """
    new_df = pd.DataFrame()
    columns = ['time','cpc']
    t=ti
    cpc = 0
    c = 0
    for i in range(0,len(cpcc)):
        if cpcc['time'][i]<t+time :
            cpc += cpcc['cpc'][i]
            c+=1
        else:
            if c != 0:
                current_dataframe = pd.DataFrame([[t,cpc/c]],columns = columns)
                new_df = new_df.append(current_dataframe,ignore_index=True)
            cpc = 0
            c = 0
            t += time
    return new_df

def average_co_data(coo,ti,time): 
    """
    input: cpc data, averaging time, ti
    purpose: average time of cpc data
    output: averaged cpc data
    """
    new_df = pd.DataFrame()
    columns = ['time','co']
    t=ti
    co = 0
    c = 0
    for i in range(0,len(coo)):
        if coo['time'][i]<t+time :
            co += coo['co'][i]
            c+=1
        else:
            if c != 0:
                current_dataframe = pd.DataFrame([[t,co/c]],columns = columns)
                new_df = new_df.append(current_dataframe,ignore_index=True)
            co = 0
            c = 0
            t += time
    return new_df
    
#def average_mz(t_init,initial_data,coumpound):
#    new_df2 = pd.DataFrame()
#    for i in range(166):
#        decompte = 300
#        time = t_init + i*60
#        cond1 = initial_data[coumpound].where(np.abs(time-initial_data['time'])<decompte)
#        mz = cond1.dropna()
#        while (len(mz)!=1):
#            decompte-=5
#            cond1=mz.where(np.abs(time-initial_data['time'])<decompte)
#            mz= cond1.dropna()
#        current_dataframe = pd.DataFrame([[time,float(mz.values)]],columns=['time',coumpound])
#        new_df2 = new_df2.append(current_dataframe,ignore_index=True)
#        print(i)
#    return new_df2
#
#def fill_empty(data,coumpound):
#    dataframe = pd.DataFrame()
#    for i in range(1,len(data)):
#        time = data['time'][i-1]
#        mz = data[coumpound][i-1]
#        current_dataframe = pd.DataFrame([[time,mz]],columns=['time',coumpound]) 
#        dataframe = dataframe.append(current_dataframe,ignore_index=True)
#        time = (data['time'][i-1] + data['time'][i])/2
#        mz = (data[coumpound][i-1] + data[coumpound][i])/2
#        current_dataframe = pd.DataFrame([[time,mz]],columns=['time',coumpound])
#        dataframe = dataframe.append(current_dataframe,ignore_index=True)   
#        
#    return dataframe