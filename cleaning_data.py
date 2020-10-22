#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:55:07 2019

@author: lheurvin
"""

import numpy as np
import pandas as pd

def core():   
    """
    input : None
    purpose : cleaning core data in a txt file, for every flight
    output: None
    """
    columns = ['time','latitude','longitude','altitude','platform_roll_angle',
               'platform_pitch_angle','platform_orientation','pressure','temperature',
               'relative humidity','dew_point_temperature', 'abs_humidity',
               'wind_dir','wind_speed','upward_air_velocity','chm_j_no2h_val_1',
               'chm_j_no2b_val_1','up_long','down_long','up_short','down_short','platform_speed_wrt_air']
    
    path = 'data/core/flight_date.csv'
    for i in range(17,37):
        flight_date = pd.read_csv(path)
        flight_date.insert(0,'number',[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        data_core = pd.read_csv('data/'+flight_date.filename[i-17],skiprows=44, sep="	 ", engine='python',
                                  names = columns)
        data_core = data_core[['time','latitude','longitude','altitude','temperature','pressure','wind_dir','wind_speed','up_long','down_long','up_short','down_short']]
        data_core=data_core.query('temperature < 100')  
        data_core=data_core.query('down_long < 1000') 
        data_core=data_core.query('altitude > 30') 
        data_core['rapport_u/d_long'] = data_core['up_long']/data_core['down_long']
        data_core['rapport_u/d_short'] = data_core['up_short']/data_core['down_short']        
        data_core['y'] = np.sin(data_core['wind_dir']*2*np.pi/360)
        data_core['x'] = np.cos(data_core['wind_dir']*2*np.pi/360)
        data_core.to_csv('data/cleaned_data/'+str(i)+'/core.txt', sep=' ', index=False)



def clean_gas_phases():
    """
    input : None
    purpose : cleaning gas phases data in a txt file, for every flight
    output: None
    """
    columns = ['time','latitude','longitude','altitude','air_pressure','air_temperature','O3',
       'NOx','NO','NO2','SO2']
    path = 'data/gas_phases/flight_date.csv'
    for i in range(17,37):
        if i!=25: #missing data for flight 25 
            flight_date = pd.read_csv(path)
            flight_date.insert(0,'number',[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
            data_o3noxnono2so2 = pd.read_csv('data/'+flight_date.filename[i-17],skiprows=33, sep="	 ", engine='python',
                                      names = columns,dtype=float)
            data_o3noxnono2so2.dropna(inplace=True)
            data_o3noxnono2so2=data_o3noxnono2so2.query('SO2 > -900')   
            data_o3noxnono2so2=data_o3noxnono2so2.query('latitude < 360 | longitude < 360')   
            for col in ['O3','NOx','NO','NO2','SO2']:
                data_o3noxnono2so2[col] = data_o3noxnono2so2[col].apply(lambda x : x if x>=0 else 0)
            data_o3noxnono2so2.to_csv('data/cleaned_data/'+str(i)+'/gas_phases.txt', sep=' ', index=False)


def clean_ch4co2co():
    """
    input : None
    purpose : cleaning ch4 ch2 co data in a txt file, for every flight
    output: None
    """
    path = 'data/coco2ch4/flight_date.csv'
    for i in range(17,37):
        if i!=35: #missing data for flight 35  
            flight_date = pd.read_csv(path,sep=' ')
            flight_date.insert(0,'number',[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
            
            data_ch4 = pd.read_csv('data/'+flight_date.filename_ch4[i-17],sep=";").drop([0])
            data_ch4.insert(3, 'time',data_ch4.Hour*3600 + data_ch4.Minute*60 + data_ch4.Second)
            data_ch4=data_ch4.query('ch4 > -500') 
            data_ch4 = data_ch4.drop(["SamplingHeight","Year","Month","Day","Hour","Minute","Second","DecimalDate"],axis=1)
            data_ch4.to_csv('data/cleaned_data/'+str(i)+'/ch4.txt', sep=' ', index=False)
            
            data_co2 = pd.read_csv('data/'+flight_date.filename_co2[i-17],sep=";").drop([0,1])
            data_co2.insert(3, 'time',data_co2.Hour*3600 + data_co2.Minute*60 + data_co2.Second)
            data_co2=data_co2.query('co2 > -500') 
            data_co2 = data_co2.drop(["SamplingHeight","Year","Month","Day","Hour","Minute","Second","DecimalDate"],axis=1)
            data_co2.to_csv('data/cleaned_data/'+str(i)+'/c02.txt', sep=' ', index=False)
            
            
            data_co = pd.read_csv('data/'+flight_date.filename_co[i-17],sep=";").drop([0,1])
            data_co.insert(3, 'time',data_co.Hour*3600 + data_co.Minute*60 + data_co.Second)
            data_co=data_co.query('co > -500')  
            data_co = data_co.drop(["SamplingHeight","Year","Month","Day","Hour","Minute","Second","DecimalDate"],axis=1)
            data_co.to_csv('data/cleaned_data/'+str(i)+'/co.txt', sep=' ', index=False)

def clean_cpc():
    """
    input : None
    purpose : cleaning cpc data in a txt file, for every flight
    output: None
    """
    path = 'data/cpc/flight_date.csv'
    for i in range(17,37):
        flight_date = pd.read_csv(path,sep=',')
        flight_date.insert(0,'number',[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        data_cpc = pd.read_csv('data/'+flight_date.filename[i-17],names = ['time','latitude',
                               'longitude','altitude','cpc','air_temperature'],skiprows=26, sep = ' ',index_col=False)
        data_cpc = data_cpc.query('cpc > -90 & cpc != 0')
        for col in ['cpc']:
                data_cpc[col] = data_cpc[col].apply(lambda x : x if x>=0 else 0)
        data_cpc.to_csv('data/cleaned_data/'+str(i)+'/cpc.txt', sep=' ', index=False)
    
def clean_voc():
    """
    input : None
    purpose : cleaning voc data in a txt file, for every flight
    output: None
    Lvoc = number pof the flights for which with have the data
    """
    Lvoc = [17,18,19,21,22,23,24,25,26,27,31,32,34,35,36]    
    path = 'data/voc/flight_date.csv'
    for i in Lvoc:
        flight_date = pd.read_csv(path,sep=',')
        columns_voc = ['time','latitude','longitude','altitude','mz33','mz42','mz45',
                       'mz59','mz63','mz69','mz71','mz73','mz79','mz93','mz107','mz137',
                       'mz121','mze33','mze42','mze45','mze59','mze63','mze69','mze71',
                       'mze73','mze79','mze93','mze107','mze121','mze137']    
        data_voc1 = pd.read_csv('data/'+flight_date.filename[i-17],names = columns_voc
                               ,skiprows=50, sep = "  | "  ,index_col=False,dtype=float)
        data_voc1 = data_voc1.query('mz69>-90')
        data_voc1 = data_voc1.query('latitude>-90')
        data_voc1 = data_voc1[['time','mz69','mze69']]
        data_voc1.to_csv('data/cleaned_data/'+str(i)+'/voc_mz69.txt', sep=' ', index=False)
        
        data_voc2 = pd.read_csv('data/'+flight_date.filename[i-17],names = columns_voc
                               ,skiprows=50, sep = "  | "  ,index_col=False,dtype=float)
        data_voc2 = data_voc2.query('mz137>-90 ')
        data_voc2 = data_voc2.query('latitude>-90')
        data_voc2 = data_voc2[['time','mz137','mze137']]
        data_voc2.to_csv('data/cleaned_data/'+str(i)+'/voc_mz137.txt', sep=' ', index=False)
        
def clean_ams():
    """
    input : None
    purpose : cleaning ams data in a txt file, for every flight
    output: None
    """
    path = 'data/ams/flight_date.csv'
    for i in range(17,37):
        flight_date = pd.read_csv(path,sep=',')
        flight_date.insert(0,'number',[17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        data_ams = pd.read_csv('data/'+flight_date.filename[i-17],names = ['time','latitude',
                               'longitude','altitude','org','so4','no3','nh4','chl'],skiprows=29, sep = ' ',index_col=False)
        data_ams = data_ams.query('so4 > -90')
        for col in ['org','so4','no3','nh4','chl']:
                data_ams[col] = data_ams[col].apply(lambda x : x if x>=0 else 0)
        data_ams.to_csv('data/cleaned_data/'+str(i)+'/ams.txt', sep=' ', index=False)

clean_ams()
#clean_ch4co2co()
#clean_cpc
#clean_voc()
#clean_gas_phases()
