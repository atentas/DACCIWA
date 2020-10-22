Made by Vincent LHEUREUX ; 19.08.19 - 20.12.19 
vincent.lheureux@hotmail.fr 1
vincent.lheureux@ensta-bretagne.org 2



The original files that Jonathan gave me are in 'autres/Stage Helsinki' including xRay data, APiTof data, flight summaries


--------------------------------------------
important .py files : 

cleaning_data.py
averaging.py 
functions.py
applyML_algorithms.py 
ML_applied.py 

BASEMAP has to be add in Your python package ; i used it for plotting the map ; there is a lisence i think   
peak list gave by Heikki is in the file 

xray.py
--------------------------------------------

cleaning_data.py

I get the data from : "http://baobab.sedoo.fr/DACCIWA/Plateform-search/" 
I downloaded and put them in the folder data. 

Then I cleaned every data : by cleaning i mean deleting nonsence or error value : 
	for ex : data_ch4=data_ch4.query('ch4 > -500') 
I did this for all flights and save data/cleaned_data/ for each flight. 

- - - - - - - - - - - - - - - - - - - - - - - -

averaging.py 

The objective is to average datapoints with a given time : 
we will often chose a 10 seconds averaging
If data are already 40 seconds averaged (AMS data), the final database will have a points all the 40 seconds at the end, but for the gas data for example, it will still be a 10 seconds average : 
Indeed we average data file per file, and then we merge the created DataFrame

- - - - - - - - - - - - - - - - - - - - - - - -

functions.py 

The functions inside are Kmeans, Zscore, PCA and correlation_Matrix 
They can be used by every other .py file

- - - - - - - - - - - - - - - - - - - - - - - -

applyML_algorithms.py 

Here you will have all the functions used to get the results
They are all commented

It's 'ML_applied.py' which excute these functions in the logical order

- - - - - - - - - - - - - - - - - - - - - - - -

ML_applied.py 
Apply the steps of the ML process ; all commented

- - - - - - - - - - - - - - - - - - - - - - - -

x_ray.py 

I did this in order to get only APiTOF data with TofTools ; So the input is a file that J. Dupplissy gave me : it's txt file to say when is the xray activated or not for each flight
I created .dat files that TofTools wants in order to generate the mask, with the time format of MATLAB. 
So it creates .dat file for every flight  
