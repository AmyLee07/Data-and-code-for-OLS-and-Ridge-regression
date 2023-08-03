# -*- coding: utf-8 -*-

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pickle

yr1=1984
yr2=2021
x1=len(np.arange(1979,yr1))*12  
x2=len(np.arange(1979,yr2))*12
nn0=x2-x1

# relative path where ozone data are
PathIN='../SData/'  

# swoosh #
ifile1=PathIN+'SWOOSH_1979_2021_72_V2.nc'    #
vn1= nc.Dataset(ifile1)
#print(vn1)
lat1 = vn1.variables["lat"][3:69]      #66   -81.25~81.25
plev1 = vn1.variables["plev"][10:37]   #27   146.78~1hPa
time1 = vn1.variables["time"][:]       #516
SW_O3 = vn1.variables["O3"][x1:x2,10:37,3:69,0]    

# simulation ERA5 #
ifile2=PathIN+'CONTROL_1979_2021_72_vmr_plev_V2.nc'
vn2= nc.Dataset(ifile2)
#print(vn2)
lat2 = vn2.variables["lat"][3:69]     
plev2 = vn2.variables["plev"][10:37]   
time2 = vn2.variables["time"][:]   
CTL_O3 = vn2.variables["O3"][x1:x2,10:37,3:69,0]   

# ML-TOMCAT #
ifile3=PathIN+'ML-TOMCAT_1979_2021_72_vmr_plev_V2.nc'  
vn3= nc.Dataset(ifile3)
#print(vn3)
lat3 = vn3.variables["lat"][3:69]    
plev3 = vn3.variables["plev"][10:37]   
time3 = vn3.variables["time"][:]   
ML_O3 = vn3.variables["O3"][x1:x2,10:37,3:69,0]  

nlev=len(plev1)  # 27
nlat=len(lat1)   # 66

####### ANOMALIES 1 % ###########
SW_O3_anom=np.zeros((nn0,nlev,nlat))  

for ilat in range(nlat):
    for ilev in range(nlev):    
        SW_O3_1=SW_O3[:,ilev,ilat]
        num=-1
        SW_O3_mm=np.zeros((int(nn0/12),12))
        for ik in range(0,nn0):
            if (ik+1)%12==0:
                num+=1
                for ikk in range(0,12):
                    SW_O3_mm[num,ikk]=SW_O3_1[ik+1-12+ikk]  
        
        SW_O3_12mm=np.mean(SW_O3_mm,0)
        A1=100*(SW_O3_mm-SW_O3_12mm)/SW_O3_12mm 
        SW_O3_anom[:,ilev,ilat]=A1.flatten()   # o3 anomalies % at XhPa 


####### ANOMALIES 2 % ###########
CTL_O3_anom=np.zeros((nn0,nlev,nlat))  

for ilat in range(nlat):
    for ilev in range(nlev):    
        CTL_O3_1=CTL_O3[:,ilev,ilat]
        num=-1
        CTL_O3_mm=np.zeros((int(nn0/12),12))
        for ik in range(0,nn0):
            if (ik+1)%12==0:
                num+=1
                for ikk in range(0,12):
                    CTL_O3_mm[num,ikk]=CTL_O3_1[ik+1-12+ikk]  
        
        CTL_O3_12mm=np.mean(CTL_O3_mm,0)
        A2=100*(CTL_O3_mm-CTL_O3_12mm)/CTL_O3_12mm 
        CTL_O3_anom[:,ilev,ilat]=A2.flatten()   # o3 anomalies % at XhPa 
        
        
####### ANOMALIES 3 % ###########
ML_O3_anom=np.zeros((nn0,nlev,nlat))  

for ilat in range(nlat):
    for ilev in range(nlev):    
        ML_O3_1=ML_O3[:,ilev,ilat]
        num=-1
        ML_O3_mm=np.zeros((int(nn0/12),12))
        for ik in range(0,nn0):
            if (ik+1)%12==0:
                num+=1
                for ikk in range(0,12):
                    ML_O3_mm[num,ikk]=ML_O3_1[ik+1-12+ikk]  
        
        ML_O3_12mm=np.mean(ML_O3_mm,0)
        A3=100*(ML_O3_mm-ML_O3_12mm)/ML_O3_12mm 
        ML_O3_anom[:,ilev,ilat]=A3.flatten()   # o3 anomalies % at XhPa   
        
# Plotting ozone anomalies at 100hPa, 31N       
plt.figure()
plt.plot(SW_O3_anom[:,2,45],label='SWOOSH v7')  
plt.plot(ML_O3_anom[:,2,45],label='ML-TOMCAT')
plt.plot(CTL_O3_anom[:,2,45],label='ERA5 TOMCAT')
plt.legend()
plt.ylabel('anom [ppmv]')
plt.title('100hPa, 31N')
MM=np.arange(0,nn0,48)
YY=np.arange(yr1,yr2+1,4)  
plt.xticks(MM,YY)

# save Ydata (o3 anom) to *.nc
def write(data, outfile):
    f = open(outfile, "w+b")
    pickle.dump(data, f)
    f.close()        

write(SW_O3_anom, PathIN+'Ydata_2_'+str(yr1)+'_'+str(yr2-1)+'_dO3_SWOOSH_v7.nc')
write(CTL_O3_anom, PathIN+'Ydata_2_'+str(yr1)+'_'+str(yr2-1)+'_dO3_CONTROL.nc')
write(ML_O3_anom, PathIN+'Ydata_2_'+str(yr1)+'_'+str(yr2-1)+'_dO3_MLTOMCAT.nc')
