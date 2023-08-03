# -*- coding: utf-8 -*-

####################### MLR factors ###########################

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
model=LinearRegression()
from scipy import stats
from pylab import genfromtxt  
import numpy.polynomial.polynomial as poly
import pickle

def smooth(y, box_pts):
    box=np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth

# de-trending 
def detrend(tdata):
    X = np.arange(0,len(tdata),1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,tdata)
    coefs= poly.polyfit(X,tdata, 1)
    ffit = poly.polyval(X,coefs)
    return tdata-ffit    

# normalising
def make_nterm(indata):
    q1 = np.ravel(indata[:])
    q2 = detrend(q1)
    q3 = (q2-np.min(q2))/(np.max(q2)-np.min(q2))
    return q3

# save Xdata with 77 terms
def write(data, outfile):
    f = open(outfile, "w+b")
    pickle.dump(data, f)
    f.close()    
        
# relatve path where climate proxies are
PathIN='../SData/'  

############################ QBO_hPa ########################
with open(PathIN+'newQBO_hPa.txt', 'r') as f: 
    f1 = f.read().splitlines()  
data1 = genfromtxt(PathIN+'newQBO_hPa.txt',skip_header=9)
iz1=np.where((data1[:,1]==7901.000)) 
iz2=np.where((data1[:,1]==2101.000))  
New_QBO30hPa=np.squeeze(data1[iz1[0][0]:iz2[0][0],5])  # 30hPa
New_QBO50hPa=np.squeeze(data1[iz1[0][0]:iz2[0][0],3])  # 50hPa
#New_QBO10hPa=np.squeeze(data1[iz1[0][0]:iz2[0][0],8])  # 10hPa
        
########################  solar #########################
with open(PathIN+'MgII_composite_2.dat', 'r') as f:
    f2 = f.read().splitlines()
data2 = genfromtxt(PathIN+'MgII_composite_2.dat',skip_header=23)
years=data2[:,0]
MgII=np.zeros((2021-1979)*12)
ik=-1
for iy in range(1979,2021):
    ik+=1
    iy1=np.where((years>=iy)&(years<iy+1))
    mms=data2[iy1,1] 
    dataMg=data2[iy1,3] 
    dataMg1=np.squeeze(dataMg)
    for im in range(12):
        im1=np.where(mms==im+1)
        dataMg_mm=np.mean(dataMg1[im1[1]])
        MgII[ik*12+im]=dataMg_mm
New_Solar0=MgII  
    
########################### ENSO ########################
with open(PathIN+'newENSO.txt', 'r') as f:
    f3 = f.read().splitlines()
data3 = genfromtxt(PathIN+'newENSO.txt',skip_header=1)
it1=np.where((data3[:,0]>=1979)&(data3[:,0]<2021))   
#New_ERSST0 = np.squeeze(data3[it1,8])   #nino3.4
New_ERSST0 = np.squeeze(data3[it1,9])   #anom
    
############################ AO ########################
with open(PathIN+'AO_2.txt', 'r') as f:
    f4 = f.read().splitlines()
data4 = genfromtxt(PathIN+'AO_2.txt',skip_header=0)
ia1=np.where((data4[:,0]>=1979)&(data4[:,0]<2021)) 
New_AO=np.squeeze(data4[ia1,2]) 

########################### AAO ########################
with open(PathIN+'AAO_2.txt', 'r') as f:
    f5 = f.read().splitlines()
data5 = genfromtxt(PathIN+'AAO_2.txt',skip_header=0)
idx1=np.where((data5[:,0]>=1979)&(data5[:,0]<2021)) 
New_AAO=np.squeeze(data5[idx1,2])  

############################ Fz50 ########################
ifile01=PathIN+'Fz_north.nc'  
vn01 = nc.Dataset(ifile01)
plev=vn01.variables["level"][:]
Fz_north=vn01.variables["Fz"][:]
ifile02=PathIN+'Fz_south.nc'  
vn02 = nc.Dataset(ifile02)
Fz_south=vn02.variables["Fz"][:]
New_Fz = Fz_north[:,14,0]+Fz_south[:,14,0]     #50 hPa (North+South)

################################################################

yr1=1984
yr2=2021

x1=len(np.arange(1979,yr1))*12  
x2=len(np.arange(1979,yr2))*12
x22=len(np.arange(1979,1998))*12
nn0=x2-x1
nn1=x22-x1
xx=np.arange(0,nn0)  


# detrended and normalised factors in 1984-2020
qbo30=make_nterm(New_QBO30hPa[x1:x2])
qbo50=make_nterm(New_QBO50hPa[x1:x2])
solar=make_nterm(New_Solar0[x1:x2])
enso=make_nterm(New_ERSST0[x1:x2])
ao=make_nterm(New_AO[x1:x2])
aao=make_nterm(New_AAO[x1:x2])            
# Fz50 with 2-month mean values (averaged over previous and current months)
fz=np.zeros(len(New_Fz))
fz[0]=New_Fz[0]
for ixx in range(1,len(New_Fz)):
    fz[ixx]=(New_Fz[ixx-1]+New_Fz[ixx])/2
fz_new1=make_nterm(fz[x1:x2])

# plotting
plt.figure()
plt.subplot(421)
plt.plot(qbo30,'',label='QBO 30hPa')
plt.legend(loc='right')
plt.subplot(422)
plt.plot(qbo50,'',label='QBO 50hPa')
plt.legend(loc='right')
plt.subplot(423)
plt.plot(solar,'r',label='Solar')
plt.legend(loc='right')
plt.subplot(424)
plt.plot(enso,'b-',label='ENSO')
plt.legend(loc='right')
plt.subplot(425)
plt.plot(smooth(ao[:],3),'m',label='AO')
plt.legend(loc='right')
plt.subplot(426)
plt.plot(smooth(aao[:],3),'c',label='AAO')
plt.legend(loc='right')
plt.tight_layout()
plt.subplot(427)
plt.plot(fz_new1,'b',label='Fz')
plt.legend(loc='right')
plt.tight_layout()
#

nlev=27
nlat=66
# save Xdata/proxy_data to *.out
Vdata = np.zeros((nn0,nlev,nlat,77))   
for ll in range(0,nlev):
    for l in range(0,nlat):
        for m in range(0,12):
            m1 = np.arange(m,nn1,12)            
            Vdata[m1,ll,l,m]=np.arange(0.5,1998-yr1+0.5,1)
            Vdata[m1,ll,l,m+12]=np.ones(1998-yr1)
            m2 = np.arange(m+nn1,nn0,12)            
            Vdata[m2,ll,l,m+24]=np.arange(0.5,yr2-1998+0.5,1)
            Vdata[m2,ll,l,m+36]=np.ones(yr2-1998)           
            mm = np.arange(m,nn0,12)
            Vdata[mm,ll,l,m+48]=qbo30[mm]
            Vdata[mm,ll,l,m+60]=qbo50[mm]
            
        Vdata[:,ll,l,72]=fz_new1[:]
        Vdata[:,ll,l,73]=solar[:]
        Vdata[:,ll,l,74]=enso[:]
        Vdata[:,ll,l,75]=ao[:]  
        Vdata[:,ll,l,76]=aao[:] 
        
proxy_data = Vdata
write(proxy_data, PathIN+'Xdata_7_'+str(yr1)+'_'+str(yr2-1)+'_77Params_era51_72lats_fz50_nh_sh_new.out')  
