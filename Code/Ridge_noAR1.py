# -*- coding: utf-8 -*-

#import netCDF4 as nc
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#from pandas import DataFrame
from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
model=LinearRegression()
from scipy import stats
import pickle
#from sklearn import linear_model
#import xarray as xr
import multiprocessing as mp
#print(mp.cpu_count())

from warnings import filterwarnings
filterwarnings('ignore')

import time
start_time=time.time()


def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()        

def read(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data    


def ridge_model(ilev):

    yr1=1984
    yr2=2020  #2018  #
    
    dfdates = pd.date_range('01/01/'+str(yr1), '12/01/'+str(yr2), freq = 'MS')
    xx1=len(np.arange(1984,1991))*12 
    dfdate0=dfdates

    x01=len(np.arange(1984,yr1))*12  
    x02=len(np.arange(1984,yr2+1))*12
    
    #PathIN='S:/Leeds/Python/OLSvRidge/SData/Pickle/'  
    PathIN='../SData/Pickle/' 
    ename='SWOOSH' # or'ERA5'  #'MLTOMCAT' #

    data1=read(PathIN+'Xdata_7_1984_2020_77Params_era51_72lats_fz50_nh_sh_new.out')
    Xdata=np.squeeze(data1[x01:x02,ilev,:,:])
    
    if ename=='SWOOSH':
        data2=read(PathIN+'Ydata_2_1984_2020_dO3_SWOOSH_v7.nc')
    elif ename=='MLTOMCAT':
        data2=read(PathIN+'Ydata_2_1984_2020_dO3_MLTOMCAT.nc')
    elif ename=='ERA5':   
        data2=read(PathIN+'Ydata_2_1984_2020_dO3_CONTROL.nc')
    Ydata=np.squeeze(data2[x01:x02,ilev,:])
    
    
    nlat=66  #72
    odata=np.zeros((nlat,77*3+1))
    #for ilev in range(0,10):
    for ilat in range(0,nlat):
        train_x0=Xdata[:,ilat,:]
        train_y0=Ydata[:,ilat]
        
        X_0=pd.DataFrame(train_x0,index=dfdates)
        Y_0=pd.DataFrame(train_y0,index=dfdates) 
        
        ### remove 1991-1992 data
        for ix in range(xx1,xx1+48):
            X_0.loc[dfdate0==dfdates[ix],:] = np.nan
            Y_0.loc[dfdate0==dfdates[ix],:] = np.nan
        ###  
        
        train_x=X_0.dropna()
        train_y=Y_0.dropna()

        #train_x = Xdata[:,ilat,:]
        #train_y = Ydata[:,ilat]
        alphas = np.logspace(-2, 2, 30)
        #reg = linear_model.RidgeCV(alphas=alphas,cv=10,normalize=True)
        reg = linear_model.RidgeCV(alphas=alphas,cv=None,normalize=True,store_cv_values=True) 
        
        reg.fit(train_x, train_y)
        alpha1 = reg.alpha_
        lm = linear_model.Ridge(alpha=alpha1,normalize=True)
        lm.fit(train_x, train_y)
        params2 = np.append(lm.intercept_,lm.coef_)
        predictions2 = lm.predict(train_x)
        #residuals=train_y-predictions
        
        newX0 = np.append(np.ones((len(train_x),1)), train_x, axis=1)
        newX=pd.DataFrame(newX0)
        
        MSE2 = (np.sum((train_y-predictions2)**2))/(len(newX0)-len(newX0[0])-1)
        var_b2 = float(MSE2)*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        sd_b2 = np.round(np.sqrt(var_b2),3)
        ts_b2 = params2/sd_b2
        p_values2 =np.round([2*(1-stats.t.cdf(np.abs(i),(len(newX0)-len(newX0[0])))) for i in ts_b2],3)
        #print(p_values2) 
        score=lm.score(train_x, train_y)
        
        '''
        ########### auto-corr ###########
        df = pd.DataFrame(residuals)
        df_shift = df.shift(1)
        corr_coeff_1 = df[0].corr(df_shift[0])      
        #print(corr_coeff_1)
        
        Y_00=pd.DataFrame(train_y)
        Y_00_2=Y_00-corr_coeff_1*df_shift
        y2 = Y_00_2.dropna()
        
        X_00=pd.DataFrame(train_x)
        X_00.loc['1984-01-01',:] = np.nan
        x2=X_00.dropna()
        
        reg.fit(x2, y2)
        alpha2 = reg.alpha_
        lm2 = linear_model.Ridge(alpha=alpha2,normalize=True)
        lm2.fit(x2, y2)
        params2 = np.append(lm2.intercept_,lm2.coef_)
        predictions2 = lm2.predict(x2)
        residuals2 = y2-predictions2
        df2 = residuals2
        df2_shift = df2.shift(1)
        corr_coeff_2 = df2[0].corr(df2_shift[0])       
        print(corr_coeff_2) 
        
        MSE2 = (np.sum((y2-predictions2)**2))/(len(newX0)-len(newX0[0])-1)
        var_b2 = float(MSE2)*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        sd_b2 = np.round(np.sqrt(var_b2),3)
        ts_b2 = params2/sd_b2
        p_values2 =np.round([2*(1-stats.t.cdf(np.abs(i),(len(newX0)-len(newX0[0])))) for i in ts_b2],3)
        #print(p_values2) 
        score=lm2.score(x2, y2)
        '''

        odata[ilat,0:77]=params2[1::]
        odata[ilat,77:77*2]=sd_b2[1::]
        odata[ilat,77*2:77*3]=p_values2[1::]
        odata[ilat,77*3]=score

    return odata

if __name__ == "__main__":


    alevs=np.arange(0,27,1)
    
    pool=mp.Pool(44)
    vdata=pool.map(ridge_model,[i for i in alevs])        
    pool.close()  
    
    print(time.time()-start_time) #s
    
    lindata=np.zeros((77*3+1,27,66))
    for j in range(len(alevs)):
        lindata[:,alevs[j],:]=np.transpose(vdata[j])
   
    
    yr1=1984
    yr2=2021
    #PathIN='S:/Leeds/Python/OLSvRidge/SData/Pickle/'  
    PathIN='../SData/Pickle/' 
    ename='SWOOSH' # or'ERA5'  #  'MLTOMCAT' #
    write(lindata,PathIN+'Ridge_7_'+ename+'_'+str(yr1)+'_'+str(yr2-1)+'_fz50_ERA51_2_new2_9194.pickle')  # No correction
    #write(lindata,PathIN+'Ridge_7_'+ename+'_'+str(yr1)+'_'+str(yr2-1)+'_fz50_ERA51_2_new2_9194_AR1.pickle') 
     






