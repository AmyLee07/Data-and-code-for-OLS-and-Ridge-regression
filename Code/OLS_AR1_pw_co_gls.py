# -*- coding: utf-8 -*-

#import netCDF4 as nc
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#from pandas import DataFrame
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
model=LinearRegression()
import statsmodels.api as sm

#from scipy import stats
import pickle
#from sklearn import linear_model
#import xarray as xr
#import multiprocessing as mp
#print(mp.cpu_count())

from warnings import filterwarnings
filterwarnings('ignore')

import time
start_time=time.time()


def read(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data    

def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()   


# cochrane-orcutt / prais-winsten with given AR(1) rho, 
# derived from ols model, default to cochrane-orcutt 
def ols_ar1(model,rho,drop1=True):
    x = model.model.exog
    y = model.model.endog
    ystar = y[1:]-rho*y[:-1]
    xstar = x[1:,]-rho*x[:-1,]
    if drop1 == False:
        ystar = np.append(np.sqrt(1-rho**2)*y[0],ystar)
        xstar = np.append([np.sqrt(1-rho**2)*x[0,]],xstar,axis=0)
    model_ar1 = sm.OLS(ystar,xstar).fit()
    return(model_ar1)

# cochrane-orcutt / prais-winsten iterative procedure
# default to cochrane-orcutt (drop1=True)
def OLSAR1(model,drop1=True):
    x = model.model.exog
    y = model.model.endog
    e = y - (x @ model.params)
    e1 = e[:-1] #; e0 = e[1:]
    rho0 = np.dot(e1,e[1:])/np.dot(e1,e1)
    rdiff = 1.0
    while(rdiff>1.0e-5):
        model1 = ols_ar1(model,rho0,drop1)
        #x = model1.model.exog
        #y = model1.model.endog
        e = y - (x @ model1.params)
        e1 = e[:-1] #; e0 = e[1:]
        rho1 = np.dot(e1,e[1:])/np.dot(e1,e1)
        rdiff = np.sqrt((rho1-rho0)**2)
        rho0 = rho1
        #print('Rho = ', rho0)
    # pint final iteration
    # print(sm.OLS(e0,e1).fit().summary())
    model1 = ols_ar1(model,rho0,drop1)
    return(model1)



yr1=1984
yr2=2021

x01=len(np.arange(1984,yr1))*12  
x02=len(np.arange(1984,yr2))*12
nn0=x02-x01
xx1=len(np.arange(1984,1991))*12 

dfdates = pd.date_range('01/01/'+str(yr1), '12/01/'+str(yr2-1), freq = 'MS')
dfdate0 = dfdates

lindata0=np.zeros((77*3+1,27,66))  
lindata1=np.zeros((77*3+1,27,66))  
lindata2=np.zeros((77*3+1,27,66))  


corr_1=np.zeros((27,66)) 
corr_2=np.zeros((27,66)) 

#PathIN='S:/Leeds/Python/OLSvRidge/SData/'  
PathIN='../SData/' 
ename='MLTOMCAT' #'SWOOSH' # or'ERA5' #

for ilev in range(27):
    
    data1=read(PathIN+'Xdata_7_1984_2020_77Params_era51_72lats_fz50_nh_sh_new.out')
    Xdata=np.squeeze(data1[x01:x02,ilev,:,:])
    
    if ename=='SWOOSH':
        data2=read(PathIN+'Ydata_2_1984_2020_dO3_SWOOSH_v7.nc')
    elif ename=='MLTOMCAT':
        data2=read(PathIN+'Ydata_2_1984_2020_dO3_MLTOMCAT.nc')
    elif ename=='ERA5':   
        data2=read(PathIN+'Ydata_2_1984_2020_dO3_CONTROL.nc')
    Ydata=np.squeeze(data2[x01:x02,ilev,:])
    
    for ilat in range(66):  
        train_x0=Xdata[:,ilat,:]
        train_y0=Ydata[:,ilat]        
        X_0=pd.DataFrame(train_x0,index=dfdates)
        Y_0=pd.DataFrame(train_y0,index=dfdates) 
        
        ### remove 1991-1994 data
        for ix in range(xx1,xx1+48):
            X_0.loc[dfdate0==dfdates[ix],:] = np.nan
            Y_0.loc[dfdate0==dfdates[ix],:] = np.nan
        ###  
        
        train_x=X_0.dropna()
        train_y=Y_0.dropna()

        # method 1 # (NO correction)
        X1 = sm.add_constant(train_x)
        model_ols = sm.OLS(train_y, X1).fit()

       
        residuals =model_ols.resid
        #corr_coeff_2 = pd.Series(residuals2.autocorr(lag=1))
        #print(corr_coeff_2[0])    
        #corr_2[ilev,ilat]=corr_coeff_2[0]
        # or:
        df1 = pd.DataFrame(residuals)
        df1_shift = df1.shift(1)
        corr_1[ilev,ilat]= df1[0].corr(df1_shift[0]) 
        

        # AR(1) based on cochrane-orcutt iterative procedure   
        ar1_co = OLSAR1(model_ols)
        # ar1_co = OLSAR1(model_ols,drop1=True)
        #print(ar1_co.summary())
        
        # AR(1) based on prais-winsten iterative procedure
        ar1_pw = OLSAR1(model_ols,drop1=False)
        #print(ar1_pw.summary())
        # the results are based on transformed model
        
        # alternatively, using statsmodels' GLSAR to estimate AR(!)
        # results may not be the same as OLSAR, need to check
        ar1_gls = sm.GLSAR(train_y, X1,1)
        results = ar1_gls.iterative_fit(maxiter=50)
        print ('Iterations used = %d Converged %s' % (results.iter, results.converged) )
        #print ('Rho =  ', ar1_gls.rho)
        #print(results.summary())



        model0=ar1_co
        model1=ar1_pw
        model2=results      
        
        lindata0[0:77,ilev,ilat]=model0.params[1::] #lm2.coef_#    
        lindata0[77:77*2,ilev,ilat]=model0.bse[1::] #sd_b2[1::]  #
        lindata0[77*2:77*3,ilev,ilat]=model0.pvalues[1::] #p_values2[1::]  #
        lindata0[77*3,ilev,ilat]=model0.rsquared #lm2.score(x2, y2)  #
        
        lindata1[0:77,ilev,ilat]=model1.params[1::] #lm2.coef_#    
        lindata1[77:77*2,ilev,ilat]=model1.bse[1::] #sd_b2[1::]  #
        lindata1[77*2:77*3,ilev,ilat]=model1.pvalues[1::] #p_values2[1::]  #
        lindata1[77*3,ilev,ilat]=model1.rsquared #lm2.score(x2, y2)  #
        
        lindata2[0:77,ilev,ilat]=model2.params[1::] #lm2.coef_#    
        lindata2[77:77*2,ilev,ilat]=model2.bse[1::] #sd_b2[1::]  #
        lindata2[77*2:77*3,ilev,ilat]=model2.pvalues[1::] #p_values2[1::]  #
        lindata2[77*3,ilev,ilat]=model2.rsquared #lm2.score(x2, y2)  #
        corr_1[ilev,ilat]= ar1_gls.rho[0]

        ###########Check again for serial correlation in the residuals ###########
        
        residuals2 = model0.resid
        #corr_coeff_2 = pd.Series(residuals2.autocorr(lag=1))
        #print(corr_coeff_2[0])    
        #corr_2[ilev,ilat]=corr_coeff_2[0]
        # or:
        df2 = pd.DataFrame(residuals2)
        df2_shift = df2.shift(1)
        corr_2[ilev,ilat]= df2[0].corr(df2_shift[0]) 
        

'''
write(lindata0,PathIN+'Pickle/OLS_7_'+ename+'_'+str(yr1)+'_'+str(yr2-1)+'_fz50_ERA51_2_new2_9194_ar1_co.pickle') #cochrane-orcutt with given AR(1) rho
write(lindata1,PathIN+'Pickle/OLS_7_'+ename+'_'+str(yr1)+'_'+str(yr2-1)+'_fz50_ERA51_2_new2_9194_ar1_pw.pickle') # prais-winsten with given AR(1) rho
write(lindata2,PathIN+'Pickle/OLS_7_'+ename+'_'+str(yr1)+'_'+str(yr2-1)+'_fz50_ERA51_2_new2_9194_ar1_gls.pickle') #using statsmodels' GLSAR to estimate AR(!)
'''
