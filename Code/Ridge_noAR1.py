# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
model=LinearRegression()
from scipy import stats
import pickle
import multiprocessing as mp
#print(mp.cpu_count())
from warnings import filterwarnings
filterwarnings('ignore')

import time
start_time=time.time()

# read Xdata and Ydata
def read(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data    

# save Ridge regression results
def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()        

def ridge_model(ilev):
    yr1=1984
    yr2=2020  
    x01=len(np.arange(1984,yr1))*12  
    x02=len(np.arange(1984,yr2+1))*12
    xx1=len(np.arange(1984,1991))*12 

    dfdates = pd.date_range('01/01/'+str(yr1), '12/01/'+str(yr2), freq = 'MS')
    dfdate0=dfdates

    # relative path where the Xdata and Ydata are
    PathIN='../SData/' 
    ename='SWOOSH' # 'ERA5'  #'MLTOMCAT' #

    # read Xdata
    data1=read(PathIN+'Xdata_7_1984_2020_77Params_era51_72lats_fz50_nh_sh_new.out')
    Xdata=np.squeeze(data1[x01:x02,ilev,:,:])

    # read Ydata
    if ename=='SWOOSH':
        data2=read(PathIN+'Ydata_2_1984_2020_dO3_SWOOSH_v7.nc')
    elif ename=='MLTOMCAT':
        data2=read(PathIN+'Ydata_2_1984_2020_dO3_MLTOMCAT.nc')
    elif ename=='ERA5':   
        data2=read(PathIN+'Ydata_2_1984_2020_dO3_CONTROL.nc')
    Ydata=np.squeeze(data2[x01:x02,ilev,:])
       
    nlat=66        
    odata=np.zeros((nlat,77*3+1))
    #77*3+1 including 77 coefficients, 77 standard errors, 1 goodness of fit(R2)

    for ilat in range(0,nlat):
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

        # Ridge regression with built-in cross-validation.  
        alphas = np.logspace(-2, 2, 30)
        reg = linear_model.RidgeCV(alphas=alphas,cv=None,normalize=True,store_cv_values=True) 
            
        # Fit Ridge regression model with cv    
        reg.fit(train_x, train_y)
        alpha1 = reg.alpha_
            
        # Linear least squares with L2 regularization.
        lm = linear_model.Ridge(alpha=alpha1,normalize=True)
        lm.fit(train_x, train_y)

        # Get parameters for Ridge coefficients
        params2 = np.append(lm.intercept_,lm.coef_)
        predictions2 = lm.predict(train_x)
        residuals=train_y-predictions2
            
        # Calculate standard errors and significane with t-test p-values
        newX0 = np.append(np.ones((len(train_x),1)), train_x, axis=1)
        newX=pd.DataFrame(newX0)   
        MSE2 = (np.sum((residuals)**2))/(len(newX0)-len(newX0[0])-1)
        var_b2 = float(MSE2)*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        sd_b2 = np.round(np.sqrt(var_b2),3)
        ts_b2 = params2/sd_b2
        p_values2 =np.round([2*(1-stats.t.cdf(np.abs(i),(len(newX0)-len(newX0[0])))) for i in ts_b2],3)
        #print(p_values2) 
        score=lm.score(train_x, train_y)

        odata[ilat,0:77]=params2[1::]  # coefficients for 77 terms
        odata[ilat,77:77*2]=sd_b2[1::] # standard errors 
        odata[ilat,77*2:77*3]=p_values2[1::]  # p-values
        odata[ilat,77*3]=score  # goodness of fit (R2)

    return odata

if __name__ == "__main__":
    
    # use multiprocessing for Ridge regression 
    pool=mp.Pool(44)
    alevs=np.arange(0,27,1)    
    vdata=pool.map(ridge_model,[i for i in alevs])        
    pool.close()     
    print(time.time()-start_time)         
    # 
    lindata=np.zeros((77*3+1,27,66))
    for j in range(len(alevs)):
        lindata[:,alevs[j],:]=np.transpose(vdata[j])
   
    PathO='../SData/Pickle/' 
    ename='SWOOSH' # 'ERA5'  #  'MLTOMCAT' #
    write(lindata,PathO+'Ridge_7_'+ename+'_1984_2020_fz50_ERA51_2_new2_9194.pickle')  
     
