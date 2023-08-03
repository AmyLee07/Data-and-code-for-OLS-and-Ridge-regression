# Data-and-code-for-OLS-and-Ridge-regression
Data and code for OLS and Ridge regression

## Multivariate linear regression (MLR) models

The MLR setup has 77 terms, including 24 monthly linear trend terms and 24 intercept terms for the independent linear trends (ILT) before and after the turnaround year (1997) close to the timing of the peak stratospheric halogen loading, 24 QBO terms at 30 and 50 hPa, and 5 proxies for the 11-year solar cycle, El-Nino Southern Oscillation (ENSO), Arctic Oscillation (AO), Antarctic Oscillation (AAO) and Eliassen-Palm (EP) flux. The monthly mean ozone anomaly time series from 1984-2020 are obtained by referencing the monthly mean O~3~(t) to the climatological mean for each calendar month.

- QBO, ENSO, AO and AAO indices are from Climate Prediction Center (<https://www.cpc.ncep.noaa.gov/>)

- Mg II solar flux term (MgII) is obtained from [http://www.iup.uni-bremen.de/UVSAT/ Datasets/mgii](http://www.iup.uni-bremen.de/UUVSAT/Datasets/mgii)

- EP flux uses the 50 hPa vertical component (Fz50) with 2-month mean values (averaged over previous and current months) integrated over mid-latitudes between 45^○^ and 75^○^ in each hemisphere from the ECMWF ERA5 reanalysis.


* All data for the climate proxies and the stratospheic ozone data from SWOOSH, ML-TOMCAT and model simulation ERA5 are available in the directory ./SData/

* The code for obtaining the Xdata (climate proxies) for the MLR is available in ./Code/Xdata\_77terms.py

* The code for obtaining the Ydata (ozone anomalies) for the MLR is available in ./Code/Ydata\_8420\_sw\_ml\_ctrl.py

* The code for running the OLS regression with AR1 correction is available in ./Code/OLS\_AR1\_pw\_co\_gls.py

* The code for running the Ridge regression without AR1 correction is available in ./Code/Ridge\_noAR1.py

