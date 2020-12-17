import numpy as np
import matplotlib.pyplot as plt
from India_Data_COVID import daily_deaths_India_RA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

# Day 0 = 01/30/20
a = date_format
b = date_format[:276]
c = date_format[276:299]
x = np.int_(np.where(Data.iloc[India,3]))[0]
y = np.array(daily_deaths_India_RA)

# Train/Test: Split Train: Feb-October; Test: November 
xtrain = np.int_(np.where(Data.iloc[India,3]))[0][:276]
xtest = np.int_(np.where(Data.iloc[India,3]))[0][276:299]
ytrain = np.array(daily_deaths_India_RA)[:276]
ytest = np.array(daily_deaths_India_RA)[276:299]

# Reshape Data
x1 = xtrain.reshape(-1,1)
x1test = xtest.reshape(-1,1)
y1 = ytrain.reshape(-1,1)
y1test = ytest.reshape(-1,1)

# Scale Data
sc_x = StandardScaler()
sc_y = StandardScaler()
x1train = sc_x.fit_transform(x1)
x1test = sc_x.fit_transform(x1test)
y1train = sc_y.fit_transform(y1)
y1test = sc_y.fit_transform(y1test)

# Model SVR with rbf kernel
reg = SVR(kernel = 'rbf')
reg.fit(x1train,y1train)
y_pred = reg.predict(x1test)
y_pred = np.around(sc_y.inverse_transform(y_pred))

# Predicitons in Excel File
c = np.array(c)
y_pred = y_pred.tolist()
ytest = np.around(ytest)
ytest = ytest.tolist()
SVRfit_India_DD = pd.DataFrame({'Dates': c, 'Deaths Prediction':y_pred, 'Deaths':ytest})
SVR_November_DD = SVRfit_India_DD.to_excel("IndiaSupportVectorRegressionNovemberDeaths.xlsx", sheet_name='NovemberDailyDeathsPrediction')

# Evaluate
R2 = r2_score(ytest, y_pred)
mse = mean_squared_error(ytest, y_pred)
rmse = np.sqrt(mse)
rmsle = np.sqrt(mean_squared_log_error(ytest,y_pred))

# Visualize
plt.figure(figsize=(12,5))
plt.grid()
plt.scatter(a, y, color = 'b')
plt.plot(c, y_pred,'r--',label = 'SVR')
plt.title('Daily Deaths In India: SVR')
plt.ylabel('Daily Deaths')
plt.xlabel('Date')
plt.legend()
plt.show()

#Predictions Next 58 days
future_days = np.arange(299,337,1)
future_days = future_days.reshape(-1,1)
future_pred = []
for i in future_days:
    m = reg.predict(sc_x.transform(np.array([i])))
    future = np.around(sc_y.inverse_transform(m))
    future_pred.append(future)

future_days = np.arange(1,np.size(future_days)+1,1) 

# Excel File    
SVR_IndiaDD_pred = pd.DataFrame({'Days since 11/23':future_days , 'Deaths Prediction':future_pred})
SVR_December_DD = SVR_IndiaDD_pred.to_excel("IndiaSupportVectorRegressionDecemberDeaths.xlsx", sheet_name='DecemberDailyDeathsPrediction')

print('------------SVR EVALUATION-------------')
print('RMSE Score: '+ np.str(rmse))
print('RMSLE Score: '+ np.str(rmsle))
print('R2 Score: '+ np.str(R2))
print(SVRfit_India_DD)
print(SVR_IndiaDD_pred)
