import numpy as np
import matplotlib.pyplot as plt
from Brazil_Data_Covid import daily_deaths_Brazil_RA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

# Day 0 = 02/25/20
a = date_format
b = date_format[:250]
c = date_format[250:273]
x = np.int_(np.where(Data.iloc[Brazil,3]))[0]
y = np.array(daily_deaths_Brazil_RA)

# Train/Test: Split Train: Feb-October; Test: November 
xtrain = np.int_(np.where(Data.iloc[Brazil,3]))[0][:250]
xtest = np.int_(np.where(Data.iloc[Brazil,3]))[0][250:273]
ytrain = np.array(daily_deaths_Brazil_RA)[:250]
ytest = np.array(daily_deaths_Brazil_RA)[250:273]

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
SVRfit_Brazil_DD = pd.DataFrame({'Dates': c, 'Deaths Prediction':y_pred, 'Deaths':ytest})
SVR_November_DD = SVRfit_Brazil_DD.to_excel("BrazilSupportVectorRegressionNovemberDeaths.xlsx", sheet_name='NovemberDailyDeathsPrediction')

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
plt.title('Daily Deaths In Brazil: SVR')
plt.ylabel('Daily Deaths')
plt.xlabel('Date')
plt.legend()
plt.show()

#Decemeber Predictions
future_days = np.arange(273,311,1)
future_days = future_days.reshape(-1,1)
future_pred = []
for i in future_days:
    m = reg.predict(sc_x.transform(np.array([i])))
    future = np.around(sc_y.inverse_transform(m))
    future_pred.append(future)

future_days = np.arange(1,np.size(future_days)+1,1) 

# Excel File    
SVR_BrazilDD_pred = pd.DataFrame({'Days since 11/23':future_days , 'Deaths Prediction':future_pred})
SVR_December_DD = SVR_BrazilDD_pred.to_excel("BrazilSupportVectorRegressionDecemberDeaths.xlsx", sheet_name='DecemberDailyDeathsPrediction')

print('------------SVR EVALUATION-------------')
print('RMSE Score: '+ np.str(rmse))
print('RMSLE Score: '+ np.str(rmsle))
print('R2 Score: '+ np.str(R2))
print(SVRfit_Brazil_DD)
print(SVR_BrazilDD_pred)
