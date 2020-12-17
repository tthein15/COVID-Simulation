import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_squared_log_error, explained_variance_score
from India_Data_COVID import daily_deaths_India_RA 
#from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler

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
#sc_x = StandardScaler()
#x1 = sc_x.fit_transform(x)
#sc_y = StandardScaler()
#y1 = sc_y.fit_transform(y)

# Linear Model
lin_reg = LinearRegression()
lin_reg.fit(x1,y1)

# Get Predictions Dates:(11/01-11/23)
y_pred = np.around(lin_reg.predict(x1test))
#y_pred = sc_y.inverse_transform(y_pred)

#Visualize
plt.figure(figsize=(12,5))
plt.grid()
plt.scatter(a, y, color = 'blue')
plt.plot(c, y_pred, 'r--')
plt.title('India Daily Deaths: Linear Regression')
plt.xlabel('Months')
plt.ylabel('Deaths')
#plt.show()


#Coefficients
coef = lin_reg.coef_
intercept = lin_reg.intercept_
lin_eq = np.str(coef) + "x + " + np.str(intercept)

# Evaluate
mse = mean_squared_error(y1test, y_pred)
rmse = np.sqrt(mse)
R2 = explained_variance_score(y1test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y1test,y_pred))

# Regression Data Table
c = np.array(c)
y_pred = y_pred.tolist()
y1test = (np.around(y1test)).tolist()
LRfit_India_DD = pd.DataFrame({'Dates': c, 'Deaths Prediction':y_pred, 'Deaths':y1test})
LR_India_DD = LRfit_India_DD.to_excel("IndiaLinearRegressionNovemberDeaths.xlsx", sheet_name='NovemberDailyDeathsPrediction')

# Predicting Daily Case Count from (11/23-12/31)
future_pred = []
future_days = np.arange(299,337,1)
for i in future_days:
    future = np.around(lin_reg.predict(np.array([[i]])))
    future_pred.append(future)

#Prediction Data Table    
future_days = np.arange(1,np.size(future_days)+1,1)    
LR_IndiaDD_pred = pd.DataFrame({'Days since 11/23':future_days , 'Deaths Prediction':future_pred})
LR_India_DD = LR_IndiaDD_pred.to_excel("IndiaLinearRegressionDecemberDeaths.xlsx", sheet_name='DecemberDailyDeathsPrediction')

# Evaluation Table
LR_India_Evaluation_DD = pd.DataFrame({'Model':lin_eq,'RMSE':[rmse], 'RMSLE':[rmsle], 'R2': [R2]})


print('Linear Regression Equation: ' + lin_eq)
print('RMSE Score: '+ np.str(rmse))
print('RMSLE Score: '+ np.str(rmsle))
print('R2 Score: '+ np.str(R2))
print(LRfit_India_DD)
print(LR_IndiaDD_pred)


  