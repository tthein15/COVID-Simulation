import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from Brazil_Data_Covid import daily_cases_Brazil_RA 
import pandas as pd
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler

# Day 0 = 02/25/20
a = date_format
b = date_format[:250]
c = date_format[250:273]
x = np.int_(np.where(Data.iloc[Brazil,3]))[0]
y = np.array(daily_cases_Brazil_RA)

# Train/Test: Split Train: Feb-October; Test: November 
xtrain = np.int_(np.where(Data.iloc[Brazil,3]))[0][:250]
xtest = np.int_(np.where(Data.iloc[Brazil,3]))[0][250:273]
ytrain = np.array(daily_cases_Brazil_RA)[:250]
ytest = np.array(daily_cases_Brazil_RA)[250:273]

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
plt.title('Brazil Daily Cases: Linear Regression')
plt.xlabel('Days')
plt.ylabel('Cases')
#plt.show()


#Coefficients
coef = lin_reg.coef_
intercept = lin_reg.intercept_
lin_eq = np.str(coef) + "x + " + np.str(intercept)

# Evaluate
R2 = r2_score(y1test, y_pred)
mse = mean_squared_error(y1test, y_pred)
rmse = np.sqrt(mse)
rmsle = np.sqrt(mean_squared_log_error(y1test,y_pred))

# Regression Data Table
c = np.array(c)
y_pred = y_pred.tolist()
y1test = (np.around(y1test)).tolist()
LRfit_Brazil_DC = pd.DataFrame({'Dates': c, 'Cases Prediction':y_pred, 'Cases':y1test})
LR_November_DC = LRfit_Brazil_DC.to_excel("BrazilLinearRegressionNovemberCases.xlsx", sheet_name='November Daily Cases Prediction')

# Predicting Daily Case Count from (11/23-12/31)
future_pred = []
future_days = np.arange(273,311,1)
for i in future_days:
    future = np.around(lin_reg.predict(np.array([[i]])))
    future_pred.append(future)

#Prediction Data Table    
future_days = np.arange(1,np.size(future_days)+1,1)    
LR_BrazilDC_pred = pd.DataFrame({'Days since 11/23':future_days , 'Cases Prediction':future_pred})
LR_December_DC = LR_BrazilDC_pred.to_excel("BrazilLinearRegressionDecemberCases.xlsx", sheet_name='December Daily Cases Prediction')

# Evaluation Table
LR_Brazil_Evaluation_DC = pd.DataFrame({'Model':lin_eq,'RMSE':[rmse], 'RMSLE':[rmsle], 'R2': [R2]})


print('Linear Regression Equation: ' + lin_eq)
print('RMSE Score: '+ np.str(rmse))
print('RMSLE Score: '+ np.str(rmsle))
print('R2 Score: '+ np.str(R2))
print(LRfit_Brazil_DC)
print(LR_BrazilDC_pred)


  