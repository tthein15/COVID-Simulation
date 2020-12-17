import numpy as np
import matplotlib.pyplot as plt
from US_Data_COVID import daily_cases_US_RA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


# Day 0 = 01/25/20
a = date_format
b = date_format[:281]
c = date_format[281:304]
x = np.int_(np.where(Data.iloc[US,3]))[0]
y = np.array(daily_cases_US_RA)

# Train/Test: Split Train: Feb-October; Test: November 
xtrain = np.int_(np.where(Data.iloc[US,3]))[0][:281]
xtest = np.int_(np.where(Data.iloc[US,3]))[0][281:304]
ytrain = np.array(daily_cases_US_RA)[:281]
ytest = np.array(daily_cases_US_RA)[281:304]

# Reshape Data
x1 = xtrain.reshape(-1,1)
x1test = xtest.reshape(-1,1)
y1 = ytrain.reshape(-1,1)
y1test = ytest.reshape(-1,1)


# Check Fits and Plot
p1 = np.polyfit(x,y,1)
p2 = np.polyfit(x,y,2)
p3 = np.polyfit(x,y,3)
p4 = np.polyfit(x,y,4)
p5 = np.polyfit(x,y,5)
p6 = np.polyfit(x,y,6)

# Check fits
plt.figure(figsize=(12,5)) 
plt.grid()
plt.scatter(x,y, facecolor = 'none', edgecolor = 'blue', label = 'Data')
#plt.plot(x,np.polyval(p1,x),'r--', label = 'Degree: 1')
#plt.plot(x,np.polyval(p2,x),'k--', label = 'Degree: 2')
plt.plot(x,np.polyval(p3,x),'y--', label = 'Degree: 3')
plt.plot(x,np.polyval(p4,x),'g--', label = 'Degree: 4')
plt.plot(x,np.polyval(p5,x),'b--', label = 'Degree: 5')
plt.plot(x,np.polyval(p6,x),'c--', label = 'Degree: 6')
plt.title('Daily Cases In US: Polynomial Regression Fits')
plt.ylabel('Daily Cases')
plt.xlabel('Days Since 12/31')
plt.legend()
#plt.show()
#plt.savefig("RegressionFits.png", dpi = 300, bbox_inches = 'tight')

#Cubic Fit
polynomial_features1 = PolynomialFeatures(degree = 3)
x_poly1 = polynomial_features1.fit_transform(x1)
x_poly_test1 = polynomial_features1.fit_transform(x1test)
model1 = LinearRegression()
model1.fit(x_poly1,y1)
y_poly_pred1 = model1.predict(x_poly_test1)

#Evaluate 
r2_d3 = r2_score(y1test,y_poly_pred1)
mse3 = mean_squared_error(y1test,y_poly_pred1)
rmse3 = np.sqrt(mse3)

#Quadratic Fit
polynomial_features2 = PolynomialFeatures(degree = 4)
x_poly2 = polynomial_features2.fit_transform(x1)
x_poly_test2 = polynomial_features2.fit_transform(x1test)
model2 = LinearRegression()
model2.fit(x_poly2,y1)
y_poly_pred2 = np.around(model2.predict(x_poly_test2))

#Evaluate
r2_d4 = r2_score(y1test,y_poly_pred2)
mse4 = mean_squared_error(y1test,y_poly_pred2)
rmse4 = np.sqrt(mse4)
rmsle4 = mean_squared_log_error(y1test,y_poly_pred2)


# 5th Degree 
polynomial_features3 = PolynomialFeatures(degree = 5)
x_poly3 = polynomial_features3.fit_transform(x1)
x_poly_test3 = polynomial_features3.fit_transform(x1test)
model3 = LinearRegression()
model3.fit(x_poly3,y1)
y_poly_pred3 = np.around(model3.predict(x_poly_test3))

#Evaluate
r2_d5 = r2_score(y1test,y_poly_pred3)
mse5 = mean_squared_error(y1test,y_poly_pred3)
rmse5 = np.sqrt(mse5)
rmsle5 = np.sqrt(mean_squared_log_error(y1test,y_poly_pred3))

# 6th Degree
polynomial_features4 = PolynomialFeatures(degree = 6)
x_poly4 = polynomial_features4.fit_transform(x1)
x_poly_test4 = polynomial_features4.fit_transform(x1test)
model4 = LinearRegression()
model4.fit(x_poly4,y1)
y_poly_pred4 = np.around(model4.predict(x_poly_test4))

#Evaluate
r2_d6 = r2_score(y1test,y_poly_pred4)
mse6 = mean_squared_error(y1test,y_poly_pred4)
rmse6 = np.sqrt(mse6)
rmsle6 = np.sqrt(mean_squared_log_error(y1test,y_poly_pred4))

# Create Data Table for Polynomial Fits
c = np.array(c)
y_poly_pred3 = y_poly_pred3.tolist()
y_poly_pred4 = y_poly_pred4.tolist()
y1test = np.around(y1test)
y1test = y1test.tolist()

# DataFrame
PRfits_US_DC = pd.DataFrame({'Dates': c, 'Cases':y1test, 'Degree = 5':y_poly_pred3,'Degree = 6':y_poly_pred4})
PR_November_DC = PRfits_US_DC.to_excel("USPolynomialRegressionNovemberCases.xlsx", sheet_name='NovemberDailyCasesPrediction')


# Evaluate 5th and 6th Degree Fit(Best fit)
coef3 = model3.coef_
intercept3 = model3.intercept_

coef4 = model4.coef_
intercept4 = model4.intercept_

PR5BestEval_US_DC = pd.DataFrame({'R2':[r2_d5],'RMSE':[rmse5],'RMSLE':[rmsle5]})    
PR6BestEval_US_DC = pd.DataFrame({'R2':[r2_d6],'RMSE':[rmse6],'RMSLE':[rmsle6]})


# December Death Predicitons
future_days = np.arange(304,342,1)
future_days = future_days.reshape(-1,1)

future_days1 = polynomial_features1.fit_transform(future_days)
future_days2 = polynomial_features2.fit_transform(future_days)
future_days3 = polynomial_features3.fit_transform(future_days)
future_days4 = polynomial_features4.fit_transform(future_days)

future_pred_1 = np.around(model1.predict(future_days1),0)
future_pred_2 = np.around(model2.predict(future_days2),0)
future_pred_3 = np.around(model3.predict(future_days3),0)
future_pred_4 = np.around(model4.predict(future_days4),0)

future_days_gr = np.arange(1,np.size(future_days)+1,1) 

#PLot Predictions
plt.figure(figsize=(12,5)) 
plt.grid()
plt.scatter(np.arange(-6,1,1),y[297:304], facecolor = 'none', edgecolor = 'blue', label = '29,758 New Cases(11/23)')
plt.plot(future_days_gr,future_pred_3,'y--', label = 'Degree: 5')
plt.plot(future_days_gr,future_pred_4,'g--', label = 'Degree: 6')
#plt.plot(future_days_gr,future_pred_3,'b--', label = 'Degree: 5')
plt.title('Daily Cases In US: Polynomial Regression(Predictions)')
plt.ylabel('Daily Cases')
plt.xlabel('Days Since 11/23/20')
plt.legend()
plt.show()

# Dataframe for Predictions
future_pred_1 = future_pred_1.tolist()
future_pred_2 = future_pred_2.tolist()
future_pred_3 = future_pred_3.tolist()
future_pred_4 = future_pred_4.tolist()
PR_Pred_US_DC = pd.DataFrame({'Days since 11/23':future_days_gr,'Degree = 5':future_pred_3,'Degree = 6':future_pred_4})
PR_December_DC = PR_Pred_US_DC.to_excel("USPolynomialRegressionDecemberCases.xlsx", sheet_name='DecemberDailyCasesPrediction')

#Print Best Fit Results
print('5th Degree Polynomial Fit:')
print('Coefficients: ' + np.str(coef3))
print('Intercept: ' + np.str(intercept3))
print('RMSE Score: '+ np.str(rmse5))
print('RMSLE Score: '+ np.str(rmsle5))
print('R2 Score: '+ np.str(r2_d5))
print('-------------------------')
print('6th Degree Polynomial Fit:')
print('Coefficients: ' + np.str(coef4))
print('Intercept: ' + np.str(intercept4))
print('RMSE Score: '+ np.str(rmse6))
print('RMSLE Score: '+ np.str(rmsle6))
print('R2 Score: '+ np.str(r2_d6))
print('-------------------------')
print('November Predictions:')
print(PRfits_US_DC)
print('-------------------------')
print('December Predictions:')
print(PR_Pred_US_DC)