import numpy as np
import matplotlib.pyplot as plt
from India_Data_COVID import daily_deaths_India_RA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

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
plt.scatter(a,y, facecolor = 'none', edgecolor = 'blue', label = 'Data')
plt.plot(a,np.polyval(p2,x),'k--', label = 'Degree: 2')
plt.plot(a,np.polyval(p3,x),'y--', label = 'Degree: 3')
plt.plot(a,np.polyval(p4,x),'g--', label = 'Degree: 4')
plt.plot(a,np.polyval(p5,x),'b--', label = 'Degree: 5')
plt.plot(a,np.polyval(p6,x),'c--', label = 'Degree: 6')
plt.title('Daily Deaths In India: Polynomial Regression Fits')
plt.ylabel('Daily Deaths')
plt.xlabel('Days Since 12/31')
plt.legend()
plt.show()


#Cubic Fit
polynomial_features1 = PolynomialFeatures(degree = 3)
x_poly1 = polynomial_features1.fit_transform(x1)
x_poly_test1 = polynomial_features1.fit_transform(x1test)
model1 = LinearRegression()
model1.fit(x_poly1,y1)
y_poly_pred1 = np.around(model1.predict(x_poly_test1))

#Evaluate 
r2_d3 = r2_score(y1test,y_poly_pred1)
mse3 = mean_squared_error(y1test,y_poly_pred1)
rmse3 = np.sqrt(mse3)
rmsle3 = mean_squared_log_error(y1test,y_poly_pred1)

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
#rmsle4 = mean_squared_log_error(y1test,y_poly_pred2)


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
#rmsle5 = np.sqrt(mean_squared_log_error(y1test,y_poly_pred3))

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
#rmsle6 = np.sqrt(mean_squared_log_error(y1test,y_poly_pred4))

# Create Data Table for Cubit Fit(Only Fit)
c = np.array(c)
y_poly_pred1 = y_poly_pred1.tolist()
y1test = np.around(y1test)
y1test = y1test.tolist()

# DataFrame
PRfits_India_DD = pd.DataFrame({'Dates': c, 'Deaths':y1test, 'Degree = 3':y_poly_pred1})
PR_November_DD = PRfits_India_DD.to_excel("IndiaPolynomialRegressionNovemberDeaths.xlsx", sheet_name='NovemberDailyDeathsPrediction')


# Evaluate Quadratic and 6th Degree Fit(Best fit)
coef1 = model1.coef_
intercept1 = model1.intercept_

PR3BestEval_India_DD = pd.DataFrame({'R2':[r2_d3],'RMSE':[rmse3],'RMSLE':[rmsle3]})    



#December Predictions
future_days = np.arange(299,337,1)
future_days = future_days.reshape(-1,1)

future_days1 = polynomial_features1.fit_transform(future_days)
future_days2 = polynomial_features2.fit_transform(future_days)
future_days3 = polynomial_features3.fit_transform(future_days)
future_days4 = polynomial_features4.fit_transform(future_days)

future_pred_1 = np.around(model1.predict(future_days1))
future_pred_2 = model2.predict(future_days2)
future_pred_3 = model3.predict(future_days3)
future_pred_4 = model4.predict(future_days4)

future_days_gr = np.arange(1,np.size(future_days)+1,1) 

# Plot Future Predictions
plt.figure(figsize=(12,5)) 
plt.grid()
plt.scatter(np.arange(-6,1,1),y[292:299], facecolor = 'none', edgecolor = 'blue')
plt.plot(future_days_gr,future_pred_1,'y--', label = 'Degree: 3')
plt.plot(future_days_gr,future_pred_2,'g--', label = 'Degree: 4')
plt.plot(future_days_gr,future_pred_3,'b--', label = 'Degree: 5')
plt.plot(future_days_gr,future_pred_4,'c--', label = 'Degree: 6')
plt.title('Daily Deaths In India: Polynomial Regression(Predictions)')
plt.ylabel('Daily Deaths')
plt.xlabel('Days Since 11/23/20')
plt.legend()
plt.show()

# Dataframe of Future Predictons
future_pred_1 = future_pred_1.tolist()
future_pred_2 = future_pred_2.tolist()
future_pred_3 = future_pred_3.tolist()
future_pred_4 = future_pred_4.tolist()
PR_Pred_India_DD = pd.DataFrame({'Days since 11/23':future_days_gr, 'Degree = 3':future_pred_1})
PR_December_DD = PR_Pred_India_DD.to_excel("IndiaPolynamialRegressionDecemberDeaths.xlsx", sheet_name='DecemberDailyDeathsPrediction')

# Print Results
print('Cubic Polynomial Fit:')
print('Coefficients: ' + np.str(coef1))
print('Intercept: ' + np.str(intercept1))
print('RMSE Score: '+ np.str(rmse3))
print('RMSLE Score: '+ np.str(rmsle3))
print('R2 Score: '+ np.str(r2_d3))
print('-------------------------')
print('November Predictions:')
print(PRfits_India_DD)
print('-------------------------')
print('December Predictions:')
print(PR_Pred_India_DD)



   
    



