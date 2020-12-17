import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
import requests
from lmfit import minimize, Parameters, Parameter, report_fit
import matplotlib as plt
plt.style.use('ggplot')
from SEIR_Data import *

"""
Reference https://www.idmod.org/docs/hiv/model-seir.html
"""
    
def ode_model(z, t, beta, sigma, gamma, mu):
    S, E, I, R, D = z
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def ode_solver(t, initial_conditions, params):
    initE, initI, initR, initN, initD = initial_conditions
    beta, sigma, gamma, mu = params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
    initS = initN - (initE + initI + initR + initD)
    res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu))
    return res

# Initialized Compartmental Populations
initN = 1353000000
initE = 1
initI = 0
initR = 0
initD = 0

#Estimating Parameters
def error(params, initial_conditions, tspan, data):
    sol = ode_solver(tspan, initial_conditions, params)
    return (sol[:, 2:5] - data).ravel()

# Parameter Guesses(Read Into)
initial_conditions = [initE, initI, initR, initN, initD]
beta = 1.14
sigma = .02
gamma = .02
mu = 0.01
params = Parameters()
params.add('beta', value=beta, min=0, max=10)
params.add('sigma', value=sigma, min=0, max=10)
params.add('gamma', value=gamma, min=0, max=10)
params.add('mu', value=mu, min=0, max=10)

days = 299
tspan = np.arange(0, days, 1)

total_recoveries_India = total_recoveries_India.tolist()
total_deaths_India = total_deaths_India.tolist()
total_active_cases_India = total_active_cases_India.tolist()
data1 = pd.DataFrame({'I':total_active_cases_India,'R':total_recoveries_India,'D':total_deaths_India})
data = data1.loc[0:(days-1),['I','R','D']].values

result = minimize(error, params, args=(initial_conditions, tspan, data), method='leastsq')
report_fit(result)

final = data + result.residual.reshape(data.shape)


observed_IRD = data1.loc[:, ['I', 'R', 'D']].values

tspan_fit_pred = np.arange(0, observed_IRD.shape[0], 1)
params['beta'].value = result.params['beta'].value
params['sigma'].value = result.params['sigma'].value
params['gamma'].value = result.params['gamma'].value
params['mu'].value = result.params['mu'].value
fitted_predicted = ode_solver(tspan_fit_pred, initial_conditions, params)
fitted_predicted_IRD = fitted_predicted[:, 2:5]

print("India SEIRD Evaluation")
print("-----------------------------------------------------------")
print("Fitted MAE")
print('Infected: ', np.mean(np.abs(fitted_predicted_IRD[:days, 0] - observed_IRD[:days, 0])))
print('Recovered: ', np.mean(np.abs(fitted_predicted_IRD[:days, 1] - observed_IRD[:days, 1])))
print('Dead: ', np.mean(np.abs(fitted_predicted_IRD[:days, 2] - observed_IRD[:days, 2])))

print("\nFitted RMSE")
print('Infected: ', np.sqrt(np.mean((fitted_predicted_IRD[:days, 0] - observed_IRD[:days, 0])**2)))
print('Recovered: ', np.sqrt(np.mean((fitted_predicted_IRD[:days, 1] - observed_IRD[:days, 1])**2)))
print('Dead: ', np.sqrt(np.mean((fitted_predicted_IRD[:days, 2] - observed_IRD[:days, 2])**2)))

print("\nFitted RMLSE")
print('Infected: ' + np.str(np.sqrt(mean_squared_log_error(fitted_predicted_IRD[:days, 0], observed_IRD[:days, 0]))))
print('Recovered: '+ np.str(np.sqrt(mean_squared_log_error(fitted_predicted_IRD[:days, 1],observed_IRD[:days, 1]))))
print('Dead: '+ np.str(np.sqrt(mean_squared_log_error(fitted_predicted_IRD[:days, 2],observed_IRD[:days, 2]))))
print("-----------------------------------------------------------")
print("The number of deaths on 11/23 according to the model is " + np.str(np.around(final[298,2])))


