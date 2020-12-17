# India COVID Data Table
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

Data = pd.read_csv("owid-covid-dataIndia.csv")
#Testing_Data = pd.read_csv("daily-tests-per-thousand-people-smoothed-7-day.csv")
Data['total_cases'] = Data['total_cases'].replace(np.nan,0)
Data['total_deaths'] = Data['total_deaths'].replace(np.nan,0)

# India First Case(1/30/20)
India = np.where(Data["location"] == "India")[0][30:]
n_days_India = np.size(India)
daily_cases_India = np.int_(Data.iloc[India,5])
daily_cases_India[daily_cases_India < 0] = 0
total_cases_India = np.int_(Data.iloc[India,4])
daily_deaths_India = np.int_(Data.iloc[India,8])
daily_deaths_India[daily_deaths_India < 0] = 0
total_deaths_India = np.int_(Data.iloc[India,7])
dates_India = Data.iloc[India,3]
date_format = [pd.to_datetime(d) for d in dates_India]

# Visualize India Data
fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_cases_India)
ax.set(xlabel = "Month", ylabel = "Cases", title = "India: Daily Case Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("India_DailyCases.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,total_cases_India)
ax.set(xlabel = "Month", ylabel = "Cases", title = "India: Total Case Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("India_TotalCases.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_deaths_India)
ax.set(xlabel = "Month", ylabel = "Deaths", title = "India: Daily Death Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("India_DailyDeaths.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,total_deaths_India)
ax.set(xlabel = "Month", ylabel = "Deaths", title = "India: Total Death Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("India_TotalDeaths.png", dpi = 300, bbox_inches = 'tight')

# 7-day Rolling Average: Daily Numbers
rolling_average_days = 7
df_dc = pd.DataFrame(daily_cases_India,columns = ['cases'])
daily_cases_India_RA = df_dc.rolling(window=rolling_average_days).mean()
daily_cases_India_RA['cases'] = daily_cases_India_RA['cases'].replace(np.nan,0)

df_dd = pd.DataFrame(daily_deaths_India,columns = ['deaths'])
daily_deaths_India_RA = df_dd.rolling(window=rolling_average_days).mean()
daily_deaths_India_RA['deaths'] = daily_deaths_India_RA['deaths'].replace(np.nan,0)

#Visualize
fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_cases_India_RA)
ax.set(xlabel = "Month", ylabel = "Number of Cases", title = "India: Daily Case Count (7-Day Rolling Average)")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
plt.savefig("India_DailyCasesRA.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_deaths_India_RA)
ax.set(xlabel = "Month", ylabel = "Number of Deaths", title = "India: Daily Death Count (7-Day Rolling Average)")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("India_DailyDeathsRA.png", dpi = 300, bbox_inches = 'tight')

