# US COVID Data Table
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

Data = pd.read_csv("owid-covid-dataUS.csv")
#Testing_Data = pd.read_csv("daily-tests-per-thousand-people-smoothed-7-day.csv")
Data['total_cases'] = Data['total_cases'].replace(np.nan,0)
Data['total_deaths'] = Data['total_deaths'].replace(np.nan,0)



# Brazil Data: Dates 1-56 were at 0 for all categories
US = np.where(Data["location"] == "United States")[0][25:]
n_days_US = np.size(US)
daily_cases_US = np.int_(Data.iloc[US,5])
daily_cases_US[daily_cases_US < 0] = 0
total_cases_US = np.int_(Data.iloc[US,4])
daily_deaths_US = np.int_(Data.iloc[US,8])
daily_deaths_US[daily_deaths_US < 0] = 0
total_deaths_US = np.int_(Data.iloc[US,7])
dates_US = Data.iloc[US,3]
date_format = [pd.to_datetime(d) for d in dates_US]

# Visualize Brazil Data
fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_cases_US)
ax.set(xlabel = "Month", ylabel = "Cases", title = "US: Daily Case Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("US_DailyCases.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,total_cases_US)
ax.set(xlabel = "Month", ylabel = "Cases", title = "US: Total Case Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("US_TotalCases.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.scatter(date_format,daily_deaths_US)
ax.set(xlabel = "Month", ylabel = "Deaths", title = "US: Daily Death Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("US_DailyDeaths.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.scatter(date_format,total_deaths_US)
ax.set(xlabel = "Month", ylabel = "Deaths", title = "US: Total Death Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("US_TotalDeaths.png", dpi = 300, bbox_inches = 'tight')

# 7-day Rolling Average: Daily Numbers
rolling_average_days = 7
df_dc = pd.DataFrame(daily_cases_US,columns = ['cases'])
daily_cases_US_RA = df_dc.rolling(window=rolling_average_days).mean()
daily_cases_US_RA['cases'] = daily_cases_US_RA['cases'].replace(np.nan,0)

df_dd = pd.DataFrame(daily_deaths_US,columns = ['deaths'])
daily_deaths_US_RA = df_dd.rolling(window=rolling_average_days).mean()
daily_deaths_US_RA['deaths'] = daily_deaths_US_RA['deaths'].replace(np.nan,0)

#Visualize
fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_cases_US_RA)
ax.set(xlabel = "Month", ylabel = "Cases", title = "US: Daily Case Count (7-Day Rolling Average)")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
plt.savefig("US_DailyCasesRA.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_deaths_US_RA)
ax.set(xlabel = "Month", ylabel = "Deaths", title = "US: Daily Death Count (7-Day Rolling Average)")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("US_DailyDeathsRA.png", dpi = 300, bbox_inches = 'tight')

