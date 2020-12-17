# Brazil COVID Data Table
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

Data = pd.read_csv("owid-covid-dataBrazil.csv")
#Testing_Data = pd.read_csv("daily-tests-per-thousand-people-smoothed-7-day.csv")
Data['total_cases'] = Data['total_cases'].replace(np.nan,0)
Data['total_deaths'] = Data['total_deaths'].replace(np.nan,0)


# Brazil Data: Dates 1-56 were at 0 for all categories: First Date 02/25/20
Brazil = np.where(Data["location"] == "Brazil")[0][56:]
n_days_Brazil = np.size(Brazil)
daily_cases_Brazil = np.int_(Data.iloc[Brazil,5])
total_cases_Brazil = np.int_(Data.iloc[Brazil,4])
daily_deaths_Brazil = np.int_(Data.iloc[Brazil,8])
total_deaths_Brazil = np.int_(Data.iloc[Brazil,7])
dates_Brazil = Data.iloc[Brazil,3]
date_format = [pd.to_datetime(d) for d in dates_Brazil]
date_format1 = date_format[0]

# Visualize Brazil Data
fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_cases_Brazil)
ax.set(xlabel = "Date", ylabel = "Number of Cases", title = "Brazil: Daily Case Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("Brazil_DailyCases.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,total_cases_Brazil)
ax.set(xlabel = "Date", ylabel = "Number of Cases", title = "Brazil: Total Case Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.savefig("Brazil_TotalCases.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_deaths_Brazil)
ax.set(xlabel = "Date", ylabel = "Number of Cases", title = "Brazil: Daily Death Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("Brazil_DailyDeaths.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,total_deaths_Brazil)
ax.set(xlabel = "Date", ylabel = "Number of Cases", title = "Brazil: Total Death Count")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.savefig("Brazil_TotalDeaths.png", dpi = 300, bbox_inches = 'tight')

# 7-day Rolling Average: Daily Numbers
rolling_average_days = 7
df_dc = pd.DataFrame(daily_cases_Brazil,columns = ['cases'])
daily_cases_Brazil_RA = df_dc.rolling(window=rolling_average_days).mean()
daily_cases_Brazil_RA['cases'] = daily_cases_Brazil_RA['cases'].replace(np.nan,0)

df_dd = pd.DataFrame(daily_deaths_Brazil,columns = ['deaths'])
daily_deaths_Brazil_RA = df_dd.rolling(window=rolling_average_days).mean()
daily_deaths_Brazil_RA['deaths'] = daily_deaths_Brazil_RA['deaths'].replace(np.nan,0)

#Visualize
fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_cases_Brazil_RA)
ax.set(xlabel = "Date(M/Y)", ylabel = "Number of Cases", title = "Brazil: Daily Case Count (7-Day Rolling Average)")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
#plt.show()
#plt.savefig("Brazil_DailyCasesRA.png", dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (12,5))
ax.grid()
ax.plot(date_format,daily_deaths_Brazil_RA)
ax.set(xlabel = "Date(M/Y)", ylabel = "Number of Deaths", title = "Brazil: Daily Death Count (7-Day Rolling Average)")
date_form = DateFormatter("%m-%y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 31))
plt.show()
#plt.savefig("Brazil_DailyDeathsRA.png", dpi = 300, bbox_inches = 'tight')

# Tests per day: Brazil Can't Perform TEST COMPARISON(not enough tests)but will be done for US
#RA_dailytests_Brazil = np.where(Testing_Data["Entity"] == "Brazil")[0]
#dailytests_Brazil = (np.array(Testing_Data.iloc[RA_dailytests_Brazil,3])*1000)
#testingdates_Brazil = Testing_Data.iloc[RA_dailytests_Brazil,3]
#testdate_format = [pd.to_datetime(d) for d in dates_Brazil]
                               
