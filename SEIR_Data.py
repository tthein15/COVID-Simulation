import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Total Deaths
Data = pd.read_csv("owid-covid-dataSEIRD.csv")
covid_data = pd.read_csv("owid-covid-dataSEIRD.csv", parse_dates=["date"], skiprows=[1])
Data['total_cases'] = Data['total_cases'].replace(np.nan,0)
Data['total_deaths'] = Data['total_deaths'].replace(np.nan,0)

# Recovery Data
rdata = pd.read_csv("time_series_covid19_recovered_global.csv")
rdata.replace(np.nan,0)

Brazil = np.where(Data["location"] == "Brazil")[0]
recov_Brazil = np.where(rdata["Country/Region"] == "Brazil")[0]
total_recoveries_Brazil = np.int_(rdata.iloc[recov_Brazil,38:311])[0]
total_cases_Brazil = np.int_(Data.iloc[Brazil,4])[56:]
total_deaths_Brazil = np.int_(Data.iloc[Brazil,7])[56:]
total_active_cases_Brazil = np.absolute(total_cases_Brazil - total_deaths_Brazil - total_recoveries_Brazil)

India = np.where(Data["location"] == "India")[0]
recov_India = np.where(rdata["Country/Region"] == "India")[0]
total_recoveries_India = np.int_(rdata.iloc[recov_India,12:311])[0]
total_deaths_India = np.int_(Data.iloc[India,7])[30:]
total_cases_India = np.int_(Data.iloc[India,4])[30:]
total_active_cases_India = np.absolute(total_cases_India - total_deaths_India - total_recoveries_India)

US = np.where(Data["location"] == "United States")[0]
recov_US = np.where(rdata["Country/Region"] == "US")[0]
# Disputed could vary from 5 million to 8 million 
total_recoveries_US = np.int_(rdata.iloc[recov_US,7:311])[0]
total_deaths_US = np.int_(Data.iloc[US,7])[25:]
total_cases_US = np.int_(Data.iloc[US,4])[25:]
total_active_cases_US = np.absolute(total_cases_US - total_deaths_US - total_recoveries_US)

#covid_data = pd.read_csv("https://tinyurl.com/t59cgxn", parse_dates=["Date"], skiprows=[1])

#Data.groupby('date').sum()[['total_deaths']].plot(figsize=(12, 5), title="Covid-19 total fatalities (world)");
