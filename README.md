# COVID-Simulation
This project takes daily death and daily case data from the Our World in Data website from January 1st - November 22nd and attempts to predict the outcomes for December in Brazil, United States, and India.

The Regression files predict the future daily case and death counts for December by using the months of January-October as training data and the month of November as testing data. Finally we make our predictions for December using the model. We also use RMSE to evaluate the fits of each model. 

The SEIRD model is taken from https://towardsdatascience.com/estimating-parameters-of-compartmental-models-from-observed-data-62f87966bb2b for estimating the parameters for our SEIRD model based on the country data. It is important to optimize initial guesses in order to get working results for the data. No method was used to obtain the guesses listed, would not recommend guess and checking. Additionally this model requires downloading the lmfit library on Anaconda.

