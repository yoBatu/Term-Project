#PELİN ÇETİN - 2148849
#BATUHAN DEMİRÖZ - 2148880
#DENİZ CAN CEBECİ- 2148740
#ERDENAY GÜMÜŞLER - 2149037

import pandas as pd 
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from math import sqrt


def calculate_rms(test,estimates):
    rms = sqrt(mean_squared_error(test, estimates))
    return rms



def estimate_moving_average(dataframe, name, windowsize, sizeestimate):
   
    estimates = []
   
    copyframe = dataframe[[name]]
    for index in range(sizeestimate): 
        
        estimate = copyframe[name].rolling(windowsize).mean().iloc[-1]
        estimate = round(estimate,4) 
        estimates.append(estimate)
        size = len(copyframe)       
        copyframe.loc[size] = [estimate]
    return estimates

def estimate_SES(dataframe, name, alpha, sizeestimate):
    
    array = np.asarray(dataframe[name])
    model = SimpleExpSmoothing(array)
    fit = model.fit(smoothing_level=alpha,optimized=False)
    forecast = fit.forecast(sizeestimate)
    for index in range ( len(forecast) ):
        forecast[index] = round(forecast[index], 4)
    return forecast

def estimate_Holt(dataframe, name, alpha, slope, sizeestimate):
    array = np.asarray(dataframe[name])
    model = Holt(array)
    fit = model.fit(smoothing_level = alpha,smoothing_slope = slope)
    forecast = fit.forecast(sizeestimate)
    for index in range ( len(forecast) ):
        forecast[index] = round(forecast[index], 4)
    return forecast


def estimate_HW(dataframe, name, number_seasons, sizeestimate):
    array = np.asarray(dataframe[name])
    size = len(array)
    model = ExponentialSmoothing(array, seasonal_periods=number_seasons ,trend='add', seasonal='add')
    fit = model.fit()
    forecast = fit.forecast(sizeestimate)
    for index in range ( len(forecast) ):
        forecast[index] = round(forecast[index], 4)
    return forecast


import warnings
warnings.filterwarnings("ignore")

for i in range(0,3):
    print("Starting... First constructing models based on training and test datasets, in order to select best method.")
    df = pd.read_csv("project.txt", sep='\t')
    size = len(df)
    time_ch = ""
    if i == 0:
        time_ch = "17-1-2019"
    elif i == 1:
        time_ch = "18-1-2019"
    else:
        time_ch = "19-1-2019"
        

    testsize = 4
    trainsize = size - testsize
    train = df[(size - testsize) - trainsize : (size - testsize) - 1]
    test = df[size - testsize:]
    testarray = np.asarray(test['VALUE'])
 
    ma_estimates = estimate_moving_average(dataframe=train, name='VALUE', windowsize=30, sizeestimate=4)
    ma_rms = calculate_rms(testarray,ma_estimates)

    ses_alphas = np.linspace(0.0, 1.0, 11)
    best_alpha = 0
    best_err = 1000000.0
    best_estimates = []
    for my_alpha in ses_alphas:
        new_estimates= estimate_SES(dataframe=train, name='VALUE', alpha=my_alpha, sizeestimate=4)
        new_rms = calculate_rms(testarray, new_estimates)
        if new_rms < best_err:
            best_err = new_rms
            best_alpha = my_alpha
            best_estimates = new_estimates
    ses_rms = best_err

    holt_alphas = np.linspace(0.0, 1.0, 11)
    best_holtalpha = 0
    best_holtslope = 0
    best_holterr= 1000000
    for my_alpha in holt_alphas:
        holt_slopes = np.linspace(0.0, 1.0, 11)
        for my_slope in holt_slopes:
            new_estimates= estimate_Holt(dataframe=train, name='VALUE', alpha=my_alpha, slope=my_slope, sizeestimate=4)
            new_rms = calculate_rms(testarray, new_estimates)
            if new_rms < best_holterr:
                best_holterr = new_rms
                best_holtalpha = my_alpha
                best_holtslope = my_slope
    holt_rms = best_holterr
    hw_seasons = 2
    hw_estimates = estimate_HW(dataframe=train, name='VALUE', number_seasons=hw_seasons, sizeestimate=4)
    hw_rms = calculate_rms(testarray, hw_estimates)


    errors = [ma_rms, ses_rms, holt_rms, hw_rms]
    min_err = min(errors)

    print("Done.")
    print("Now running selected best method.")

    if ma_rms == min_err:
        print("Best method for test data is Moving Average.")
        ma_estimates = estimate_moving_average(dataframe=df, name='VALUE', windowsize=30, sizeestimate=4)
        print("MA estimate for " + time_ch + ":", ma_estimates[-1])
        with open("project.txt", "a") as myfile:
            myfile.write("\n" + time_ch +" \t" + str(ma_estimates[-1]))
    elif ses_rms == min_err:
        print("Best method for test data is Simple Exponential Smoothing.")
        ses_alpha = best_alpha
        ses_estimates= estimate_SES(dataframe=df, name='VALUE', alpha=ses_alpha, sizeestimate=4)
        print("SES Estimate for " + time_ch + ": ", ses_estimates[-1])
        with open("project.txt", "a") as myfile:
            myfile.write("\n" + time_ch +" \t" + str(ses_estimates[-1]))
    elif hw_rms == min_err:
        print("Best method for test data is Holt-Winters.")
        hw_seasons = 2
        hw_estimates = estimate_HW(dataframe=df, name='VALUE', number_seasons=hw_seasons, sizeestimate=4)
        print("HW Estimate for " + time_ch + ":", hw_estimates[-1])
        with open("project.txt", "a") as myfile:
            myfile.write("\n" + time_ch +" \t" + str(hw_estimates[-1]))
    elif holt_rms == min_err:
        print("Best method for test data is Holt.")
        holt_estimates= estimate_Holt(dataframe=df, name='VALUE', alpha=best_holtalpha, slope=best_holtslope, sizeestimate=4)
        print("Holt Estimate for " + time_ch + ":", holt_estimates[-1])
        with open("project.txt", "a") as myfile:
            myfile.write("\n" + time_ch +" \t" + str(holt_estimates[-1]))
            
file = open("project.txt")
lines = file.readlines()
file.close()
write_file = open("project.txt",'w')
write_file.writelines([item for item in lines[:-3]])
write_file.close()
