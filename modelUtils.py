"""
Created on Wed Jan 22 2022

@author: L GOLINO
"""


import gzip, pickle, scipy
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, r2_score 
from scipy.optimize import curve_fit


###############################################
###### TRAINING DATA IMPORT AND HANDLING ######
###############################################


def import_data(metadata, fileloc, index_of_file=5, limit=-1):
    #Using the metadata dataframe, scan and attempt to open each CSV. 
    #This will save the STD DEV of both the real and distorted profiles as well as the full profiles.
    real_x_stddevs = []
    real_x_hists = []
    distorted_x_hists = []
    
    i = 0
    number_variables=0
    while(number_variables == 0):
        try:
            with gzip.open(fileloc + metadata.iloc[i][index_of_file] + '.pkl.gz', 'rb') as f:
                        find_len = pickle.load(f)
            number_variables = len(find_len)
        except:
            i=i+1
    
    i = 0
    for row in metadata.iterrows():
        if(i%100==0):
            print('Opening ', fileloc + row[1][index_of_file] + '.pkl.gz')
        i = i+1
        real_stddev = np.NaN
        real_profile_hist = []
        distorted_profile_hist_max = []
       
        try:
            if number_variables == 5:
                with gzip.open(fileloc + row[1][index_of_file] + '.pkl.gz', 'rb') as f:
                    real_stddev, bins, real_profile_hist, bins2, distorted_profile_hist_max = pickle.load(f)
                    
            if number_variables == 6:
                with gzip.open(fileloc + row[1][index_of_file] + '.pkl.gz', 'rb') as f:
                    real_stddev, dis_stddev, bins, real_profile_hist, bins2, distorted_profile_hist_max = pickle.load(f)

            if number_variables == 13:
                with gzip.open(fileloc + row[1][index_of_file] + '.pkl.gz', 'rb') as f:
                    real_stddev, bins, real_profile_hist, bins2_max, distorted_profile_hist_max, distorted_profile_hist_5k, \
                    distorted_profile_hist_10k, distorted_profile_hist_25k, distorted_profile_hist_50k, distorted_profile_hist_100k, \
                    distorted_profile_hist_200k, distorted_profile_hist_500k, distorted_profile_hist_1mil = pickle.load(f)
                    
            if number_variables == 14:
                with gzip.open(fileloc + row[1][index_of_file] + '.pkl.gz', 'rb') as f:
                    real_stddev, dis_stddev, bins, real_profile_hist, bins2_max, distorted_profile_hist_max, distorted_profile_hist_5k, \
                    distorted_profile_hist_10k, distorted_profile_hist_25k, distorted_profile_hist_50k, distorted_profile_hist_100k, \
                    distorted_profile_hist_200k, distorted_profile_hist_500k, distorted_profile_hist_1mil = pickle.load(f)

        except:
            print('FAILED: ', row[1][index_of_file])
            print('This one didnt work, likely a condor error. If some of your jobs didnt run, its fine.')
            #If this is the case we will append empty vectors to the df so the data is still alligned then drop NaNs later
            real_stddev = np.NaN
            real_profile_hist = []
            distorted_profile_hist = []

        #Append the data to required vectors. Feel free to use whatever you want below to train and test.
        real_x_stddevs.append(real_stddev)
        real_x_hists.append(real_profile_hist)
        distorted_x_hists.append(distorted_profile_hist_max)
        if(limit>0) and (i>limit):
            break
    print('Finished: Total = ', i)
    return real_x_stddevs, real_x_hists, distorted_x_hists

def import_data_as_vector(metadata, fileloc, index_of_file=5, limit=-1):
    #Using the metadata dataframe, scan and attempt to open each CSV.
    #This will save the STD DEV of both the real and distorted profiles as well as the full profiles.
    data = []
    i = 0
    for row in metadata.iterrows():
        if(i%100==0):
            print('Opening ', fileloc + row[1][index_of_file] + '.pkl.gz')
        i = i+1
        try:
            with gzip.open(fileloc + row[1][index_of_file] + '.pkl.gz', 'rb') as f:
                a = pickle.load(f)
        except:
            print('FAILED: ', row[1][index_of_file])
            print('This one didnt work, likely a condor error. If some of your jobs didnt run, its fine.')
            #If this is the case we will append empty vectors to the df so the data is still alligned then drop NaNs later
            a = [np.NaN]

        #Append the data to required vectors. Feel free to use whatever you want below to train and test.
        data.append(a)
        if(limit>0) and (i>limit):
            break
    print('Finished: Total = ', i)
    return data



def convert_data_to_df(real_x_stddevs, real_x_hists, distorted_x_hists):
    #The pipeline will do both of these and compare them.
    real_profiles = pd.DataFrame(real_x_hists) #Use this to reconstruct the real profile from the distorted.
    real_stddev = pd.DataFrame(real_x_stddevs) #Use this to reconstruct the entire profile.
    distorted_profiles = pd.DataFrame(distorted_x_hists) #This will always be xs. It's the signal on the detectors.

    #Drop NaN rows entirely.
    real_profiles = real_profiles.dropna()
    real_stddev = real_stddev.dropna()
    distorted_profiles = distorted_profiles.dropna()


    #Just a quick check that all vectors have the same length.
    if(len(real_profiles) == len(real_stddev) == len(distorted_profiles)):
        print('All three of your training data sizes are good. You may continue.')
    else:
        print('Something bad has happened. The length of your training data is not equal.', len(real_profiles), len(real_stddev), len(distorted_profiles))

    return real_profiles, real_stddev, distorted_profiles

def save_to_pickle(metadata, fileloc, real_profiles, real_stddev, distorted_profiles, append='_warning_511-likely_overwrite'):
    #Sort out distorted data for our model. X data is the distorted profiles.
    profilemetrics = metadata
    profilemetrics = profilemetrics.drop('Unnamed: 0', axis=1)
    profilemetrics = profilemetrics.drop('Configuration filename', axis=1)
    profilemetrics = profilemetrics.drop('Beams/Beam[0]/BunchTrain/LongitudinalOffset', axis=1)
    profilemetrics = profilemetrics.drop('Simulation/Output/Parameters/Filename', axis=1)
    all_distorted_data = pd.concat([profilemetrics, distorted_profiles], axis=1)
    all_distorted_data = all_distorted_data.dropna()
    #Save to pickles.
    all_distorted_data.to_pickle(fileloc + "DistortedProfilesDF_" + append + ".pkl")
    real_stddev.to_pickle(fileloc + "RealStdDevsDF_" + append + ".pkl" )
    real_profiles.to_pickle(fileloc + "RealProfilesDF_" + append + ".pkl")
    print('All data saved to pickle with the suffix '+ append)


def import_and_save(metadata, fileloc, name='_warning_511-likely_overwrite', limit=100, index_of_file=5):
    real_x_stddevs, real_x_hists, distorted_x_hists = import_data(metadata, fileloc,index_of_file, limit)
    real_profiles, real_stddev, distorted_profiles = convert_data_to_df(real_x_stddevs, real_x_hists, distorted_x_hists)
    save_to_pickle(metadata, fileloc, real_profiles, real_stddev, distorted_profiles, name)

####################################
###### END OF TRAINING IMPORT ######
####################################


def find_errors(metadata, fileloc, index_of_file=5, limit=-1):
    i = 0

    for row in metadata.iterrows():
        try:
            with gzip.open(fileloc + row[1][index_of_file] + '.pkl.gz', 'rb') as f:
                continue
        except:
            print(row[1][index_of_file])

        if(limit>0) and (i>limit):
            break

    print('Finished: Total = ', i)
    return real_x_stddevs, real_x_hists, distorted_x_stddevs, distorted_x_hists

#####################################
###### BEGIN OF TESTING IMPORT ######
#####################################


def import_testing_data(metadata, fileloc, index_of_file=5, limit=-1):
    # Using the metadata dataframe, scan and attempt to open each CSV.
    # This will save the STD DEV of both the real and distorted profiles as well as the full profiles.
    real_x_stddevs = []
    real_x_hists = []
    # distorted_x_stddevs = []

    distorted_x_hists_max = []
    distorted_x_hists_5k = []
    distorted_x_hists_10k = []
    distorted_x_hists_25k = []
    distorted_x_hists_50k = []
    distorted_x_hists_100k = []
    distorted_x_hists_200k = []
    distorted_x_hists_500k = []
    distorted_x_hists_1mil = []

    i = 0
    number_variables=0
    while(number_variables == 0):
        try:
            with gzip.open(fileloc + metadata.iloc[i][index_of_file] + '.pkl.gz', 'rb') as f:
                        find_len = pickle.load(f)
            number_variables = len(find_len)
        except:
            i=i+1

    i=0        
    for row in metadata.iterrows():
        if (i % 100 == 0):
            print('Opening ', row[1][index_of_file])
        i = i + 1
        try:
            if number_variables == 13:
                with gzip.open(fileloc + row[1][index_of_file] + '.pkl.gz', 'rb') as f:
                    real_stddev, bins, real_profile_hist, bins2_max, distorted_profile_hist_max, distorted_profile_hist_5k, \
                    distorted_profile_hist_10k, distorted_profile_hist_25k, distorted_profile_hist_50k, distorted_profile_hist_100k, \
                    distorted_profile_hist_200k, distorted_profile_hist_500k, distorted_profile_hist_1mil = pickle.load(f)
                    
            if number_variables == 14:
                with gzip.open(fileloc + row[1][index_of_file] + '.pkl.gz', 'rb') as f:
                    real_stddev, dis_stddev, bins, real_profile_hist, bins2_max, distorted_profile_hist_max, distorted_profile_hist_5k, \
                    distorted_profile_hist_10k, distorted_profile_hist_25k, distorted_profile_hist_50k, distorted_profile_hist_100k, \
                    distorted_profile_hist_200k, distorted_profile_hist_500k, distorted_profile_hist_1mil = pickle.load(f)

        except:
            print('FAILED: ', row[1][index_of_file])
            print('This one didnt work, likely a condor error. If some of your jobs didnt run, its fine.')
            # If this is the case we will append empty vectors to the df so the data is still alligned then drop NaNs later
            real_stddev = np.NaN
            real_profile_hist = []
            # distorted_stddev = np.NaN
            distorted_profile_hist_max = []
            distorted_profile_hist_5k = []
            distorted_profile_hist_10k = []
            distorted_profile_hist_25k = []
            distorted_profile_hist_50k = []
            distorted_profile_hist_100k = []
            distorted_profile_hist_200k = []
            distorted_profile_hist_500k = []
            distorted_profile_hist_1mil = []

        # Append the data to required vectors. Feel free to use whatever you want below to train and test.
        real_x_stddevs.append(real_stddev)
        real_x_hists.append(real_profile_hist)
        # distorted_x_stddevs.append(distorted_stddev)
        distorted_x_hists_max.append(distorted_profile_hist_max)
        distorted_x_hists_5k.append(distorted_profile_hist_5k)
        distorted_x_hists_10k.append(distorted_profile_hist_10k)
        distorted_x_hists_25k.append(distorted_profile_hist_25k)
        distorted_x_hists_50k.append(distorted_profile_hist_50k)
        distorted_x_hists_100k.append(distorted_profile_hist_100k)
        distorted_x_hists_200k.append(distorted_profile_hist_200k)
        distorted_x_hists_500k.append(distorted_profile_hist_500k)
        distorted_x_hists_1mil.append(distorted_profile_hist_1mil)
        if (limit > 0) and (i > limit):
            break

    print('Finished: Total = ', i)
    return real_x_stddevs, real_x_hists, distorted_x_hists_max, distorted_x_hists_5k, distorted_x_hists_10k, \
           distorted_x_hists_25k, distorted_x_hists_50k, distorted_x_hists_100k, distorted_x_hists_200k, \
           distorted_x_hists_500k, distorted_x_hists_1mil

def convert_testing_data_to_df(real_x_stddevs, real_x_hists, distorted_x_hists_max, distorted_x_hists_5k, distorted_x_hists_10k, \
           distorted_x_hists_25k, distorted_x_hists_50k, distorted_x_hists_100k, distorted_x_hists_200k, \
           distorted_x_hists_500k, distorted_x_hists_1mil):
    #The pipeline will do both of these and compare them.
    real_profiles = pd.DataFrame(real_x_hists) #Use this to reconstruct the real profile from the distorted.
    real_stddev = pd.DataFrame(real_x_stddevs) #Use this to reconstruct the entire profile.
    distorted_profiles_5k = pd.DataFrame(distorted_x_hists_5k) #This will always be xs. It's the signal on the detectors.
    distorted_profiles_10k = pd.DataFrame(distorted_x_hists_10k) #This will always be xs. It's the signal on the detectors.
    distorted_profiles_25k = pd.DataFrame(distorted_x_hists_25k) #This will always be xs. It's the signal on the detectors.
    distorted_profiles_50k = pd.DataFrame(distorted_x_hists_50k) #This will always be xs. It's the signal on the detectors.
    distorted_profiles_100k = pd.DataFrame(distorted_x_hists_100k) #This will always be xs. It's the signal on the detectors.
    distorted_profiles_200k = pd.DataFrame(distorted_x_hists_200k) #This will always be xs. It's the signal on the detectors.
    distorted_profiles_500k = pd.DataFrame(distorted_x_hists_500k) #This will always be xs. It's the signal on the detectors.
    distorted_profiles_1mil = pd.DataFrame(distorted_x_hists_1mil) #This will always be xs. It's the signal on the detectors.
    distorted_profiles_max = pd.DataFrame(distorted_x_hists_max) #This will always be xs. It's the signal on the detectors.

    #Drop NaN rows entirely.
    real_profiles = real_profiles.dropna()
    real_stddev = real_stddev.dropna()
    distorted_profiles_5k = distorted_profiles_5k.dropna()
    distorted_profiles_10k = distorted_profiles_10k.dropna()
    distorted_profiles_25k = distorted_profiles_25k.dropna()
    distorted_profiles_50k = distorted_profiles_50k.dropna()
    distorted_profiles_100k = distorted_profiles_100k.dropna()
    distorted_profiles_200k = distorted_profiles_200k.dropna()
    distorted_profiles_500k = distorted_profiles_500k.dropna()
    distorted_profiles_1mil = distorted_profiles_1mil.dropna()
    distorted_profiles_max = distorted_profiles_max.dropna()


    #Just a quick check that all vectors have the same length.
    if(len(real_profiles) == len(real_stddev) == len(distorted_profiles_5k) == len(distorted_profiles_10k) == len(distorted_profiles_25k) == len(distorted_profiles_50k) == len(distorted_profiles_100k) 
    == len(distorted_profiles_200k) == len(distorted_profiles_500k) == len(distorted_profiles_1mil) == len(distorted_profiles_max)):
        print('All three of your training data sizes are good. You may continue.')
    else:
        print('Something bad has happened. The length of your training data is not equal.')

    return real_profiles, real_stddev, distorted_profiles_5k, distorted_profiles_10k, distorted_profiles_25k, \
           distorted_profiles_50k, distorted_profiles_100k, distorted_profiles_200k, distorted_profiles_500k, \
           distorted_profiles_1mil, distorted_profiles_max

###################################
###### END OF TESTING IMPORT ######
###################################




##################################################################################
###### BEGIN LOADING FROM PICKLE AND USING DATA TO TRAIN AND MEASURE MODELS ######
##################################################################################


def load_from_pickle(fileloc, append, drop_params=True):
    #So we load in from the pickles. This allows us to save different processing of the data and pull from
    #differently processed data.
    X = pd.read_pickle(fileloc + "DistortedProfilesDF_" + append + ".pkl")

    #For now for example we don't want this data but we will soon want to be able to incorperate this.
    #This data is stored in the pickle files so we don't have to change any of the preprocessing to access this again.
    X = X.drop('Beams/Beam[0]/BunchShape/Parameters/TransverseSigma', axis=1)
    X = X.drop('Beams/Beam[0]/BunchTrain/TransverseOffset', axis=1)
    X = X.drop('GuidingFields/Electric/Parameters/ElectricField', axis=1)
    X = X.drop('GuidingFields/Magnetic/Parameters/MagneticField', axis=1)

    #Also drop these
    X = X.drop('Beams/Beam[0]/Parameters/Energy', axis=1)
    if(drop_params):
        X = X.drop('Beams/Beam[0]/Parameters/BunchPopulation', axis=1)
        X = X.drop('Beams/Beam[0]/BunchShape/Parameters/LongitudinalSigmaLabFrame', axis=1)
    X = X.drop('Simulation/NumberOfParticles', axis=1)

    #Same with target (y) data.
    ystddev = pd.read_pickle(fileloc + "RealStdDevsDF_" + append + ".pkl" )
    yhists = pd.read_pickle(fileloc + "RealProfilesDF_" + append + ".pkl")
    #Convert to metres. Not really needed.
    #ystddev = ystddev*1e6
    ystddev

    return X, ystddev, yhists

def normalise_profile_percent(profile_orig):
    profile = profile_orig.div(profile_orig.sum(axis=1), axis=0)
    profile = profile*100
    return profile

def print_model_metrics(X_train, y_train, X_test, y_test, model):
    # Model 1 std dev approach:
    print('Printing metrics about model 1 (std dev).')
    predicted = model.predict(X_test)
    # The coefficients
    #print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.15f in um." % (mean_squared_error(y_test, predicted)))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.15f" % r2_score(y_test, predicted))
    #Score on training data
    print("Score: %.15f \n" % model.score(X_train, y_train))


def plot_example_recon(index, bin_width, X, y_test_profile, y_pred_profile, y_pred_stddev):
    #Plotting the profile and its prediction. 
    xaxis = np.arange(-0.01, 0.01, bin_width)

    #This is the real profile - we need to normalise so the area underneath is 1.
    real_profile = y_test_profile.iloc[index]
    normalised_real_profile = real_profile*(1/((real_profile * bin_width).sum()))

    #This is a normal dist with a sigma as predicted.
    normal_dist = stats.norm.pdf(xaxis, 0, y_pred_stddev[index])

    #This is the predicted profile
    pred_profile = y_pred_profile[index]
    normalised_pred_profile = pred_profile*(1/((pred_profile * bin_width).sum()))

    #Distorted profile.
    distorted_profile = X.iloc[index]*(1/((X.iloc[index] * bin_width).sum()))
    fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=80)

    axs[0,0].plot(xaxis, normalised_real_profile, label = 'Real Profile')
    axs[0,0].plot(xaxis, normal_dist, label = 'Normal distribution from predicted std dev')
    axs[0,0].plot(xaxis, distorted_profile.loc[0:], label = 'Measured/Distorted profile')
    axs[0,0].legend()

    axs[0,1].plot(xaxis, normalised_real_profile, label = 'Real profile')
    axs[0,1].plot(xaxis, normalised_pred_profile, label = 'Predicted profile')
    axs[0,1].plot(xaxis, distorted_profile.loc[0:], label = 'Measured/Distorted profile')
    axs[0,1].legend() 

    axs[1,0].plot(xaxis, abs(normalised_real_profile - normal_dist), color='g') 
    axs[1,0].set_title('Absolute difference between real profile and normal distribution.')

    axs[1,1].plot(xaxis, abs(normalised_real_profile - normalised_pred_profile), color='r')
    axs[1,1].set_title('Absolute difference between real profile and reconstructed profile.')

    #Calculate chi2 between two plots
    chi2_sd = (normal_dist - normalised_real_profile)**2 / normalised_real_profile
    chi2_fp = (normalised_pred_profile - normalised_real_profile)**2 / normalised_pred_profile

    chi2_sd.replace([np.inf, -np.inf], np.nan, inplace=True)
    chi2_fp.replace([np.inf, -np.inf], np.nan, inplace=True)

    #Print distances between plots
    print('SD: Manhattan distance between plots = ', abs(normalised_real_profile - normal_dist).sum() / len(normalised_real_profile))
    print('SD: MS distance between plots = ', (((normalised_real_profile - normal_dist) ** 2).sum() ) / len(normalised_real_profile))
    print('SD: Chi2 goodness of fit:', chi2_sd.dropna().sum())
    print('\n')
    print('FP: Manhattan distance between plots = ', abs(normalised_real_profile - normalised_pred_profile).sum() / len(normalised_real_profile))
    print('FP: MS distance between plots = ', (((normalised_real_profile - normalised_pred_profile) ** 2).sum() ) / len(normalised_real_profile))
    print('FP: Chi2 goodness of fit:', chi2_fp.dropna().sum())


def chi2(observed, expected):
    chi2_val = (observed - expected)**2 / expected
    chi2_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    return chi2_val.dropna().sum()

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    
def fit_gaussian(x, y):
    popt, pcov = curve_fit(gaussian, x, y)
    return popt

def print_error_plots(bin_width, X_test, y_test_profile, y_pred_profile, y_pred_stddev):
    manhat_stddev = []
    mse_stddev = []
    chi2_stddev = []
    manhat_profile = []
    mse_profile = []
    chi2_profile = []
    xaxis = np.arange(-0.01, 0.01, bin_width)

    for i in range(len(X_test)):
        #This is the real profile - we need to normalise so the area underneath is 1.
        real_profile = y_test_profile.iloc[i]
        normalised_real_profile = real_profile*(1/((real_profile * bin_width).sum()))

        #This is a normal dist with a sigma as predicted.
        normal_dist = stats.norm.pdf(xaxis, 0, y_pred_stddev[i])

        #This is the predicted profile
        pred_profile = y_pred_profile[i]
        normalised_pred_profile = pred_profile*(1/((pred_profile * bin_width).sum()))

        #This is the chi2
        chi2_sd = (normal_dist - normalised_real_profile)**2 / normalised_real_profile
        chi2_fp = (normalised_pred_profile - normalised_real_profile)**2 / normalised_pred_profile
        chi2_sd.replace([np.inf, -np.inf], np.nan, inplace=True)
        chi2_fp.replace([np.inf, -np.inf], np.nan, inplace=True)

        manhat_stddev.append(abs(normalised_real_profile - normal_dist).sum() / len(normalised_real_profile))
        mse_stddev.append((((normalised_real_profile - normal_dist) ** 2).sum() ) / len(normalised_real_profile))
        chi2_stddev.append(chi2_sd.dropna().sum())
        manhat_profile.append(abs(normalised_real_profile - normalised_pred_profile).sum() / len(normalised_real_profile))
        mse_profile.append((((normalised_real_profile - normalised_pred_profile) ** 2).sum() ) / len(normalised_real_profile))
        chi2_profile.append(chi2_fp.dropna().sum())

    #Printing the mean of the errors.
    print('Mean of the manhat distance between real and recon via SD approach:', np.mean(manhat_stddev))
    print('MSE between real and recon via SD approach:', np.mean(mse_stddev))
    print('Mean of the manhat distance between real and recon via FP approach:', np.mean(manhat_profile))
    print('MSE between real and recon via FP approach:', np.mean(mse_profile))

    #Plotting the individual errors.
    fig, axs = plt.subplots(1, 3, figsize=(20, 10), dpi=80)
    errx = range(len(X_test))
    axs[0].plot(errx, manhat_stddev, label = 'Reconstruction from stddev')
    axs[0].plot(errx, manhat_profile, label = 'Full reconstruction')
    axs[0].set_title('Manhattan distance between real profile and reconstructed.')
    axs[0].legend()
    axs[1].plot(errx, mse_stddev, label = 'Reconstruction from stddev')
    axs[1].plot(errx, mse_profile, label = 'Full reconstruction')
    axs[1].set_title('MSE between real profile and reconstructed.')
    axs[1].legend()
    axs[2].plot(errx, chi2_stddev, label = 'Reconstruction from stddev')
    axs[2].plot(errx, chi2_profile, label = 'Full reconstruction')
    axs[2].set_title('Chi2 between real profile and reconstructed.')
    axs[2].legend()

def generate_normal_dataframe(bin_width, y_pred_stddev):
    xaxis = np.arange(-0.01, 0.01, bin_width)
    normal_profiles = []
    for i in range(len(y_pred_stddev)):
        normal_dist = stats.norm.pdf(xaxis, 0, y_pred_stddev[i])
        normal_profiles.append(normal_dist)

    normal_df = pd.DataFrame(normal_profiles)
    return normal_df


def plot_sum(bin_width, X_test, y_test_profile, y_pred_profile, y_pred_stddev):
    #This is a normal dist with a sigma as predicted.
    normal_df = generate_normal_dataframe(bin_width, y_pred_stddev)

    factor_X = (X_test.loc[:, 0:].sum(axis=0)*bin_width).sum()
    factor_test = (y_test_profile.loc[:, 0:].sum(axis=0)*bin_width).sum()
    factor_pred = (y_pred_profile.sum(axis=0)*bin_width).sum()
    factor_normal = (normal_df.sum(axis=0)*bin_width).sum()

    normalised_X =  X_test.loc[:, 0:].sum(axis=0)/factor_X
    normalised_test = y_test_profile.loc[:, 0:].sum(axis=0)/factor_test
    normalised_pred = y_pred_profile.sum(axis=0)/factor_pred
    normalised_normal = (normal_df/factor_normal).sum()


    plt.figure(figsize=(10, 10), dpi=80)
    xaxis = np.arange(-0.01, 0.01, bin_width)

    plt.plot(xaxis, normalised_X, label = 'X (observed/distorted profiles')
    plt.plot(xaxis, normalised_test, label = 'y (real) profile' )
    plt.plot(xaxis, normalised_pred, label = 'full predicted profile')
    plt.plot(xaxis, normalised_normal, label = 'Normal curve from predicted std dev')
    
    plt.legend()
    plt.title('Sum of all profiles')

def plot_errors(bin_width, X_test, y_test_profile, y_pred_profile, y_pred_stddev):

    #This is a normal dist with a sigma as predicted.
    normal_df = generate_normal_dataframe(bin_width, y_pred_stddev)
    factor_X = (X_test.loc[:, 0:].sum(axis=0)*bin_width).sum()
    factor_test = (y_test_profile.loc[:, 0:].sum(axis=0)*bin_width).sum()
    factor_pred = (y_pred_profile.sum(axis=0)*bin_width).sum()
    factor_normal = (normal_df.sum(axis=0)*bin_width).sum()

    normalised_X =  X_test.loc[:, 0:].sum(axis=0)/factor_X
    normalised_test = y_test_profile.loc[:, 0:].sum(axis=0)/factor_test
    normalised_pred = y_pred_profile.sum(axis=0)/factor_pred
    normalised_normal = (normal_df/factor_normal).sum()
    
    plt.figure(figsize=(10, 10), dpi=80)
    xaxis = np.arange(-0.01, 0.01, bin_width)

    #plt.plot(xaxis, normalised_X, label = 'X (observed/distorted profiles')
    plt.plot(xaxis, abs(normalised_test - normalised_pred), label = 'predicted profile' )
    plt.plot(xaxis, abs(normalised_test - normalised_normal), label = 'normal profile' )

    plt.legend()
    plt.title('Difference in sum of all profiles')

def plot_sum_and_errors(bin_width, X_test, y_test_profile, y_pred_profile, y_pred_stddev):
    #This is a normal dist with a sigma as predicted.
    normal_df = generate_normal_dataframe(bin_width, y_pred_stddev)

    factor_X = (X_test.loc[:, 0:].sum(axis=0)*bin_width).sum()
    factor_test = (y_test_profile.loc[:, 0:].sum(axis=0)*bin_width).sum()
    factor_pred = (y_pred_profile.sum(axis=0)*bin_width).sum()
    factor_normal = (normal_df.sum(axis=0)*bin_width).sum()

    normalised_X =  X_test.loc[:, 0:].sum(axis=0)/factor_X
    normalised_test = y_test_profile.loc[:, 0:].sum(axis=0)/factor_test
    normalised_pred = y_pred_profile.sum(axis=0)/factor_pred
    normalised_normal = (normal_df/factor_normal).sum()

    plt.figure(figsize=(10, 10), dpi=80)
    xaxis = np.arange(-0.01, 0.01, bin_width)

    plt.plot(xaxis, normalised_X, label = 'X (observed/distorted profiles')
    plt.plot(xaxis, normalised_test, label = 'y (real) profile' )
    plt.plot(xaxis, normalised_pred, label = 'full predicted profile')
    plt.plot(xaxis, normalised_normal, label = 'Normal curve from predicted std dev')

    #plt.plot(xaxis, normalised_X, label = 'X (observed/distorted profiles')
    plt.plot(xaxis, abs(normalised_test - normalised_pred), label = 'predicted profile' )
    plt.plot(xaxis, abs(normalised_test - normalised_normal), label = 'normal profile' )

    plt.legend()
    plt.title('Sum of all profiles and the error.')

def get_all_errors(bin_width, y_test_stddev, y_pred_stddev, y_pred_profile, scale=1e6, scale_y=1e6):
    residuals_fp_squared = []
    residuals_sd_squared = []
    residuals_fp = []
    residuals_sd = []
    stddev_fp = []
    xaxis = np.arange(-0.01, 0.01, bin_width)
    centers = np.arange(-0.009975, 0.009975, bin_width)
    #y_pred_profile[0][0:399]
    #y_pred_profile[0][1:400]

    for stindex in range(len(y_test_stddev)):
        #finding the stddev of the full profile reconstruction:
        mean = np.average(centers, weights=y_pred_profile[stindex][0:399])
        var = np.average((centers - mean)**2, weights=y_pred_profile[stindex][0:399])
        std = np.sqrt(var)
        stddev_fp.append(std*scale)
        #Residuals of FP std dev (calculated above), and SD approach.
        full_profile_residual = abs(y_test_stddev.iloc[stindex][0]*scale_y - std*scale)
        #std_dev_residual = abs(y_test_stddev.iloc[stindex][0]*scale_y - y_pred_stddev[stindex][0]*scale)
        std_dev_residual = abs(y_test_stddev.iloc[stindex][0]*scale_y - y_pred_stddev[stindex]*scale)
        
        
        #Append to vectors the residual, and residual squared.
        residuals_fp.append(full_profile_residual)
        residuals_sd.append(std_dev_residual)
        
        #Residual squared.
        residuals_fp_squared.append(full_profile_residual**2)
        residuals_sd_squared.append(std_dev_residual**2)
        
    dictionary_residuals = {'Real Std Dev': y_test_stddev[0]*scale_y, 'Predicted from profile':stddev_fp, 'Predicted from stddev':np.ravel(y_pred_stddev)*scale,'Residual of FP approach':residuals_fp, 'Residual of SD approach': residuals_sd,'Residual squared FP approach':residuals_fp_squared, 'Residual squared SD approach': residuals_sd_squared}
    df_residuals = pd.DataFrame(dictionary_residuals)
    
    print(tabulate(df_residuals, headers='keys', tablefmt='plain'))
    print('\n\nMeans of above table:')
    print(df_residuals.mean())
    print('\n\nStd Dev of above table:')
    print(df_residuals.std())
    print('\n\nMax of above table:')
    print(df_residuals.max())
    print('\n\nMin of above table:')
    print(df_residuals.min())
    print('\nMSE of FP approach:',  sum(residuals_fp_squared) / float(len(residuals_fp_squared)))
    print('MSE of SD approach:',sum(residuals_sd_squared) / float(len(residuals_sd_squared)))   

    return df_residuals, stddev_fp

def plot_scatters(y_test_stddev, y_pred_stddev, stddev_fp):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
    fig.suptitle('Scatter plot of predicted vs true stddevs')

    axs[0].scatter(y_test_stddev, y_pred_stddev)
    axs[0].set_xlabel('Real Value')
    axs[0].set_ylabel('Predicted Value')
    axs[0].set_title('Using SD approach')

    axs[1].scatter(y_test_stddev, stddev_fp)
    axs[1].set_xlabel('Real Value')
    axs[1].set_ylabel('Predicted Value')
    axs[1].set_title('Using FP approach')

    print('Left plot below:')
    try:
        print('Pearsons R corr for SD approach:', scipy.stats.pearsonr(np.array(y_test_stddev[0]), np.array(y_pred_stddev))[0][0])
    except:
        print('Pearsons R corr for SD approach:', scipy.stats.pearsonr(np.array(y_test_stddev[0]), np.array(y_pred_stddev).ravel())[0])
    print('Spearman R corr for SD approach:', scipy.stats.spearmanr(np.array(y_test_stddev[0]), np.array(y_pred_stddev))[0])
    print('Kendall Tau R corr for SD approach:', scipy.stats.kendalltau(np.array(y_test_stddev[0]), np.array(y_pred_stddev))[0])
    print('\n')
    
    print('Right plot below:')
    print('Pearsons R corr for FP approach:', scipy.stats.pearsonr(np.array(y_test_stddev[0]), np.array(stddev_fp))[0])
    print('Spearman R corr for FP approach:', scipy.stats.spearmanr(np.array(y_test_stddev[0]), np.array(stddev_fp))[0])
    print('Kendall Tau R corr for FP approach:', scipy.stats.kendalltau(np.array(y_test_stddev[0]), np.array(stddev_fp))[0])

def percent_errors(y_test_stddev, y_pred_stddev, stddev_fp):
    percent_errors_sd = (abs(y_test_stddev-y_pred_stddev)/y_pred_stddev) * 100
    percent_errors_fp = (abs(y_test_stddev[0] - np.array(stddev_fp)*1e-6) / (np.array(stddev_fp)*1e-6)) * 100

    dictionary_percents = {'From SD': percent_errors_sd[0], 'From FP':percent_errors_fp} 
    df_percents = pd.DataFrame(dictionary_percents)

    print('Mean of percentage errors:')
    print(df_percents.mean())
    print('\n')
    print('Std dev of percentage errors:')
    print(df_percents.std())
    print('\n')
    print('Max of percentage errors:')
    print(df_percents.max())
    print('\n')
    print('Min of percentage errors:')
    print(df_percents.min())

    return df_percents


def rate_model_sd(test, predicted, model_name):
    residuals_sd_squared = []
    residuals_sd = []

    for stindex in range(len(test)):
        #Residuals of FP std dev (calculated above), and SD approach.
        #std_dev_residual = abs(y_test_stddev.iloc[stindex][0]*scale_y - y_pred_stddev[stindex][0]*scale)
        std_dev_residual = abs(test.iloc[stindex][0] - predicted[stindex])

        #Append to vectors the residual, and residual squared.
        residuals_sd.append(std_dev_residual)

        #Residual squared.
        residuals_sd_squared.append(std_dev_residual**2)

    #dictionary_residuals = {'Real Std Dev': y_test_stddev[0]*scale_y, 'Predicted from profile':stddev_fp, 'Predicted from stddev':np.concatenate(y_pred_stddev, axis=0)*scale,'Residual of FP approach':residuals_fp, 'Residual of SD approach': residuals_sd,'Resdual squared FP approach':residuals_fp_squared, 'Residual squared SD approach': residuals_sd_squared}
    dictionary_residuals = {'mean of residuals': np.mean(residuals_sd),'std of residuals': np.std(residuals_sd),'r2 score':r2_score(test, predicted),'MSE': np.mean(residuals_sd_squared)}

    df_residuals = pd.DataFrame(dictionary_residuals, index=[model_name])

    print(tabulate(df_residuals, headers='keys', tablefmt='plain'))
    
    
def plot_profile(X, ystddev, yhists, bin_width, index=0):
    #Plotting the profile and its prediction. 
    xaxis = np.arange(-0.01, 0.01, bin_width)

    #This is the real profile - we need to normalise so the area underneath is 1.
    real_profile = yhists.iloc[index]
    normalised_real_profile = real_profile*(1/((real_profile * bin_width).sum()))

    #This is a normal dist with a sigma as predicted.
    normal_dist = stats.norm.pdf(xaxis, 0, ystddev.iloc[index])

    #Distorted profile.
    distorted_profile = X.iloc[index]*(1/((X.iloc[index] * bin_width).sum()))
    plt.figure(figsize=(30, 10), dpi=80)

    plt.plot(xaxis, normalised_real_profile, label = 'Real Profile')
    plt.plot(xaxis, normal_dist, label = 'Normal distribution from given std dev')
    plt.plot(xaxis, distorted_profile.loc[0:], label = 'Measured/Distorted profile')
    plt.legend()

    
def dump_model_to_pickle(model, name, append):
    pickle.dump(model, open('Model_' + name + '_' + append + '.pkl', 'wb'))
    
def load_model_from_pickle(name, append):
    pickled_model = pickle.load(open('Model_' + name + '_' + append + '.pkl', 'rb'))
    return pickled_model



###############################################
############## CURVE FITTING ##################
###############################################
# q-Gaussian stuff
def qexp(q, x):
    vec = []
    if(q==1):
        return np.exp(x)
    for element in x:
        if(1+(1-q)*element > 0):
            vec.append( (1 + (1-q)*element) ** (1/(1-q)) )
        elif(1+(1-q)*element <= 0):
            vec.append(0 ** (1/(1-q)))
    #print(vec)
    return vec
    
def cfactor(q):
    if(q < 1):
        return ( 2 * np.sqrt(np.pi) * scipy.special.gamma(1/(1-q)) ) / ( (3-q) * np.sqrt(1-q) * scipy.special.gamma((3-q)/(2*(1-q))) )
    elif(q == 1):
        return np.sqrt(np.pi)
    elif(q < 3):
        return ( np.sqrt(np.pi) * scipy.special.gamma((3-q)/(2*(q-1))) ) / ( np.sqrt(q-1) * scipy.special.gamma(1/(q-1)) ) 
    else:
        return 0

def qgauss(x, q, beta):
    #print(cfactor(q))
    return np.array([i * np.sqrt(beta) / cfactor(q) for i in qexp(q, -1 * beta * (x ** 2))])

    

def testfunction():
    return pd.DataFrame()



###############################################
###### NXCALS AND SPARK FRAMEWORK #############
###############################################
import imageio, os
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
from pyspark.sql import Row

def gif_that_df(df, k, name, SPLIT_COUNT, SPLIT_SIZE):
    """
    Function to turn a series of profiles into a gif. The dataframe must contain profiles in 2014 bins, and have at least 90 elements.
    Your directory must contain a folder titled 'gifs'.
    :df: dataframe of profiles you would like to convert to gif. Must have final 1024 columns at profiles[0], ..., profiles[1023]
    :k: Which supercycle to take within the dataframe
    :name: What to save the gif as
    return: NULL
    """
    if(len(df) == 0):
        print('Your dataframe is empty, maybe the supercycle was dropped')
        return -1
    filenames = []
    for i in range(k*SPLIT_COUNT,(k+1)*SPLIT_COUNT):
        centers = np.linspace(-28, 28, SPLIT_SIZE)
        plt.plot(centers, df.iloc[i][-SPLIT_SIZE:], label=str(df.iloc[0].current)+'A')
        plt.legend()

        plt.title(f'{i}')

        # create file name and append it to a list
        filename = f'gifs/{i}.png'
        filenames.append(filename)

        # save frame
        plt.savefig(filename)
        plt.close()# build gif
    with imageio.get_writer((name+'.gif'), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)

        
        
def initate_df_from_datas(profile_data, magnet_data, cut=0.5):
    """
    This function will take both your profile and magnet data and create the final dataframe with your data formatted nicely to be used. 
    :profile_data: profiledata from NXCALS
    :magnet_data: magnetdata from NXCALS
    :cut: How much above and below the mean number of hits to ignore.
    return: final_df: a spark dataframe with each profile, its cycle and supercycle number, and the current of the magnet at the time the profile was recorded
    """
    SPLIT_COUNT = 90
    SPLIT_SIZE = 1024


    #Select relevent data from profile structure and move into its own column.
    spark_p = profile_data.select("profiles", '__record_timestamp__', 'profileTotalNumberOfEvents')
    spark_p = spark_p.withColumn("dimensions", F.col("profiles").getField("dimensions") )
    spark_p = spark_p.withColumn("profiles", F.col("profiles").getField("elements") )
    spark_p = spark_p.withColumn("profileTotalNumberOfEvents", F.col("profileTotalNumberOfEvents").getField("elements") )
    spark_p = spark_p.filter((spark_p.dimensions[0] == 90) & (spark_p.dimensions[1] == 1024))
    spark_p = spark_p.drop('dimensions')

    #Give supercycle number
    spark_p = spark_p.withColumn("supercycle", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1 )

    #Drops any supercycles with less than half the average beam strength
    spark_p = spark_p.withColumn('sum1', sum([F.col('profileTotalNumberOfEvents').getItem(i) for i in range(SPLIT_COUNT)]))
    spark_p = spark_p.filter((spark_p.sum1 <= F.lit(3e6)))
    mean_low = spark_p.select(F.mean('sum1')).collect()[0][0]*cut
    spark_p = spark_p.withColumn('mean_low', F.lit(mean_low))
    mean_hi = spark_p.select(F.mean('sum1')).collect()[0][0]*(2-cut)
    spark_p = spark_p.withColumn('mean_hi', F.lit(mean_hi))

    #So here we plot the counts per supercycle before and after dropping 'weak' rows. Just to see if our cut was acceptable.
    plt.figure(figsize=(10, 10), dpi=80)
    plt.plot(spark_p.select("supercycle").toPandas()['supercycle'], spark_p.select("sum1").toPandas()['sum1'], label='Before dropping low count supercycles')
    spark_p = spark_p.filter((spark_p.sum1 <= spark_p.mean_hi))
    spark_p = spark_p.filter((spark_p.sum1 >= spark_p.mean_low))
    plt.plot(spark_p.select("supercycle").toPandas()['supercycle'], spark_p.select("sum1").toPandas()['sum1'], label='After dropping low count supercycles')
    plt.axhline(y=mean_hi, color='r', linestyle='-', label='Cut')
    plt.axhline(y=mean_low, color='r', linestyle='-', label='Cut')
    plt.legend()

    spark_p = spark_p.drop('profileTotalNumberOfEvents','sum1', 'mean')


    #Split this long vector into the sizes above.
    slices = [F.slice(F.col('profiles'), i * SPLIT_SIZE + 1, SPLIT_SIZE) for i in range(SPLIT_COUNT)]

    #Explode and rename columns. 
    spark_p = spark_p.select(F.posexplode(F.array(*slices)), F.col('__record_timestamp__'), F.col('supercycle'))
    spark_p = spark_p.withColumn("cycle", F.col("pos") )
    spark_p = spark_p.withColumn("profiles", F.col("col") )
    spark_p = spark_p.drop('pos').drop('col')

    #Merge the magnet data in.
    spark_m = magnet_data.select("value", '__record_timestamp__', )
    spark_p = spark_p.withColumn('value', F.lit(None))
    spark_m = spark_m.withColumn('profiles', F.lit(None))
    spark_m = spark_m.withColumn('cycle', F.lit(None))
    spark_m = spark_m.withColumn('supercycle', F.lit(None))

    final_df = spark_p.unionByName(spark_m)
    w = Window.orderBy('__record_timestamp__').rowsBetween(Window.unboundedPreceding, -1)
    final_df = final_df.withColumn('value', F.last('value', True).over(w)).filter(~F.isnull('profiles'))

    final_df = final_df.orderBy("supercycle","cycle")

    #rename columns for cleaner df
    final_df = final_df.withColumnRenamed('value', 'current').select('__record_timestamp__', 'supercycle', 'current', 'cycle', 'profiles')
    final_df = final_df.withColumn('current', F.round('current'))
    #final_df.describe().show()
    
    #Give new supercycle number, this will exist from 0-max number of supercycles whereas the original supercycle column will have gaps and maintain its location in the whole dataset.
    final_df = final_df.withColumn("new_supercycle", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1 )

    #Expand profiles
    dlist = final_df.columns
    final_df = final_df.select(dlist+[(F.col("profiles")[x]).alias("profiles"+str(x)) for x in range(0, SPLIT_SIZE)]).drop('profiles')

    #Show that badboy, yo pass that.
    #final_df.show(20)
    return final_df