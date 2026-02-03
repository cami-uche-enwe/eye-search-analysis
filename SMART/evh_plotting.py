# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:00:41 2019

Datafiles should be saved in a folder called 'data'
You need to create an empty folder called 'exported_data'
You need to create an empty folder called 'plots', with the following subfolders: 
    exp1
    exp2
    dataset1
    dataset2
    
    figures are automatically saved in the plots/* folder 
@author: Elle van Heusden
"""

#import numpy as np 
import matplotlib.pyplot as plt 
import os
import evh_functions



plt.close('all')

# loop over the experiments and datasets 
adds = ['exp1', 'exp2','dataset1', 'dataset2']
for add in adds: 
    e = evh_functions.exp_class() 
    e.add = add 
    e.plt_folder = add
    # get the data (cleaned)
    df = e.data_cleanup()  # this used the e.add variable to open up the right file from the exported_data folder 
    filename = os.getcwd() + '/exported_data/limbo_' + add # where the needed dataframe for analysis will be stored 
    # put the data in a dataframe that the functions need 
    SMART_df = e.make_SMART_df(df, filename)
    ########################################################
    ######### make saliency x relevance plot ##############
    # this also output the 'calc_NSNT' which we need for a later function 
    # c panels in the figures 
    ########################################################

    plt_name = os.getcwd() + '/plots/' +e.plt_folder  + '/saliency_relevance_effect'
    calc_NSNT = e.make_saliency_relevance_plot(filename, plt_name )
    plt.close('all')
    plt_name = os.getcwd() + '/plots/' +e.plt_folder  + '/prop_to_nsd_allsubs'
    
    ########################################################################################################
    #### this function makes the limbo plot and outputs the results (parameters and AIC) of all the different fits
    # d panels in the figures
    #################################################################
    [results_model,results_saliency, results_relevance, results_dependent,results_stable, AICs] =  e.make_limbo_plot(filename,df,e, plt_name, calc_NSNT,color =  e.colors[3])    
    plt.close()
    ##########################################
    # print this information for in the tables
    ##########################################
    print(results_model)
    print(results_saliency)
    print(results_relevance)
    print(results_stable)
    print(results_dependent)
    print(AICs)
    ########################################################################################
    # with the results from the fit, reconstruct the contributions of saliency and relevance
    # e panels in the figures 
    ########################################################################################
    e.make_model_plot(results_model[0])
    plt.close('all')
    ######################
    # this function creates the 'standard' plots, i.e. to target/ salient item as a function of saccade latency 
    # b panels in the figures 
    #####################
    plt_name = os.getcwd() + '/plots/' +e.plt_folder  + '/subplot_standard_plots'
    e.make_paper_subplot(filename, plt_name)
    plt.close('all')