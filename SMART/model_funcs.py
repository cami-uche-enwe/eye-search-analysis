import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import SMART_Funcs as SF
from SMARTClass import SMART

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

##################### 
# FITTING FUNCTIONS #
#####################

###################
# basic functions #
###################

# S(t)
def salient(time,t0,r): # r means rate (a in the paper)
    y = np.zeros((len(time)))
    for i,t in enumerate(time): 
        if t < t0: 
            y[i] =1 
        else: 
            y[i] = np.e**(-r*(t-t0))
    return y 

# R(t)
def relevance(time,t0,r):
    y = np.zeros((len(time)))
    for i,t in enumerate(time): 
        if t < t0: 
            y[i] = 0 
        else: 
            y[i] = 1 - np.e**(-r*(t-t0))
    return y



def full_model(t,rS,rR,t0S,t0R):
    S = salient(t,t0S,rS)
    G = relevance(t,t0R,rR)
    return S, G

def dependent_model(t,rS,t0S): # t0R and rR and now the same as rS and t0S
    S = salient(t,t0S,rS)
    G = relevance(t,t0S,rS)
    return S, G

# salience only model
def salience_model(t,rS,t0S):
    S = salient(t,t0S,rS)
    return S

# relevance only model
def relevance_model(t,rR,t0R):
    G = relevance(t,t0R,rR)
    return G


#######################
# optimizer functions #
#######################

# full model 
def optim_func(var,*args): 
    y_real_s = args[0]
    y_real_r = args[1]
    ys = full_model(timeVect,var[0],var[1],var[2],var[3])[0]
    yr = full_model(timeVect,var[0],var[1],var[2],var[3])[1]
    dif_s = np.sum((y_real_s-ys)**2) # least squares method - sqrt as chris did, results are super similar to not using sqrt, but not identical
    dif_r = np.sum((y_real_r-yr)**2)
    dif = dif_s + dif_r
    return dif

def optim_func2(var,y_real_s, y_real_r, *args): # full model but with explicit args y_real_s and y_real_r  to use to calc AIC
    y_real_s = y_real_s
    y_real_r = y_real_r
    ys = full_model(timeVect,var[0],var[1],var[2],var[3])[0]
    yr = full_model(timeVect,var[0],var[1],var[2],var[3])[1]
    dif_s = np.sum((y_real_s-ys)**2) # least squares method - sqrt as chris did, results are super similar to not using sqrt, but not identical
    dif_r = np.sum((y_real_r-yr)**2)
    dif = dif_s + dif_r
    return dif

# dependent model  
def optim_func_dependent(var,*args): 
    y_real_s = args[0]
    y_real_r = args[1]
    ys = dependent_model(timeVect,var[0],var[1])[0]
    yr = dependent_model(timeVect,var[0],var[1])[1]
    dif_s = np.sum((y_real_s-ys)**2) # least squares method - sqrt as chris did, results are super similar to not using sqrt, but not identical
    dif_r = np.sum((y_real_r-yr)**2)
    dif = dif_s + dif_r
    return dif 

def optim_func_dependent2(var, y_real_s, y_real_r, *args): # dependent model but with explicit args y_real_s and y_real_r  to use to calc AIC
    y_real_s = y_real_s
    y_real_r = y_real_r
    ys = dependent_model(timeVect,var[0],var[1])[0]
    yr = dependent_model(timeVect,var[0],var[1])[1]
    dif_s = np.sum((y_real_s-ys)**2) # least squares method - sqrt as chris did, results are super similar to not using sqrt, but not identical
    dif_r = np.sum((y_real_r-yr)**2)
    dif = dif_s + dif_r
    return dif 

# salience only model
def optim_func_saliency(var,*args): 
    y_real = args[0]
    y = salience_model(timeVect,var[0],var[1])
    dif = np.sum((y_real-y)**2) # least squares method
    return dif 
    
# relevance only model
def optim_func_relevance(var,*args): 
    y_real = args[0]
    y = relevance_model(timeVect,var[0],var[1])
    dif = np.sum((y_real-y)**2) # least squares method
    return dif 


# calculate AIC
def calc_AIC(RSS, p):
    n = len(timeVect)
    AIC = 2*((n/2) *np.log(RSS/n))+(2*p)
    return AIC 


#################
# model fitting #
#################

def MLE_fit(input_data, effect, whichmodel): 
    
    if whichmodel == 'full' or whichmodel == 'dep':
        nSubs = np.shape(input_data[0])[0]
    else:
        nSubs = np.shape(input_data)[0]

    # first do a round of Least Squares to get good starting points 
    results_salience = np.zeros((nSubs, 2)) # two vars  = salience only model
    results_relevance = np.zeros((nSubs,2)) # two vars  = relevance only model 
    results_full  = np.zeros((nSubs, 4))  # four vars  = full model 
    results_dependent = np.zeros((nSubs, 2)) # two vars = dependent model 
    AICs = np.zeros((nSubs,4)) # AIC score 

    from scipy.optimize import minimize
    for n in range(nSubs): 
        # y_real = input_data[n,:]
        
        if whichmodel == 'full':
            y_real_s = input_data[0][n,:]
            y_real_r = input_data[1][n,:]

            ####################
            # full model #######
            ####################
            
            var = [0.01,0.01,120,230]
            result = minimize(optim_func, var,args = (y_real_s, y_real_r),options={'disp': False, 'maxfun':1000},method = 'TNC',bounds = [(0,None),(0,None),(0,None),(0,None)])# 0 here in args, because of leastsquares   
            results_full[n,:] = result.x
            RSS = optim_func2(result.x, y_real_s, y_real_r, 0)
            AICs[n,0] = calc_AIC(RSS,4)
        
        elif whichmodel == 'dep':
            y_real_s = input_data[0][n,:]
            y_real_r = input_data[1][n,:]

            ####################
            # dependent model ##
            ####################
            
            var = [0.01,200]#155] 
            result = minimize(optim_func_dependent, var,args = (y_real_s, y_real_r),options={'disp': False, 'maxfun':1000},method = 'TNC',bounds = [(0,None),(0,None)])# 0 here in args, because of leastsquares   
            results_dependent[n,:] = result.x
            RSS = optim_func_dependent2(result.x, y_real_s, y_real_r, 0)
            AICs[n,1] = calc_AIC(RSS,2)
        
        elif whichmodel == 'indep':
            y_real = input_data[n,:]

            if effect == 'salience':
                #####################
                # salience model ####
                #####################
                var =  [0.01,155]# 
                result = minimize(optim_func_saliency, var,args = (y_real),options={'disp': False, 'maxfun':1000},method = 'TNC',bounds = [(0,None),(0,None)])# 0 here in args, because of leastsquares
                results_salience[n,:] = result.x
                RSS = optim_func_saliency(result.x,y_real,0)
                AICs[n,2] = calc_AIC(RSS,2)
            
            elif effect == 'relevance':
                ###################
                # relevance model #
                ###################
                
                var =  [0.01,230]
                result = minimize(optim_func_relevance, var,args = (y_real),options={'disp': False, 'maxfun':1000},method = 'TNC',bounds = [(0,None),(0,None)])# 0 here in args, because of leastsquares
                results_relevance[n,:] = result.x
                RSS = optim_func_relevance(result.x,y_real,0)
                AICs[n,3] = calc_AIC(RSS,2)
    
    if whichmodel == 'full':
        return results_full, AICs
    
    elif whichmodel == 'dep':
        return results_dependent, AICs
    
    elif whichmodel == 'indep':
        if effect == 'salience':
            return results_salience, AICs
        elif effect == 'relevance':
            return results_relevance, AICs
