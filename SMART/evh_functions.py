# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:57:07 2019

@author: ehn268
"""
import numpy as np 
import pandas as pd 
import pickle 
import SMART_Funcs as SF
import matplotlib.pyplot as plt 
import os
from SMARTClass import SMART



class exp_class: 

    # some handy variables 
    krnSize = 10 # kernelsize 
    stepTime = 1 # steps of 1 ms 
    nPerm = 1000  # amount of permutations done in the stats analysis 
    baseline = 0.5 # if tested against baseline 
    sigLevel = 0.05  # significant is p lower than this value 
    colors = ['red','#0089cf','#559834', '#810d70','orange','#eb6734','#14691f', '#47a654','#0d4714']  # 0red, 1blue, 2green, 3purple,4yellow-orange,5orange, 6darkgreen, 7lightgreen, 8darkergreen     #['red', '#0089cf', '#306a30','orange','purple']
              
              

    
    def make_SMART_df(self,df, filename): 
        # PREPRARING THE DATA AS SUCH THAT IS IS IN THE RIGHT FORMAT FOR THE ANALYSIS 
        # outputs a dataframe in the way the SMART function expect it so be 
        df_SMART = pd.DataFrame() 
        nSubjects = len(np.unique(df.subject))
        for i, ecc in enumerate([0]): # this loop is redundant 
                    onset_list_salient = []
                    target_list_salient = []
                    
                    onset_list_nonsalient = []
                    target_list_nonsalient = []

                    salient_target_to_salient = []
                    nonsalient_target_to_salient = []
                    
                    salient_target_to_salient_t = []
                    nonsalient_target_to_salient_t = []
                                        
                    # LIMBO 
                    onset_list_to_dist_nonsalient = []
                    distractor_list_nonsalient = []
                    
                    #Wieske analyse 
                    distractor_list_eqsalient = []
                    onset_list_to_dist_eqsalient = []
                    
                    
                    for pp in np.unique(df.subject): 
                        # fixed so that this works with one subject 
                        if nSubjects ==1: 
                            pp = np.unique(df.subject)[0]
                            
                        
                        pp_sel = (df.subject == pp) & ((df.to_target ==1) | (df.to_dist ==1))
                        df_tmp = df[pp_sel]
                        
                        
                        # SELECTION OF NON SLIENT TARGET NST
                        sel = (np.array(df_tmp.saliency) == 0) #& (np.round(df_tmp.condition) == round(ecc))
                        #sel_condition = (np.round(df_tmp.condition)== np.round(ecc))
                        onset_list_nonsalient.append(np.array(df_tmp.onsets[sel]))
                        target_list_nonsalient.append(np.array(df_tmp.to_target[sel]))
                        
                        
                        # LIMBO SELECTION, NSNT TRIALS  ##
                        # saliency is 1 here, because that means the target is salient so the dist is nonsalient 
                        sel = (np.array(df_tmp.saliency) == 1) 
                        onset_list_to_dist_nonsalient.append(np.array(df_tmp.onsets[sel]))
                        distractor_list_nonsalient.append(np.array(df_tmp.to_dist[sel]))   # CRITICALLY, THIS IS IF THE DISTRACTOR WAS SELECTED 
                        
                        
                        # WIESKE SELECTION #
                        sel = np.isnan(df.saliency)
                        distractor_list_eqsalient.append(np.array(df_tmp.to_dist[sel]))
                        onset_list_to_dist_eqsalient.append(np.array(df_tmp.onsets[sel]))
                        
                        
                        # SELECTION OF SALIENT TARGET (ST)
                        sel = (np.array(df_tmp.saliency) == 1) 
                        onset_list_salient.append(np.array(df_tmp.onsets[sel]))
                        target_list_salient.append(np.array(df_tmp.to_target[sel])) # AND HERE, THE TARGET IS SELECTED 
                        
                        
                        
                        # SEELCTION OF SALIENT TARGET (ST)
                        sel = (np.array(df_tmp.saliency) == 1) 
                        salient_target_to_salient.append(np.array(df_tmp.to_salient[sel]))
                        salient_target_to_salient_t.append(np.array(df_tmp.onsets[sel]))
                        
                        # SELECTION OF SALIENT NON TARGET (SNT)
                        sel = (np.array(df_tmp.saliency) == 0) #distractor is most salient item 
                        nonsalient_target_to_salient.append(np.array(df_tmp.to_salient[sel]))
                        nonsalient_target_to_salient_t.append(np.array(df_tmp.onsets[sel]))

                    # saving information (there is some redundancy here, not everything is used later on)
                    df_SMART['onset_nonsalient' + '_' + str(i)] = onset_list_nonsalient 
                    df_SMART['onset_salient' + '_' + str(i)] = onset_list_salient
                    df_SMART['to_target_salient' + '_' + str(i)] = target_list_salient 
                    df_SMART['to_target_nonsalient'+ '_' + str(i)] = target_list_nonsalient
                    df_SMART['salient_to_salient'+ '_' + str(i)] = salient_target_to_salient
                    df_SMART['non_salient_to_salient'+ '_' + str(i)] = nonsalient_target_to_salient
                    df_SMART['salient_to_salient_t'+ '_' + str(i)] = salient_target_to_salient_t
                    df_SMART['non_salient_to_salient_t'+ '_' + str(i)] = nonsalient_target_to_salient_t                    
                    # limbo information 
                    df_SMART['to_dist_non_salient'+ '_' + str(i)] = distractor_list_nonsalient
                    df_SMART['to_dist_non_salient_t'+ '_' + str(i)] = onset_list_to_dist_nonsalient
                    # extra analysis for Wieske 
                    df_SMART['equally_salient_to_dist_non_salient'+ '_' + str(i)] = distractor_list_eqsalient
                    df_SMART['equally_salient_to_dist_non_salient_t'+ '_' + str(i)] = onset_list_to_dist_eqsalient
                    
                    
        with open(filename, 'wb') as handle: 
            pickle.dump(df_SMART, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return df_SMART
        
        
    
    # def normalize_data(self,df,dv): 
    #     nSubjects = len(df)
    #     new_values = np.zeros(np.shape(dv))
    #     for subject in range(nSubjects): 
    #         for t in range(len(self.timeVect)):  
    #                 grand_average = np.mean( [(df['difference_'  + self.add][s][t]) for s in range(nSubjects)])
    #                 subject_average = (df['difference_'  +  self.add][subject][t])
    #                 new_values[subject,t] = dv[subject,t] - subject_average + grand_average 
                
                     
    #     return new_values
    
    
    def data_cleanup(self, drop_ecc = False): 
        # this function cleans up the data and adds some variables to the df
       
        # open file we need for plotting 
        fname = 'data/data_' + self.add + '.pickle' # open the datafile 
        df = pd.read_pickle(fname)
        
        # get information from the data 
        self.nSubs = len(np.unique(df.subject))
        self.subject_dir = [str(i) for i in range(self.nSubs)]
        self.targetDistances = np.unique(df.condition)
        
        # add this variable 
        df['to_salient'] = ((df.saliency == 1) & (df.to_target ==1)) | ((df.saliency == 0) & (df.to_target ==0))
        
        ################################ 
        ### clean the data here ########
        ###############################
        # get rid of the values that seem very unrealistic (analysis error)
        l0 = len(df)
        df = df[df.onsets > 80]
        #### get rid of 5% of the most extreme data to avoid analysis issues 
        df = df.reset_index()
        indeces = np.argsort(df.onsets)
        cut_off = int(0.025*len(df.onsets))
        remove = list(indeces[:cut_off ]) + list(indeces[-cut_off:])
        df = df.drop(remove)
        df = df.reset_index() 
        
        self.minTime = np.min(df.onsets)
        self.maxTime = np.max(df.onsets)
        if self.maxTime > 500: 
            self.maxTime = 500 
        
        df = df[df.onsets > self.minTime]
        df = df[df.onsets < self.maxTime]
        l1 = len(df)
        loss = (l0-l1)/float(l0) *100
        print(self.add)
        print('number of trials before' )
        print(l0)
        print('Data loss due to latency is: ' + str(loss) + ' %' )
        
      
        # get rid of the practise trials 
        if self.add == 'exp1': 
            df = df[df.trial > 50]
        if self.add == 'dataset1':
            df = df[df.trial > 36]
        if self.add == 'dataset2':
            df = df[df.trial > 48]
        else: 
            df = df[df.trial > 24]
            
        # add selected item var
        # we need to get the information of where you went to on a trial in the df 
        sel1 = (df.saliency == 1) & (df.to_dist==1) # non-salient distractor (target = salient)
        sel2 = (df.saliency == 0) & (df.to_target==1) # non-salient target 
        sel3 = (df.saliency == 0) & (df.to_dist==1) # salient distractor 
        sel4 = (df.saliency == 1) & (df.to_target==1) # salient target 
        
        # init this var 
        df['selected_item'] = np.zeros(len(df))
        df.loc[sel1,'selected_item'] = 'nsd'
        df.loc[sel2,'selected_item'] = 'nst'
        df.loc[sel3,'selected_item'] = 'sd'
        df.loc[sel4,'selected_item'] = 'st'
        return df 

    def make_limbo_plot(self,filename,df,e, plt_name,calc_NSNT, trial_selection = None, color =  'red', host = None, title = None, legend = True, ylim_KDE = 400, ylim = 0.8  ): 
        
        import time
       
        df_tmp = df.copy()
        if trial_selection:  # you can also run this function on a selection of trials    
            df_tmp = df_tmp.iloc[trial_selection]
        
        df_SMART = self.make_SMART_df(df_tmp,filename) #THIS FUNCTION CREATES 'FILENAME', and returns df_SMART 
        time.sleep(1) # just to make sure it is actually finished, and does not use an old file with the same name  
        
    
        # plotting part 
        if not host: 
            fig = plt.figure()
            host = fig.add_subplot(111)
        
        # doing the analysis as described in van Leeuwen et al. 
        res = SMART(filename, 'to_dist_non_salient_0', 'to_dist_non_salient_t_0')
        res.runSmooth(self.krnSize, self.minTime, self.maxTime, self.stepTime)
        res.runPermutations(self.nPerm, 0.5)  # test where data is sig different from 0.5 
        res.runStats(self.sigLevel)
        print(res.nPP)
        self.timeVect = res.timeVect  # set this variable to use later 


        # plotting part 
        if not host: 
            fig = plt.figure()
            host = fig.add_subplot(111)
        plt.ylabel('P(NSNT)', fontsize = 18)
        plt.xlabel('Saccade latency (ms)', fontsize = 18)

                
        # make the line thicker where the difference is significant  
        for ind, i in enumerate(res.sigCL):
            if res.sumTvals[ind] >= res.sigThres:  
                    host.plot(self.timeVect[res.sigCL[ind]], res.weighDv1Average[res.sigCL[ind]], color = color, linewidth =5  )
                    
        # plot the actual data here (otherwise fit is messed up)
        host.fill_between(self.timeVect,res.weighDv1Average - res.conf95, res.weighDv1Average + res.conf95, color=color, alpha=0.25,zorder = 2)
        host.plot(self.timeVect, res.weighDv1Average, color = color,zorder = 2, linewidth = 2)
         # report something about the peaks 
        print('real data peaked at ' + str(self.timeVect[np.argsort(res.weighDv1Average)[-1]]) + ' ms with performance of ' + str(np.max(res.weighDv1Average)))
        
        
        ####################################
        # getting the orange line in there 
        # calc_NSNT is the variable we took from the make_saliency_relevance_plot function 
        # they are the smoothed averages and weights of the saliency and relevance effects, 
        # which we need to do the calculation 
        ######################################
        
        saliency_effect = calc_NSNT['saliency']
        saliency_weights = calc_NSNT['saliency_weights']
        relevance_effect = calc_NSNT['relevance']
        relevance_weights = calc_NSNT['relevance_weights']
        
        #############################################
        ###### variables needed for plotting  #############
        weighDv1_diff = SF.weighArraysByColumn(saliency_effect, saliency_weights)
        saliency = np.nansum(weighDv1_diff, axis=0)
        weighDv1_diff = SF.weighArraysByColumn(relevance_effect, relevance_weights)
        relevance = np.nansum(weighDv1_diff, axis=0)
        calc_NSNT_line =  (1-saliency) * (1-relevance) *0.5
        #################################################
        
        #################################################
        ####### variables needed for the stats  
        ############################################
        calc_NSNT_per_sub = (1-saliency_effect) * (1-relevance_effect)*0.5 
        weights_per_sub = np.sum([saliency_weights, relevance_weights], axis = 0)
        
        # see where these lines are different from eachother 
        [t_values1, p_values1] = SF.weighted_ttest_rel(calc_NSNT_per_sub,res.smooth_dv1,weights_per_sub,res.weights_dv1) #cond1, cond2, weights1, weights2

        # now do the permutation part 
        df1 = pd.DataFrame({'diffs_dv': list(calc_NSNT_per_sub), 'condition': np.ones(self.nSubs), 'sum_weights': list(weights_per_sub) })
        df2 = pd.DataFrame({'diffs_dv': list(res.smooth_dv1),'condition': 2*np.ones(self.nSubs),'sum_weights': list(res.weights_dv1)})
        df_new = pd.concat([df1,df2])
         
        permDistr1 = self.permute(df_new, self.nPerm)
        # determine the threshold 
        sigThres1 = np.percentile(permDistr1, 95)
        # get cluster 
        clusters, indx = SF.getCluster(p_values1 < 0.05) 
        # caculate sum of t-vals (only for those clusters where it is lower than 0.05 )
        sigCL1 = [indx[i] for i in range(len(clusters)) if clusters[i][0] == True]
        sumTvals1 = [np.sum(abs(t_values1[i])) for i in sigCL1]
        # plot where they are different (which is never)
        for ind, i in enumerate(sigCL1):
                            if sumTvals1[ind] >= sigThres1:  #only if it is really significant 
                                    x = self.timeVect[sigCL1[ind]]
                                    y  = [0.1]* len(x)
                                    plt.plot(x,y,'orange', linewidth = 5)
                                    plt.plot(x,y,'purple',linestyle = '--', linewidth = 5, dashes = [2,2])
                                    
        # get p values to report in the paper 
        from scipy.stats import percentileofscore
        obs_clusters = sumTvals1
        percentile_values = [percentileofscore(permDistr1, obs_cluster) for obs_cluster in obs_clusters]
        p_values = 1 - np.array(percentile_values)/100.
        print('Cluster p-values: ', p_values)      
                           
       
        
        # actually plot the orange line
        host.plot(self.timeVect, calc_NSNT_line, color = 'orange', linestyle =   (0, (1, 1)), linewidth = 5)
        # calculate and print AIC value of the calculated model 
        RSS_fitted = np.sum((res.weighDv1Average - calc_NSNT_line)**2) 
        AIC_fitted = self.calc_AIC(RSS_fitted,0)
        print('orange line AIC: ' + str(AIC_fitted))
    
        # add title if need be 
        if title: 
            host.set_title(title, fontsize = 16)
        # do all the fits, plot the full model it in the figure as well 
        if not trial_selection: 
            [results_model,results_saliency, results_relevance, results_dependent,results_stable, AICs] = self.MLE_fit(np.atleast_2d(res.weighDv1Average))
            fitted = self.chance_to_NSNT(self.timeVect, results_model[0][0],results_model[0][1],results_model[0][2],results_model[0][3])
            host.plot(self.timeVect,fitted, color = 'black', linestyle =  (0, (1, 1)), linewidth =3,zorder = 2) 
            
        # report something on the peaks of the fitted line 
        print('FIT peaked at ' + str(self.timeVect[np.argsort(fitted)[-1]]) + ' ms with performance of ' + str(np.max(fitted)))

        # resume plotting 
        host.tick_params(axis='both', which='major', labelsize=14)
        host.tick_params(axis='both', which='minor', labelsize=12)
       
        
        params = {'text.latex.preamble' : [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
        plt.rcParams.update(params) 
        if (not trial_selection): 
            if legend: 
                host.legend(['Data', 'Model fit', 'Predicted from ' + r'$\Delta$'+ 'P'], fontsize = 14, frameon = False, loc =1, ncol =1 )                
                ax = plt.gca()
                leg = ax.get_legend()
                leg.legendHandles[1].set_color(('black')) # model 
                leg.get_lines()[1].set_linewidth(3)
                leg.get_lines()[1].set_linestyle( (0, (1, 1)))
                leg.legendHandles[2].set_color(('orange')) # model 
                leg.get_lines()[2].set_linewidth(3)
                leg.get_lines()[2].set_linestyle( (0, (2, 2)))

#                
#           
        # init these for kde later
        # remove the top and right lines 
        # Hide the right and top spines
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        par2 = host.twinx()
        par2.set_axisbelow(True)    
        host.hlines(0.50, self.timeVect[0],self.timeVect[-1])  # add line at 0.5
        host.set_ylim(0,ylim)

            
        host.set_xlim(self.timeVect[0],self.timeVect[-1])        
#       # Plot kernel density estimation KDE   
        sTimes1, unqT1, countT1 = SF.getKDE(np.hstack(res.data[res.t1]),self.timeVect, self.krnSize) 
        countT1 = self.smooth(countT1,3)
        par2.set_ylim(1, ylim_KDE)
        par2.plot(self.timeVect, sTimes1, '--', alpha = 1,color = 'black',zorder = 1 )
        par2.bar(unqT1, countT1, color='black', alpha = 0.5,zorder = 1)
        par2.yaxis.set_ticks([])
        par2.yaxis.set_ticks(np.linspace(0,np.max(np.hstack([sTimes1])),2, dtype=int))
        host.set_zorder(par2.get_zorder()+1)
        host.patch.set_visible(False)
        
        par2.spines['right'].set_visible(False)
        par2.spines['top'].set_visible(False)

         
        figure = plt.gcf() # get current figure
        figure.set_size_inches(6,4)
        figure.tight_layout()
        plt.savefig(plt_name+ '.png')
        
        return [results_model,results_saliency, results_relevance, results_dependent,results_stable, AICs] 

    def make_standard_plot(self, res,effect): 
          plt.figure(num=3)
          print(effect)
          if effect == 'saliency': 
              ylabel = 'Proportion to target'
          else: 
              ylabel = 'Proportion to salient'
          self.af_make_plot(res, smooth = True,  color1 = '#0089cf', color2 = '#306a30',ylabel = ylabel)                 
          figure = plt.gcf() # get current figure
          figure.set_size_inches(6,3)
          figure.tight_layout()                  
          plt.savefig(os.getcwd() + '/plots/'+ self.add + '/' + effect + self.add)
          plt.close()  
    
    def calc_AIC(self, RSS, p):
        n = len(self.timeVect)
        AIC = 2*((n/2) *np.log(RSS/n))+(2*p)
        return AIC 
    
    def MLE_fit(self, input_data): 
        
        nSubs = np.shape(input_data)[0]
        
        # first do a round of Least Squares to get good starting points 
        results_saliency = np.zeros((nSubs, 2)) # two vars  = saliency only model
        results_relevance = np.zeros((nSubs,2)) # two vars  = relevance only model 
        results_model  = np.zeros((nSubs, 4))  # four vars  = full model 
        results_dependent = np.zeros((nSubs, 2)) # two vars = interdependent model 
        results_stable = np.zeros((nSubs,1)) # one var      = time invariant 
        AICs = np.zeros((nSubs,5)) # AIC score 
        from scipy.optimize import minimize
        for n in range(nSubs): 
            y_real = input_data[n,:]
            
            ###################
            # full model fit #######
            ####################
            
            var = [0.01,0.01,155,230] # QUESTION how are these numbers decided?
            result = minimize(self.optim_func, var,args = (y_real),options={'disp': False},method = 'TNC',bounds = [(0,None),(0,None),(0,None),(0,None)])# 0 here in args, because of leastsquares   
            results_model[n,:] = result.x
            RSS = self.optim_func(result.x,y_real,0)
            AICs[n,0] = self.calc_AIC(RSS,4)
            

            #####################
            # saliency model ###
            #####################
            var =  [0.01,155]# 
            result = minimize(self.optim_func_saliency, var,args = (y_real),options={'disp': False},method = 'TNC',bounds = [(0,None),(0,None)])# 0 here in args, because of leastsquares
            results_saliency[n,:] = result.x
            RSS = self.optim_func_saliency(result.x,y_real,0)
            AICs[n,1] = self.calc_AIC(RSS,2)
            
            
            ###################
            # relevance model #
            ###################
            
            var =  [0.01,230]
            result = minimize(self.optim_func_relevance, var,args = (y_real),options={'disp': False},method = 'TNC',bounds = [(0,None),(0,None)])# 0 here in args, because of leastsquares
            results_relevance[n,:] = result.x
            RSS = self.optim_func_relevance(result.x,y_real,0)
            AICs[n,2] = self.calc_AIC(RSS,2)
            
            
            ##################
            ## dependent model # 
            ###################
            var = [0.01,194]
            result = minimize(self.optim_func_dependent, var,args = (y_real),options={'disp': False},method = 'TNC',bounds = [(0,None),(0,None)])# 0 here in args, because of leastsquares
            results_dependent[n,:] = result.x
            RSS = self.optim_func_dependent(result.x,y_real,0)
            AICs[n,3] = self.calc_AIC(RSS,2)
            
            
            ###############
            ## stable model 
            ###############
            var = [0.2]
            result = minimize(self.optim_func_stable,var, args = (y_real),options={'disp': False},method = 'TNC')
            results_stable[n,:] = result.x
            RSS = self.optim_func_stable(result.x,y_real,0)
            AICs[n,4] = self.calc_AIC(RSS,1)
            
  
        
        return [results_model,results_saliency, results_relevance, results_dependent,results_stable, AICs]

              
     
    def make_saliency_relevance_plot(self,fname, plt_name, shading = True, color = '', title = None, legend = True ): 
        effects = ['saliency','relevance']
        calc_NSNT = {}
        colors = [self.colors[0],self.colors[1]]
        for i, effect in enumerate(effects): 
            # pick the variables we need based on the effect we are calculating
            
            if not 'relevance' == effect: 
                depVar1 = 'to_target_salient_0'
                timeVar1 ='onset_salient_0'
                depVar2 = 'to_target_nonsalient_0'
                timeVar2 = 'onset_nonsalient_0'
            else: 
                depVar1 = 'salient_to_salient_0'
                timeVar1 = 'salient_to_salient_t_0'
                depVar2 = 'non_salient_to_salient_0'
                timeVar2 = 'non_salient_to_salient_t_0'
            
            res = SMART(fname, depVar1, timeVar1,depVar2, timeVar2)
            res.runSmooth(self.krnSize, self.minTime, self.maxTime, self.stepTime)
            res.runPermutations(self.nPerm, 0) # see if they are different from 0 
            res.runStats(self.sigLevel)
            
            self.timeVect = res.timeVect
            
            # make the standard plot as well 
            #self.make_standard_plot(res, effect) # unindent thi sto mke the standard plot 
            # cannot do both at the same time cause they will plot over eachother 
            
            diffs_dv = res.smooth_dv1  - res.smooth_dv2
            sum_weights = res.weights_dv1 + res.weights_dv2
        
            #these are for plotting , not used in analysis perse, but are incorporated in the t-test 
            weighDv1_diff = SF.weighArraysByColumn(diffs_dv, sum_weights)
            weighDv1Average_diff = np.nansum(weighDv1_diff, axis=0)
            calc_NSNT[effect] = diffs_dv # effect_per_sub, used in make_limbo_plot
            calc_NSNT[effect + '_weights'] = sum_weights 

            
            # plot this 
            plt.plot(self.timeVect, weighDv1Average_diff, color = colors[i])
            # add CI 
            conf95 = SF.weighConfOneSample95(diffs_dv, sum_weights)
            plt.fill_between(self.timeVect,weighDv1Average_diff - conf95, weighDv1Average_diff + conf95, color=colors[i], alpha=0.25)
            # add where they are different from zero with a thicker line 
            for ind, j in enumerate(res.sigCL):
                        if res.sumTvals[ind] >= res.sigThres:
                            plt.plot(self.timeVect[res.sigCL[ind]], weighDv1Average_diff[res.sigCL[ind]], color=colors[i], linewidth =4  )
            
          
        plt.xlabel('Saccade latency (ms)', fontsize = 18)
        plt.ylabel('Effect (' + r'$\Delta$'+ 'P)', fontsize = 18 )
        # get rid of the lines at the top and right 
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if title: 
            plt.title(title, fontsize = 16 )
        
        # add custom legend 
        from matplotlib.lines import Line2D
        saliency = Line2D([0], [0], color=self.colors[0], lw=4, label = 'Saliency')
        relevance= Line2D([0], [0], color=self.colors[1], lw=4, label = 'Relevance')
                
        if legend: 
            plt.legend(handles=[saliency, relevance], fontsize = 16, frameon =False, ncol =2 )
        
        # add the KDE (which should be the same, because it's all the data)
        ax1 = plt.gca()
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='both', which='minor', labelsize=12)
        from matplotlib.ticker import FormatStrFormatter
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # Plot kernel density estimation KDE
        sTimes1, unqT1, countT1 = SF.getKDE(np.hstack(res.data[res.t1]),self.timeVect, res.krnSize) 
        countT1 = self.smooth(countT1,3)
        ax1_1 = ax1.twinx()
        ax1_1.set_ylim(1, 400)
        ax1_1.plot(self.timeVect, sTimes1, '--k', alpha = 0.6)
        ax1_1.bar(unqT1, countT1, color='black', alpha = 0.6)
        ax1_1.set_xlim(self.minTime,self.maxTime)
        ax1_1.set_yticks(np.linspace(0,np.max(np.hstack([sTimes1])),3, dtype=int))
        ax1_1.spines['right'].set_visible(False)
        ax1_1.spines['top'].set_visible(False)

        figure = plt.gcf() # get current figure
        figure.set_size_inches(6,4)
        figure.tight_layout()
        plt.savefig(plt_name + '.png')
        
        return calc_NSNT
    

    
    def af_make_plot(self,res,df_SMART_ampl = None, smooth = True, color1 = '#0089cf', color2 = 'black', label1 = 'Target salient', label2 = 'Target non-salient', xlabel = 'Saccade latency (ms)', ylabel = 'Proportion', legend = True, legend_loc = 4, extra_line_res = None,linepos = 0, kde = True,shading = True, fillcolor = 'red'  ): 
        import time 
        if smooth: 
            res.runSmooth(self.krnSize, self.minTime, self.maxTime, self.stepTime)
        print('starting now...')
        print(self.nPerm)
        t0 = time.time() 
        res.runPermutations(self.nPerm)
        t1 = time.time() 
        print('time elapsed: ' +  str(np.round(t1 -t0) ) + ' seconds')
        res.runStats(self.sigLevel)
        # determine the 95 confidence level for this condition 
        if shading: 
            dv1_95 = res.conf95
            plt.fill_between(self.timeVect,res.weighDv1Average - dv1_95, res.weighDv1Average + dv1_95, color=color1, alpha=0.25)

        # plot it 
        plt.plot(self.timeVect, res.weighDv1Average, color1)
        # THIS WAY IT ALSO WORKS IF THERE IS JUST ONE CONDITION
        # will just not execute if there is no res.weighDv2Average 
        try: 
            if shading: 
                dv2_95 = res.conf95
                plt.fill_between(self.timeVect,res.weighDv2Average - dv2_95, res.weighDv2Average + dv2_95, color=color2, alpha=0.25)
            # plot it     
            plt.plot(self.timeVect, res.weighDv2Average, color2)
            
            # add shading between both lines 
            upper = res.weighDv1Average- dv1_95
            lower = res.weighDv2Average + dv1_95
            plt.fill_between(self.timeVect,upper,lower, where = upper > lower, color = fillcolor, alpha = 0.25)
        except: 
            pass 
    
        ax1 = plt.gca()
        if legend: 
            legend_loc = 'upper center'
            ax1.legend([label1, label2],loc=legend_loc,fontsize = 14,frameon=False, ncol = 2, bbox_to_anchor = (0.5,-0.2))
        
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='both', which='minor', labelsize=12)
        from matplotlib.ticker import FormatStrFormatter
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
        # plot condition differences
        if len(res.dv2) > 0: # if there is indeed a 2nd variable 
            c = 0 
            for ind, i in enumerate(res.sigCL):
                if res.sumTvals[ind] >= res.sigThres:  #only if it is really significant 
                        x = self.timeVect[res.sigCL[ind]]
                        y  = [linepos + c*-0.05]* len(x)
                        c+=1 # use this instead of ind, becuase it will only count the sig iterations 
                        plt.plot(x,y,color1, linewidth = 5)
                        plt.plot(x,y,color2,linestyle = '--', linewidth = 5, dashes = [2,2])
                        print('There is a sig difference here')
                    
        ax1.set_xlim(res.timeMin, res.timeMax-1)
        ax1.set_ylim(-0.5, 1.1)
        ax1.set_xlabel(xlabel, fontsize=18)
        ax1.set_ylabel(ylabel, size=18)
        
        # put the distribution there as well 
        if smooth: 
            if kde: 
                # Plot kernel density estimation KDE
                sTimes1, unqT1, countT1 = SF.getKDE(np.hstack(res.data[res.t1]),self.timeVect, self.krnSize) 
                sTimes2, unqT2, countT2 = SF.getKDE(np.hstack(res.data[res.t2]),self.timeVect, self.krnSize)
                countT1 = self.smooth(countT1,5)
                countT2 = self.smooth(countT2,5)
                ax1_1 = ax1.twinx()
                ax1_1.plot(self.timeVect, sTimes1, '--k', alpha = 0.6)
                ax1_1.plot(self.timeVect, sTimes2, '-k', alpha = 0.3)
                ax1_1.bar(unqT1, countT1, color='black', alpha = 0.25)
                ax1_1.bar(unqT2, countT2, color='black', alpha = 0.25)
                ax1_1.set_ylim(0, 400)
                ax1_1.set_xlim(self.minTime,self.maxTime)
                ax1_1.spines['top'].set_visible(False)
                ax1_1.spines['right'].set_visible(False)
                ax1_1.set_yticks(np.linspace(0,np.max(np.hstack([sTimes1, sTimes2])),3, dtype=int))
        if extra_line_res: 
            # indicate where the line is differnt from zero (only if cluster is sig )
            for res_nr,res in enumerate(extra_line_res): 
                if res_nr == 0: 
                    color = color1 
                else: 
                    color = color2 
                for ind, i in enumerate(res.sigCL):
                        if res.sumTvals[ind] >= res.sigThres:
                            plt.plot(self.timeVect[res.sigCL[ind]], res.weighDv1Average[res.sigCL[ind]], color = color, linewidth =3  )
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        
     ################    
     # FITTING FUNCTIONS  
     ##################
    
     # the basic function 
     
    # S(t)
    def salient(self,time,t0,r): # shouldn't r be s? doesn't really matter ala you call rS as positional arg when calling this function i guess
        y = np.zeros((len(time)))
        for i,t in enumerate(time): 
            if t < t0: 
                y[i] =1 
            else: 
                y[i] = np.e**(-r*(t-t0))
        return y 
    
    # R(t)
    def relevance(self,time,t0,r):
        y = np.zeros((len(time)))
        for i,t in enumerate(time): 
            if t < t0: 
                y[i] = 0 
            else: 
                y[i] = 1 - np.e**(-r*(t-t0))
        return y
    
    # NSNT = (1-S) * (1-R) * 0.5
    # this is the full model 
    def chance_to_NSNT(self,t,rS,rR,t0S,t0R):
        S = self.salient(t,t0S,rS)
        G = self.relevance(t,t0R,rR)
        P = (1-G) * (1-S) * 0.5
        return P
    
    # same, but t0R and rR and now the same as rS and t0S
    # this is the interdendent model 
    def chance_to_NSNT_dependent(self,t,rS,t0S):
        S = self.salient(t,t0S,rS)
        G = self.relevance(t,t0S,rS)
        P = (1-G) * (1-S) * 0.5
        return P
    # saliency only model, relevance is set to 0
    def chance_to_NSNT_saliency(self,t,rS,t0S):
        S = self.salient(t,t0S,rS)
        P = (1-0) * (1-S) * 0.5
        return P
    
    # relevance only model, saliency is set to 0
    def chance_to_NSNT_relevance(self,t,rR,t0R):
        G = self.relevance(t,t0R,rR)
        P = (1-G) * (1-0) * 0.5
        return P
    
    #######################
    ## optimizer functions 
    #######################
    
    # full model 
    def optim_func(self,var,*args): 
        y_real = args[0]
        y = self.chance_to_NSNT(self.timeVect,var[0],var[1],var[2],var[3])        
        dif = np.sum((y_real-y)**2) # least squares method
        return dif 
       
            
    def optim_func_stable(self,var,*args): 
        y_real = args[0]
        y = [var[0]]*len(self.timeVect)
        dif = np.sum((y_real-y)**2) # least squares method
        return dif 
        
    def optim_func_dependent(self,var,*args): 
        y_real = args[0]
        y = self.chance_to_NSNT_dependent(self.timeVect,var[0],var[1])        
        dif = np.sum((y_real-y)**2) # least squares method
        return dif 
      
    
    def optim_func_saliency(self,var,*args): 
        y_real = args[0]
        y = self.chance_to_NSNT_saliency(self.timeVect,var[0],var[1])
        dif = np.sum((y_real-y)**2) # least squares method
        return dif 
        
            
    def optim_func_relevance(self,var,*args): 
        y_real = args[0]
        y = self.chance_to_NSNT_relevance(self.timeVect,var[0],var[1])
        dif = np.sum((y_real-y)**2) # least squares method
        return dif 
          
        
    def make_model_plot(self,results_model_ML):
        t = self.timeVect
        [rS,rR,t0S,t0R] = results_model_ML
        
        plt.plot(t,self.salient(t,t0S,rS), color = self.colors[0], linewidth =4)
        plt.plot(t,self.relevance(t,t0R,rR), color = self.colors[1], linewidth =4)
        
        diff = abs(self.salient(t,t0S,rS) - self.relevance(t,t0R,rR))
        intersection = self.timeVect[np.argsort(diff)[0]] 
        print('intersection: ' + str(intersection))
        plt.xlabel('Time (ms)', fontsize = 18)
        plt.ylabel('Bias', fontsize = 18)
        # resume plotting 
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        # get rid of the lines at the top and right 
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        
        plt.ylim(-0.1,1.1)
        # exact same legend as above plot 
        from matplotlib.lines import Line2D
        saliency = Line2D([0], [0], color=self.colors[0], lw=4, label = 'Saliency')
        relevance= Line2D([0], [0], color=self.colors[1], lw=4, label = 'Relevance')
        
        plt.legend(handles=[saliency, relevance], fontsize = 16, frameon =False, ncol =2 )
        figure = plt.gcf() # get current figure
        figure.set_size_inches(6,4)
        figure.tight_layout()
        plt.savefig(os.getcwd() + '/plots/' + self.add + '/reconstructed_model.png')
        
    
  
    def make_paper_subplot(self, fname, plt_name, color1 = 'gray', color2 = 'black' ): 
        plt.figure() 
        effects = ['saliency','relevance']
        legends = [True, False]
        for i, effect in enumerate(effects): 
            plt.subplot(2,1,i+1)
            if not 'relevance' == effect: 
                depVar1 = 'to_target_salient_0'
                timeVar1 ='onset_salient_0'
                depVar2 = 'to_target_nonsalient_0'
                timeVar2 = 'onset_nonsalient_0'
            else: 
                depVar1 = 'salient_to_salient_0'
                timeVar1 = 'salient_to_salient_t_0'
                depVar2 = 'non_salient_to_salient_0'
                timeVar2 = 'non_salient_to_salient_t_0'
            
           
            res = SMART(fname, depVar1, timeVar1,depVar2, timeVar2)
            res.runSmooth(self.krnSize, self.minTime, self.maxTime, self.stepTime)
            res.runPermutations(self.nPerm, 0)
            res.runStats(self.sigLevel)
            
            if effect == 'saliency': 
              ylabel = ' P(target)'
              xlabel = '' 
              fillcolor = 'red'
            else: 
              ylabel = 'P(salient)'
              xlabel = 'Saccade latency (ms)'
              fillcolor = self.colors[1]
              
            self.af_make_plot(res, smooth = True,  color1 = color1, color2 =  color2,ylabel = ylabel, legend = legends[i],xlabel= xlabel, fillcolor = fillcolor)
        figure = plt.gcf() # get current figure
        figure.set_size_inches(6,4)
        figure.tight_layout()
        plt.savefig(plt_name)

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def permute(self,df_new, nPerm):
        sum_t_vals = [] # saving the biggest t val for every permutation 
        for p in range(nPerm):         
            groups_list = []
            for name, group in df_new.groupby([df_new.index]): 
                np.random.shuffle(group['condition'].values)
                groups_list.append(group) # put together all subjects again
            
            df_shuf = pd.concat(groups_list) # make a new data frame with labels shuffled for each subject
            
            # select new permuted conditions
            cond1_df = df_shuf[df_shuf['condition']==1]
            cond2_df = df_shuf[df_shuf['condition']==2]
            
            # convert to 2d numpy array
            cond1 = np.vstack(cond1_df.diffs_dv)
            cond2 = np.vstack(cond2_df.diffs_dv)
            weight1 = np.vstack(cond1_df.sum_weights)
            weight2 = np.vstack(cond2_df.sum_weights)
            
            # now put in t-test for comparisions between conditions 
            [t_vals,p_vals] = SF.weighted_ttest_rel(cond1,cond2, weight1, weight2) 
            # extract biggest cluster 
            clusters, indx = SF.getCluster(p_vals < 0.05) # I think using this function is OK
            # calculate sum of t-vals (only for those clusters where it is lower than 0.05 )
            sigCl = [indx[i] for i in range(len(clusters)) if clusters[i][0] == True]
            sums_t = [np.sum(abs(t_vals[i])) for i in sigCl]
            # see which one is the biggest, and add it here 
            if len(sums_t) ==0: 
                sum_t_vals.append(np.max(t_vals))

            else: 
                sum_t_vals.append(np.sort(sums_t)[-1])
        return sum_t_vals  

