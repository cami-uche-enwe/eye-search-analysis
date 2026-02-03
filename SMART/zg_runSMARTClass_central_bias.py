#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:34:36 2018

@author: Jonathan van Leeuwen
"""
#==============================================================================
#==============================================================================
# # Run the SMART pipeline between conditions (within subject)
#==============================================================================
#==============================================================================
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem, t

t = time.time()
#==============================================================================
# Assumes that the data is in a pickle file and structured identical to 
# 'ExampleDataSMART.p'.
# 
# A pandas dataframe with:
#       Each participant in its own row 
#       Time var1 in 1 column, all the data in 1 cell
#       Dep. var1 in 1 column, all the data in 1 cell
#       Time var2 in 1 column, all the data in 1 cell # if testing between conditions
#       Dep. var2 in 1 column, all the data in 1 cell # if testing between conditions
#
# Example:
#       Index/pp	            TimeVar1	                    DepVar1	                     TimeVar2                      DepVar2
#           0	    [155 192 279 ..., 143 142 149]	    [0 0 1 ..., 0 1 0]	    [159 163 201 ..., 149 229 154]	    [0 1 0 ..., 1 0 1]
#           1	    [379 312 272 ..., 278 288 267]	    [1 1 1 ..., 0 1 1]	    [386 437 422 ..., 226 319 237]	    [1 1 1 ..., 1 1 0]
#           2	    [192 208 236 ..., 175 268 171]	    [0 0 0 ..., 0 0 0]	    [180 227 189 ..., 172 180 205]	    [1 1 1 ..., 1 1 1]
#           3	    [397 291 412 ..., 457 408 366]	    [1 1 1 ..., 1 1 1]	    [392 452 459 ..., 378 342 444]	    [1 1 0 ..., 1 1 1]

from SMARTClass import SMART

# Settings
fName = '/Users/camillaucheomaenwereuzor/Desktop/Coding/eye-search-analysis/SMART/allfiltered.p'
depVar1 = 'tgt_first_ns_pre'
timeVar1 ='sacc_lat_ns_pre'
depVar2 = 'tgt_first_s_pre'
timeVar2 = 'sacc_lat_s_pre'
depVar3 = 'tgt_first_ns_post'
timeVar3 = 'sacc_lat_ns_post'
depVar4 = 'tgt_first_s_post'
timeVar4 = 'sacc_lat_s_post'


krnSize = 10
minTime = 125
maxTime = 326
stepTime = 1
nPerm = 1
baseline = 0.5
sigLevel = 0.05

# ==============================================================================
# # Central Selection Bias (StoUp_close_diff vs StoUp_close_same) - Blue - Red
# ==============================================================================

# Initialize and run SMART analysis
pairedSamp1 = SMART(fName, depVar1, timeVar1, depVar2, timeVar2)
pairedSamp1.runSmooth(krnSize, minTime, maxTime, stepTime)
pairedSamp1.runPermutations(nPerm)
pairedSamp1.runStats(sigLevel)

# Original plot for pairedSamp1
pairedSamp1.runPlot()
cf = plt.gcf()
cf.set_size_inches(12, 8)  # Increase figure size for clarity
pairedSamp1.ax1.legend(['TC-Same','TC-Different'], prop={"size": 14}, loc=1, framealpha=1, fancybox=True)
pairedSamp1.ax1.set_xlabel('Saccade Latency (ms)', fontsize=24)
pairedSamp1.ax1.set_ylabel('P(critical singleton)', size=22)

# Adjust tick font sizes for x and y axes
pairedSamp1.ax1.tick_params(axis='x', labelsize=16)  # Adjust x-axis tick font size
pairedSamp1.ax1.tick_params(axis='y', labelsize=16)  # Adjust y-axis tick font size
pairedSamp1.ax1.set_ylim([-0.31, 1.51])
pairedSamp1.ax1.set_yticks([0, 0.5, 1.0]) # Set y-axis ticks to 0, 0.5, 1.0

plt.subplots_adjust(hspace=0.4)



import numpy as np
from scipy.stats import sem

# ==============================================================================
# Difference Plot with Red/Blue Bold Lines for Significant Clusters
# ==============================================================================

# Calculate the difference between smoothed averages
difference = pairedSamp1.weighDv2Average - pairedSamp1.weighDv1Average

# Calculate confidence intervals for the difference
# Standard error of the mean (SEM) across participants for the difference
sem_diff = sem(pairedSamp1.smooth_dv2 - pairedSamp1.smooth_dv1, axis=0)
# 95% confidence interval: mean ± 1.96 * SEM
diff_conf95 = 1.96 * sem_diff

# Create a new figure for the difference plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the difference line
ax.plot(pairedSamp1.timeVect, difference, color='blue', linewidth=2, label='Central selection bias in TC conditinos')
# Plot the difference line


# Plot the confidence intervals
ax.fill_between(pairedSamp1.timeVect, 
                difference - diff_conf95, 
                difference + diff_conf95, 
                color='blue', alpha=0.25)

# Highlight significant clusters with red or blue bold lines
for ind, cluster in enumerate(pairedSamp1.sigCL):
    # Use red if the mean difference is positive, blue if negative
    cluster_color = 'blue' 
    
    ax.plot(pairedSamp1.timeVect[cluster], difference[cluster], color=cluster_color, linewidth=4)

# Add a horizontal dashed line at y=0
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Adjust plot settings
ax.set_xlim([125, 325])  # Extra space before 120ms and after 300ms
ax.set_ylim([-0.2, 1.0])  # Adjust based on your data
ax.set_xlabel('Saccade Latency (ms)', fontsize=24)
ax.set_ylabel('Central Selection Bias', fontsize=22)
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
# Add a legend
custom_lines = [
    plt.Line2D([0], [0], color='blue', linewidth=2, label='Central selection bias in TC conditions'),
    plt.Line2D([0], [0], color='blue', linewidth=4, label='Significant cluster')
]
ax.legend(handles=custom_lines, loc='best', fontsize=22)
# Show the plot
plt.show()


# ==============================================================================
# # Central Selection Bias (StoUp_far_same vs StoUp_far_diff) - Red - Blue
# ==============================================================================

# Initialize and run SMART analysis
pairedSamp2 = SMART(fName, depVar3, timeVar3, depVar4, timeVar4)
pairedSamp2.runSmooth(krnSize, minTime, maxTime, stepTime)
pairedSamp2.runPermutations(nPerm)
pairedSamp2.runStats(sigLevel)

# Original plot for pairedSamp2
pairedSamp2.runPlot()
cf = plt.gcf()
cf.set_size_inches(12, 8)  # Increase figure size for clarity
pairedSamp2.ax1.legend(['TF-Same','TF-Different'], prop={"size": 16}, loc=1, framealpha=1, fancybox=True)
pairedSamp2.ax1.set_xlabel('Saccade Latency (ms)', fontsize=24)
pairedSamp2.ax1.set_ylabel('P(critical singlton)', size=22)
# Adjust tick font sizes for x and y axes
pairedSamp2.ax1.tick_params(axis='x', labelsize=16)  # Adjust x-axis tick font size
pairedSamp2.ax1.tick_params(axis='y', labelsize=16)  # Adjust y-axis tick font size
pairedSamp2.ax1.set_ylim([-0.31, 1.51])
pairedSamp2.ax1.set_yticks([0, 0.5, 1.0])

plt.subplots_adjust(hspace=0.4)

# ==============================================================================
# Difference Plot 2: Red - Red (StoUp_far_same vs StoUp_far_diff)
# ==============================================================================

# Calculate the second difference
difference2 = pairedSamp2.weighDv1Average - pairedSamp2.weighDv2Average

# Calculate confidence intervals for the difference
sem_diff2 = sem(pairedSamp2.smooth_dv1 - pairedSamp2.smooth_dv2, axis=0)  # SEM for the difference
conf95_diff2 = 1.96 * sem_diff2  # 95% confidence interval

# Create a new figure for the second difference plot
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Plot the difference line
ax2.plot(
    pairedSamp2.timeVect,
    difference2,
    color='red',
    linewidth=2,
    label='Central selection bias in TF conditions'
)

# Plot the confidence intervals
ax2.fill_between(
    pairedSamp2.timeVect,
    difference2 - conf95_diff2,
    difference2 + conf95_diff2,
    color='red',
    alpha=0.25
)

# Highlight significant clusters with bold red lines
for ind, cluster in enumerate(pairedSamp2.sigCL):
    cluster_color = 'red'
    ax2.plot(pairedSamp2.timeVect[cluster], difference2[cluster], color=cluster_color, linewidth=4)

# Add a horizontal dashed line at y=0
ax2.axhline(0, color='black', linestyle='--', linewidth=1)

# Adjust plot settings
ax2.set_xlim([125, 325])  # Set x-axis limits for consistency
ax2.set_ylim([-0.2, 1.0])  # Adjust based on your data
ax2.set_xlabel('Saccade Latency (ms)', fontsize=24)
ax2.set_ylabel('Central Selection Bias', fontsize=22)
ax2.tick_params(axis='x', labelsize=22)
ax2.tick_params(axis='y', labelsize=22)

# Add a legend
custom_lines = [
    plt.Line2D([0], [0], color='red', linewidth=2, label='Central selection bias in TF conditions'),
    plt.Line2D([0], [0], color='red', linewidth=4, label='Significant cluster')
]
ax2.legend(handles=custom_lines, loc='best', fontsize=22)

# Show the plot
plt.show()


# Identify significant clusters
significant_indices2 = np.where((difference2 - conf95_diff2) > 0)[0]
if significant_indices2.size > 0:
    significant_clusters2 = np.split(significant_indices2, np.where(np.diff(significant_indices2) != 1)[0] + 1)
    for cluster in significant_clusters2:
        start_time = pairedSamp2.timeVect[cluster[0]]
        end_time = pairedSamp2.timeVect[cluster[-1]]
        print(f"Significant cluster from {start_time} to {end_time} ms for difference2")
else:
    print("No significant clusters found for Upper Bias")




# Highlight significant clusters and print details
print("Significant Clusters Information for Difference Plot 2:")
for ind, cluster in enumerate(pairedSamp2.sigCL):
    cluster_time = pairedSamp2.timeVect[cluster]  # Time range for the current cluster
    t_value = pairedSamp2.sumTvals[ind]  # Sum of t-values for the cluster
    is_significant = t_value >= pairedSamp2.sigThres  # Check if the cluster is significant
    cluster_diff = difference2[cluster]  # Difference values for the cluster

    print(f"Cluster {ind + 1}:")
    print(f"  Time Range: {cluster_time[0]}ms to {cluster_time[-1]}ms")
    print(f"  Sum of T-Values: {t_value}")
    print(f"  Significant: {'Yes' if is_significant else 'No'}")
    print(f"  Difference Values: {cluster_diff}")
# ==============================================================================
# Average of the Two Difference Plots
# ==============================================================================

# Calculate the average of the two difference plots
average_difference = (difference + difference2) / 2

# Calculate pooled SEM and confidence intervals for the average
pooled_sem = np.sqrt(sem(pairedSamp1.smooth_dv2 - pairedSamp1.smooth_dv1, axis=0)**2 +
                     sem(pairedSamp2.smooth_dv1 - pairedSamp2.smooth_dv2, axis=0)**2) / 2
average_conf95 = 1.96 * pooled_sem

# Identify areas where the average difference is significantly greater than 0
significant_indices = np.where((average_difference - average_conf95) > 0)[0]
if len(significant_indices) > 0:
    significant_start = pairedSamp1.timeVect[significant_indices[0]]
    significant_end = pairedSamp1.timeVect[significant_indices[-1]]
    # Calculate t-cluster as the sum of t-values
    t_cluster = np.sum((average_difference / pooled_sem)[significant_indices])
    # Calculate effect size (d-cluster)
    d_cluster = t_cluster / np.sqrt(len(pairedSamp1.smooth_dv1))
    # Estimate p-value (using permutation distribution, for example)
    # Assuming pairedSamp1.permDistr contains permutation distribution of t-cluster
    p_value = np.mean(pairedSamp1.permDistr >= t_cluster)
    # Print the significant time window and statistics
    print(f"Significant time window: {significant_start:.0f} ms to {significant_end:.0f} ms")
    print(f"p-value: {p_value:.3f}")
    print(f"t-cluster: {t_cluster:.2f}")
    print(f"d-cluster: {d_cluster:.2f}")
else:
    print("No significant time window found.")

# ==============================================================================
# # Difference Plot 1: Blue - Red (StoUp_close_diff vs StoUp_close_same)
# ==============================================================================

# Calculate the first difference
difference1 = pairedSamp1.weighDv2Average - pairedSamp1.weighDv1Average
conf95_diff1 = np.sqrt(pairedSamp1.conf95**2 + pairedSamp1.conf95**2)

# Create a new figure for the first difference plot
fig1, ax1 = plt.subplots(figsize=(14, 8))#figsize=(10, 6)#)

ax1.plot(
    pairedSamp1.timeVect,
    difference1,
    color='blue',
    linewidth=2,
    label='Central selection bias in TC conditions'
)

ax1.fill_between(
    pairedSamp1.timeVect,
    difference1 - conf95_diff1,
    difference1 + conf95_diff1,
    color='blue',
    alpha=0.25
)

ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax1.set_xlabel('Saccade Latency (ms)', fontsize=24)
ax1.set_ylabel('Central Selection Bias', fontsize=22)
ax1.set_ylim([-0.2, 1])

# Explicitly set the x-axis limit to give some space (110 to 310ms)
ax1.set_xlim([125, 325])  # Extra space before 120ms and after 300ms

# Define the ticks at 20ms intervals between 120ms and 300ms
ticks_avg = list(range(125, 326, 25))  # Generates ticks at 120, 140, 160, ..., 300
ax1.set_xticks(ticks_avg)
ax1.tick_params(axis='x', labelsize=18)  # X-axis tick labels font size
ax1.tick_params(axis='y', labelsize=18)  # Y-axis tick labels font size

ax1.legend(loc='best', fontsize=22) #adjust the label
#ax1.set_title('Central Selection Bias (Difference 1: Top-close vs Close-same)')

plt.show()

# ==============================================================================
# Averaged Difference Plot with Red/Blue Bold Lines for Significant Clusters
# ==============================================================================

# Compute the averaged difference
average_difference = (difference1 + difference2) / 2
conf95_avg_diff = np.sqrt((conf95_diff1**2 + conf95_diff2**2) / 2)

# Identify significant time windows
significant_indices = np.where((average_difference - conf95_avg_diff) > 0)[0]
significant_clusters = []
if significant_indices.size > 0:
    significant_clusters = np.split(significant_indices, np.where(np.diff(significant_indices) != 1)[0] + 1)
    for cluster in significant_clusters:
        start_time = pairedSamp1.timeVect[cluster[0]]
        end_time = pairedSamp1.timeVect[cluster[-1]]
        print(f"Significant cluster from {start_time:.1f} to {end_time:.1f} ms")
else:
    print("No significant clusters found")

# Clip the data to the desired x-axis range (120 to 300 ms)
valid_indices_avg = (pairedSamp1.timeVect >= 120) & (pairedSamp1.timeVect <= 325)
timeVect_avg_clipped = pairedSamp1.timeVect[valid_indices_avg]
average_difference_clipped = average_difference[valid_indices_avg]
conf95_avg_diff_clipped = conf95_avg_diff[valid_indices_avg]

# Create a new figure for the averaged difference plot
fig3, ax3 = plt.subplots(figsize=(12, 8))

# Plot the averaged difference line (with clipped data)
ax3.plot(timeVect_avg_clipped, average_difference_clipped, color='purple', linewidth=2, label='Overall central selection bias')

# Plot the confidence intervals for the averaged difference (with clipped data)
ax3.fill_between(
    timeVect_avg_clipped,
    average_difference_clipped - conf95_avg_diff_clipped,
    average_difference_clipped + conf95_avg_diff_clipped,
    color='purple',
    alpha=0.25
)

# Highlight significant clusters with red or blue bold lines
for cluster in significant_clusters:
    # Determine the color based on the mean difference in the cluster
    cluster_color = 'purple'
    start_time = pairedSamp1.timeVect[cluster[0]]
    end_time = pairedSamp1.timeVect[cluster[-1]]
    cluster_indices = (pairedSamp1.timeVect >= start_time) & (pairedSamp1.timeVect <= end_time)
    ax3.plot(
        pairedSamp1.timeVect[cluster_indices],
        average_difference[cluster_indices],
        color=cluster_color,
        linewidth=4
    )

# Add a horizontal line at y=0
ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)

# Add labels
ax3.set_xlabel('Saccade Latency (ms)', fontsize=24)
ax3.set_ylabel('Central Selection Bias', fontsize=22)
ax3.set_ylim([-0.2, 1])

# Explicitly set the x-axis limit to give some space (110 to 310ms)
ax3.set_xlim([125, 325])  # Extra space before 120ms and after 300ms

# Define the ticks at 20ms intervals between 120ms and 300ms
ticks_avg = list(range(125, 326, 25))  # Generates ticks at 120, 140, 160, ..., 300
ax3.set_xticks(ticks_avg)
ax3.tick_params(axis='x', labelsize=22, pad=10)  # X-axis tick labels font size
ax3.tick_params(axis='y', labelsize=22, pad=10)  # Y-axis tick labels font size

# Add a legend
custom_lines = [
    plt.Line2D([0], [0], color='purple', linewidth=2, label='Average of TC and TF'),
    #plt.Line2D([0], [0], color='purple', linewidth=4, label='Significant cluster')
]
ax3.legend(handles=custom_lines, loc='best', fontsize=22)

# Show the plot
plt.show()



# # ==============================================================================
# # # Upper Bias (StoClose_upper_close vs StoClose_bottom_close) - Upper vs Bottom
# # ==============================================================================

# # Initialize and run SMART analysis
# pairedSamp3 = SMART(fName, depVar5, timeVar5, depVar6, timeVar6)
# pairedSamp3.runSmooth(krnSize, minTime, maxTime, stepTime)
# pairedSamp3.runPermutations(nPerm)
# pairedSamp3.runStats(sigLevel)


# # Initial plot
# pairedSamp3.runPlot()

# cf = plt.gcf()
# cf.set_size_inches(12, 8)  # Increase figure size for clarity
# pairedSamp3.ax2.remove()
# plt.suptitle('')


# # Modify legend and labels

# pairedSamp3.ax1.legend(['top_close', 'bottom_close', 'Sig. difference'], prop={"size": 10}, loc=1)
# pairedSamp3.ax1.set_xlabel('Saccade Latency (ms)', fontsize=12)
# pairedSamp3.ax1.set_ylabel('P(Saccade to Close)', size=12)
# pairedSamp3.ax1.set_ylim([-0.31, 1.51])

# pairedSamp3.runPlot(lineColor1=[0.0, 255.0, 0.0], lineColor2=[255.0, 165.0, 0.0])  # Green for Upper, Orange for Bottom
# pairedSamp3.ax2.set_ylabel('Frequency (log)', size=15)
# pairedSamp3.ax2.set_xlabel('Sum of cluster t-values', fontsize=15)
# pairedSamp3.ax2.legend(['95th percentile', 'Clusters'], loc=1, prop={"size": 8})

# plt.suptitle('')
# plt.subplots_adjust(hspace=0.4)

# pairedSamp3.ax1.remove()
# pairedSamp3.ax1_1.remove()
# cf = plt.gcf()

# # Plotting the difference (Upper - Bottom)
# difference3 = pairedSamp3.weighDv1Average - pairedSamp3.weighDv2Average
# conf95_diff3 = np.sqrt(pairedSamp3.conf95**2 + pairedSamp3.conf95**2)

# # Create a new figure for the third difference plot
# fig3, ax3 = plt.subplots()

# # Plot the third difference line
# ax3.plot(pairedSamp3.timeVect, difference3, color='green', linewidth=2, label='Difference (top - Bottom)')

# # Plot the confidence intervals for the third difference
# ax3.fill_between(
#     pairedSamp3.timeVect,
#     difference3 - conf95_diff3,
#     difference3 + conf95_diff3,
#     color='green',
#     alpha=0.25
# )

# # Add a horizontal line at y=0
# ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)

# # Add labels and legend
# ax3.set_xlabel('Saccade Latency (ms)', fontsize=12)
# ax3.set_ylabel('Effect of Upper Bias', fontsize=12)
# ax3.set_ylim([-0.2, 1])
# ax3.legend(loc='best')

# plt.title('Upper Bias')
# plt.show()

# # ==============================================================================
# # # Identify significant time windows for Upper Bias
# # ==============================================================================
# significant_indices3 = np.where((difference3 - conf95_diff3) > 0)[0]
# if significant_indices3.size > 0:
#     significant_clusters3 = np.split(significant_indices3, np.where(np.diff(significant_indices3) != 1)[0] + 1)
#     for cluster in significant_clusters3:
#         start_time = pairedSamp3.timeVect[cluster[0]]
#         end_time = pairedSamp3.timeVect[cluster[-1]]
#         print(f"Significant cluster from {start_time} to {end_time} ms for Upper Bias")
# else:
#     print("No significant clusters found for Upper Bias")



####### identify the precentage of the trials in specific time window #######
def calculate_proportion_within_window_per_participant(data, timeVar, minTime, maxTime, condition_name):
    # Extract the column of time data
    time_data = data[timeVar]
    
    # Initialize variables
    total_trials = 0
    total_within_window = 0
    participant_proportions = []
    
    # Loop through each participant
    for participant_times in time_data:
        participant_times = np.array(participant_times)  # Ensure data is a numpy array
        within_window = (participant_times >= minTime) & (participant_times <= maxTime)
        
        # Count trials for this participant
        num_trials = len(participant_times)
        num_within_window = np.sum(within_window)
        
        # Update totals
        total_trials += num_trials
        total_within_window += num_within_window
        
        # Record proportion for this participant
        if num_trials > 0:
            participant_proportions.append(num_within_window / num_trials)
    
    # Calculate overall and average proportions
    overall_proportion = total_within_window / total_trials if total_trials > 0 else 0
    avg_participant_proportion = np.mean(participant_proportions) if participant_proportions else 0
    
    # Print results
    print(f"Condition: {condition_name}")
    print(f"Total trials across all participants: {total_trials}")
    print(f"Trials within [{minTime}, {maxTime}] ms: {total_within_window}")
    print(f"Overall proportion of trials within window: {overall_proportion:.2%}")
    print(f"Average proportion per participant: {avg_participant_proportion:.2%}")
    print("-" * 40)
    return overall_proportion, avg_participant_proportion

data = pd.read_pickle(fName)  # Load the pickle file into a DataFrame
overall_prop1, avg_participant_prop1 = calculate_proportion_within_window_per_participant(data, timeVar1, minTime, maxTime, depVar1)
overall_prop2, avg_participant_prop2 = calculate_proportion_within_window_per_participant(data, timeVar2, minTime, maxTime, depVar2)

############ identify the time window contain 98%of the trials #######
import numpy as np

def identify_middle_98(data, timeVar):
    """
    Identifies the middle 98% of trials based on saccade latency.
    Args:
        data (pd.DataFrame): DataFrame containing the saccade latencies.
        timeVar (str): Column name for the saccade latency data (lists or arrays).
    Returns:
        (float, float): The 2nd and 98th percentiles of the saccade latencies.
    """
    # Flatten the list of saccade latencies across all participants
    all_times = np.concatenate(data[timeVar].values)
    
    # Compute the 2nd and 98th percentiles
    lower_bound = np.percentile(all_times, 2)
    upper_bound = np.percentile(all_times, 95)
    
    print(f"The middle 95% of trials fall between {lower_bound:.2f} ms and {upper_bound:.2f} ms.")
    return lower_bound, upper_bound

# Example usage
timeVar1 = 'saccade_latency_StoTop_far_same'  # Replace with your column name
lower, upper = identify_middle_98(data, timeVar1)



# Identify significant clusters and print formatted results
def print_significant_clusters(paired_sample, difference, conf95_diff, time_vect, sig_level=0.001, df=19):
    significant_indices = np.where((difference - conf95_diff) > 0)[0]
    if significant_indices.size > 0:
        # Identify clusters
        significant_clusters = np.split(significant_indices, np.where(np.diff(significant_indices) != 1)[0] + 1)
        for cluster in significant_clusters:
            start_time = time_vect[cluster[0]]
            end_time = time_vect[cluster[-1]]
            
            # Calculate cluster-level statistics
            cluster_diff = difference[cluster]
            tcluster = np.sum(cluster_diff)  # Sum of t-values
            dcluster = tcluster / np.sqrt(df + 1)  # Estimate of cluster-level effect size
            
            # Print in the required format
            print(f"{start_time:.0f}-{end_time:.0f} ms, p < {sig_level:.3f}, tcluster ({df}) = {tcluster:.2f}, dcluster = {dcluster:.2f}")
    else:
        print("No significant clusters found.")

# Example usage for difference1 (Central Selection Bias: Close-diff vs Close-same)
print_significant_clusters(
    paired_sample=pairedSamp1,
    difference=difference1,
    conf95_diff=conf95_diff1,
    time_vect=pairedSamp1.timeVect,
    sig_level=0.01,  # significance level
    df=23  # degrees of freedom
)

# # Example usage for difference3 (Upper Bias: Upper vs Bottom)
# print_significant_clusters(
#     paired_sample=pairedSamp3,
#     difference=difference3,
#     conf95_diff=conf95_diff3,
#     time_vect=pairedSamp3.timeVect,
#     sig_level=0.01,  # significance level
#     df=23  # degrees of freedom
# )

# Save data for the TOP condition
np.savez("TOP_bias_data.npz", 
         timeVect=timeVect_avg_clipped, 
         avg_diff=average_difference_clipped, 
         conf95=conf95_avg_diff_clipped)

# Save data for the TOP condition
np.savez("top_blue_bias_data.npz", 
         timeVect=pairedSamp1.timeVect, 
         avg_diff=difference1, 
         conf95=conf95_diff1)

# Save data for the TOP condition
np.savez("top_red_bias_data.npz", 
         timeVect=pairedSamp2.timeVect, 
         avg_diff=difference2, 
         conf95=conf95_diff2)



# Create a new figure for the combined plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the first difference (Blue line for TC conditions)
ax.plot(
    pairedSamp1.timeVect,
    difference1,
    color='blue',
    linewidth=1.5,
    label='Central selection bias in TC conditions'
)

# Plot the first confidence interval (Blue shaded area)
ax.fill_between(
    pairedSamp1.timeVect, 
    difference1 - conf95_diff1, 
    difference1 + conf95_diff1, 
    color='blue', 
    alpha=0.25
)

# Plot the second difference (Red line for TF conditions)
ax.plot(
    pairedSamp2.timeVect,
    difference2,
    color='red',
    linewidth=1.5,
    label='Central selection bias in TF conditions'
)

# Plot the second confidence interval (Red shaded area)
ax.fill_between(
    pairedSamp2.timeVect, 
    difference2 - conf95_diff2, 
    difference2 + conf95_diff2, 
    color='red', 
    alpha=0.25
)

# Highlight significant clusters for `difference1` (Blue bold lines)
for ind, cluster in enumerate(pairedSamp1.sigCL):
    ax.plot(pairedSamp1.timeVect[cluster], difference1[cluster], color='blue', linewidth=3)

# Highlight significant clusters for `difference2` (Red bold lines)
for ind, cluster in enumerate(pairedSamp2.sigCL):
    ax.plot(pairedSamp2.timeVect[cluster], difference2[cluster], color='red', linewidth=3)

# Add a horizontal dashed line at y=0
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Adjust plot settings
ax.set_xlim([125, 325])  # Extra space before 120ms and after 300ms
ax.set_ylim([-0.2, 1.0])  # Adjust based on your data
ax.set_xlabel('Saccade Latency (ms)', fontsize=24)
ax.set_ylabel('Central Selection Bias', fontsize=22)
ax.tick_params(axis='x', labelsize=22, pad=10)
ax.tick_params(axis='y', labelsize=22, pad=10)

# Add a legend
custom_lines = [
    plt.Line2D([0], [0], color='blue', linewidth=2, label='TC'),
    plt.Line2D([0], [0], color='red', linewidth=2, label='TF'),
]
ax.legend(handles=custom_lines, loc='best', fontsize=18)

# Show the plot
plt.show()
