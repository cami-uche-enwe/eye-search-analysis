#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 2025

@author: Camilla Ucheoma Enwereuzor
"""

from math import atan2, degrees
import numpy as np
from pygazeanalyser import gazeplotter

# Screen parameters
monitor_height = 29.6 # Monitor height in cm
view_dist = 70 # Distance between monitor and participant in cm
vert_res = 1080 # Vertical resolution of the monitor in pix
# dist_in_pix = 267.82 # The stimulus distance/position/size in pixels

# pix/deg conversion functions adapted from Sebastiaan Mathôt: https://www.osdoc.cogsci.nl/3.1/visualangle/

def pix2deg(dist_in_pix, monitor_height=monitor_height, view_dist=view_dist, vert_res=vert_res):
    
    # Calculate the number of degrees that correspond to a single pixel
    deg_per_px = degrees(atan2(.5*monitor_height, view_dist)) / (.5*vert_res)
    # print(f'{deg_per_px} degrees correspond to a single pixel') 
    # Calculate the size of the stimulus in degrees
    dist_in_deg = round((dist_in_pix * deg_per_px), 2)
    # print(f'The size of the stimulus is {dist_in_pix} pixels and {dist_in_deg} visual degrees')
    
    return dist_in_deg

def deg2pix(dist_in_deg, monitor_height=monitor_height, view_dist=view_dist, vert_res=vert_res):
    
    # Calculate the number of degrees that correspond to a single pixel
    deg_per_px = degrees(atan2(.5*monitor_height, view_dist)) / (.5*vert_res)
    # print(f'{deg_per_px} degrees correspond to a single pixel') 
    # Calculate the size of the stimulus in pixels
    dist_in_pix = round((dist_in_deg / deg_per_px), 2)
    # print(f'The size of the stimulus is {dist_in_pix} pixels and {dist_in_deg} visual degrees')

    return dist_in_pix


def eucli_dist(gaze_x, gaze_y, stim_x, stim_y):
    """
    Calculate the Euclidian distance between two coordinates.

    Arguments
    
    gaze_x, _y      : x or y coordinate of either a fixation, Esac or Ssac

    stim_x, _y      : x or y coordinate of the target, distractor, or fix cross
    """

    dist = np.sqrt((gaze_x-stim_x)**2+(gaze_y-stim_y)**2)

    return dist


def calc_latency(gaze_latency, trial_start):
    """
    Calculate the latency of a gaze object (fixation or saccade) by subtracting from it the timestamp for search array onset.

    Arguments

    gaze_latency    : timestamp of a fixation (Sfix) or saccade (Ssac)

    trial_start     : timestamp of search array onset
    """

    latency = gaze_latency - trial_start

    return latency

def sum_stats_grouped(df, target_var, group_var):
    """
    Returns mean, std, and count of a target variable grouped by a group variable,
    dropping NAs in the target variable.

    Arguments:
    
    df:               pandas df

    target_var:  column to summarize

    group_var:   column to group by

    Returns:

    summary:    a pandas df with count, mean, and std for each group
    """
    # Drop rows where the target variable is missing
    df_clean = df[df[target_var].notna()]

    # Group and calculate stats
    summary = df_clean.groupby(group_var)[target_var].agg(['count', 'mean', 'std', 'sem'])

    return summary

def filtered_stats_grouped(df, target_var, group_var, filter_var, filter_value, group=None):
    """
    Returns mean, std, and count of a target variable grouped by a group variable,
    dropping NAs in the target variable and filtering by a filter variable.

    Arguments:
    
    df:             pandas df

    target_var:     column to summarize

    group_var:      column to group by

    filter_var:     column to filter by

    filter_value:   value to filter on

    Returns:

    summary:    a pandas df with count, mean, and std for each group
    """
    # Filter the dataframe based on the filter variable and value
    df_filtered = df[df[filter_var] == filter_value]

    # Drop rows where the target variable is missing
    df_clean = df_filtered[df_filtered[target_var].notna()]

    if group:
        # Group and calculate stats
        summary = df_clean.groupby(group_var)[target_var].agg(['count', 'mean', 'std', 'sem'])

    else:
        summary = df_clean[target_var].agg(['count', 'mean', 'std', 'sem'])
    
    return summary
