#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:36:50 2024

@author:
"""


import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import h5py

'''
Dataset: https://datadryad.org/stash/dataset/doi:10.5061/dryad.46786

Bieniek, Magdalena M.; Bennett, Patrick J.;
Sekuler, Allison B.; Rousselet, Guillaume A. (2015).
Data from: A robust and representative lower bound on object
processing speed in humans [Dataset].
Dryad. https://doi.org/10.5061/dryad.46786
'''

# In[]

data_path="/home/phebden/Glasgow/DRP-3-code/"


# In[]
#             trials,  time_pts,     electrodes
# eeg_data  # ~ (143,      451,      119)

# single-trial ERPs from 120 participants, 74 of whom were tested twice.
# so, 194 sessions total.
#
# participant 1, session 1.
#


p_idx = 0
p_num = p_idx+1
s_num = 1

e_idx_list=[38,39,40]

file_path = data_path + "/ftonsets_erps_avg/ftonsets_p%d_s%d.mat" % (p_num, s_num)

f3=h5py.File(file_path, 'r')
dataset=f3["/"]
eeg_data = dataset["data"]
times = eeg_data["times"][:]
x_times=times[:].flatten()

face_avg  = eeg_data["face_trials_avg"][:]   # (119, 451)
noise_avg = eeg_data["noise_trials_avg"][:]

for e_idx in e_idx_list:
    ref=eeg_data['chanlocs/labels'][e_idx] # reps A10 or A11
    channel_name = [ chr(ch[0]) for ch in eeg_data[ref[0]] ]
    channel_name= "".join(channel_name)

    plt.figure(figsize=(9,6))
    plt.plot(x_times, face_avg[e_idx],  color='red', label='face', lw=3)
    plt.plot(x_times, noise_avg[e_idx], color= [.7,.7,.7], label='texture', lw=3)
    plt.xlim([-300, 600])
    plt.xlabel("time (ms)", fontsize=22)
    plt.ylabel("amplitude ($\mu$V)", fontsize=22)
    plt.grid(False)
    e_num = e_idx+1
    plt.title("Avg face and texture ERPs: elec %d, %s" % (e_num, channel_name), fontsize=24)
    plt.legend(fontsize=24)

    plt.savefig("avg_ERPs_p%d_s%d_e%s.png" %(p_num, s_num, channel_name), format="png", bbox_inches="tight")


f3.close()