#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 01:22:55 2024

@author: phebden
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created 2024

@author: peter
'''

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

file_path = data_path + "ftonsets_demographics/ftonsets_2ses.mat"
f1=h5py.File(file_path, 'r')
sessions=f1["/"]
sessions.keys()
sessions['ftonsets_2ses'].shape   # (1, 120), an array of ones and zeros
session_list=sessions['ftonsets_2ses'][0]

file_path = data_path + "ftonsets_demographics/ftonsets_onsets.mat"
f2=h5py.File(file_path, 'r')
onsets=f2["/"]
onsets.keys()
onsets['onsets'][0][0]

#             trials,  time_pts,     electrodes
# eeg_data  # ~ (143,      451,      119)

# In[]

# single-trial ERPs from 120 participants, 74 of whom were tested twice.
# so, 194 sessions total.
#
# participant 1, session 1.
#

ii=0
jj=0

file_path = data_path + "ftonsets_erps/ftonsets_p%d_s%d.mat" % (ii+1, jj+1)
print(file_path)
f3=h5py.File(file_path, 'r')
dataset=f3["/"]
eeg_data = dataset["data"]
times = eeg_data["times"]
x_times=times[:].flatten()

est_onset_a = onsets['onsets'][0][ii]
est_onset_b = x=eeg_data['onset_in_ms']
est_onset_b = est_onset_b[0][0]

f1.close()
f2.close()

face_avg  = eeg_data["face_trials_avg"][:]   # (119, 451)
noise_avg = eeg_data["noise_trials_avg"][:]

face_trials  = eeg_data["face_trials"][:]    # (143, 451, 119)
noise_trials = eeg_data["noise_trials"][:]

f3.close()

#
### test the null hypothesis that the means of two groups are equal ###
#

#### the max electrode recipe ###
# compute t-tests across trials at every electrode and time point.
# Save the max t2 values across electrodes -- that's your virtual electrode.
#
# In a permutation test, shuffle trials between conditions and repeat the
# same steps. Use the permutation distribution to derive a cluster-forming
# threshold, which is then applied to the permutation distributions and the
# original data.


n_electrodes = face_trials.shape[2]

n_trials = face_trials.shape[0]
n_time_pts = face_trials.shape[1]

t_val_list = []
p_val_list = []


for elec_idx in range(n_electrodes):

    # get trials for this electrode and permute the trials
    face_cols  = face_trials[:,  :, elec_idx]   # face_cols (143, 451)
    noise_cols = noise_trials[:, :, elec_idx]

    all_trials= np.vstack([face_cols, noise_cols])
    # Generate a random permutation of row indices
    permuted_idx = np.random.permutation(all_trials.shape[0])

    all_trials = all_trials[permuted_idx , :]
    face_cols  = all_trials[0:143, : ]
    noise_cols = all_trials[143:,  : ]

    tt = stats.ttest_ind(a=face_cols, b=noise_cols, equal_var=True)

    t_val_list.append( abs(tt.statistic) ) # tt.statistic ~119 x 451
    p_val_list.append( tt.pvalue )


t_val_2d_array = np.vstack(t_val_list)  # (119, 451)
p_val_2d_array = np.vstack(p_val_list)

# max of all rows for each column
max_t_vals = np.max( t_val_2d_array, axis=0)  # (451, )

# find p-values for two-tailed test
deg_free = n_trials
max_t_p_vals = stats.t.sf(abs(max_t_vals), df=deg_free)*2


# Print the variance of both data groups
# print("variance: face trials=%0.3f, noise trials=%0.3f" %
# ( np.var(face_col), np.var(noise_col)) )

# face_avg is a 2d matrix
# face_avg[0] is the avg ERP of the first electrode

#add a row of the face-noise ERP difference.
#Add a reference line at zero in the ERP and t plots.
#add red dots for significant time points in the t plot.

elec_num=1
p_num=1
s_num=1

#face_noise_diff = face_avg[0] - noise_avg[0]

# threshold=0.05

# below_threshold_idx      = p_val_list[0] < threshold
# below_threshold_t_values = t_val_list[below_threshold_idx]
# below_threshold_times    = x_times[below_threshold_idx]

# zeroeth electrode is electrode 1

plt.figure(figsize=(8,12))
plt.subplot(4,1,1)
plt.plot(x_times, face_avg[0], label="face"  )
plt.plot(x_times, noise_avg[0], label = "noise" )
plt.axhline(y = 0, color = 'black', lw=1 )
plt.axvline(x = est_onset_a, color = 'black', lw=2 )
plt.axvline(x = est_onset_b, color = 'red', ls="dotted", lw=2  )
plt.ylabel("amplitude")
plt.title("participant %d, session %d, electrode %d" % (p_num, s_num, elec_num))
plt.legend(["face ERP", "noise ERP"])

# plt.subplot(4,1,2)
# plt.plot(x_times, face_noise_diff)
# plt.axhline(y = 0, color = 'black', lw=1 )
# #plt.xlabel("time (ms)"),
# plt.ylabel("erp difference")
# plt.title("face electrode ERP minus noise electrode ERP")

plt.subplot(4,1,3)
plt.plot(x_times, max_t_vals)
#plt.plot(below_threshold_times, below_threshold_t_values, '.')
plt.axhline(y = 0, color = 'black', lw=1 )
#plt.xlabel("time (ms)"),
plt.ylabel("t-value")
plt.title("t-values for virtual face electrode versus noise electrode")

plt.subplot(4,1,4)
plt.plot(x_times, max_t_p_vals , 'g')
plt.xlabel("time (ms)"),
plt.ylabel("p-value")
plt.title("p-values for virtual face electrode vs noise electrode")

plt.tight_layout()

### max t2 analysis

# 1) Combine Data:
#
#   Combine the ERP data for both conditions (face and noise) into a
#   single dataset.
#
# 2) Permute Labels:
#
#   Randomly shuffle the condition labels to create a new permutation.
#   This means reassigning each trial as either face or noise randomly while
#   keeping the overall number of face and noise trials the same.
#
# 3) Compute Permutation t-Values:
#   For each permutation, compute the t-values for each time point and electrode,
#   and then square these t-values to create a new 119x451 matrix of squared t-values.
#
# 4) Create Permutation Max t2 Vector:
#
#   For the permuted dataset, calculate the maximum squared t-value (t2) for each
#   time point, resulting in a new 1x451 vector.
#
# 5) Repeat Permutations:
#
#   Repeat the permutation steps (2-4) a large number of times (e.g., 1000 or more)
#   to build a null distribution of max t2 values for each time point.
#
# 6) Determine Significance:
#
#   Compare the observed max t2 vector from the original data to the null
#   distribution of max t2 vectors.
#
#   For each time point, determine the p-value by finding the proportion of
#   permuted max t2 values that are greater than or equal to the observed max t2 value.
#
#   Adjust for multiple comparisons, if necessary, using methods such as Bonferroni
#   correction or False Discovery Rate (FDR).







