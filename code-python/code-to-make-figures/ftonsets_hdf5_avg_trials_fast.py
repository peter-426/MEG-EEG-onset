#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 00:59:26 2024

@author: peter
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import time


'''
Dataset: https://datadryad.org/stash/dataset/doi:10.5061/dryad.46786

Bieniek, Magdalena M.; Bennett, Patrick J.;
Sekuler, Allison B.; Rousselet, Guillaume A. (2015).
Data from: A robust and representative lower bound on object
processing speed in humans [Dataset].
Dryad. https://doi.org/10.5061/dryad.46786
'''

# In[]

core_path = '/home/phebden/Glasgow/DRP-3-code'
ses_path    = core_path + "/ftonsets_demographics/ftonsets_2ses.mat"
onsets_path = core_path + "/ftonsets_demographics/ftonsets_onsets.mat"
print(ses_path)
print(onsets_path)

f1=h5py.File(ses_path, 'r')
sessions=f1["/"]
sessions.keys()
sessions['ftonsets_2ses'].shape   # (1, 120), an array of ones and zeros
session_list=sessions['ftonsets_2ses'][0]


f2=h5py.File(onsets_path, 'r')
onsets=f2["/"]
onsets.keys()
onsets['onsets'][0][0]

# f_erp is for the many erp data files, see below

# In[]

# single-trial ERPs from 120 participants, 74 of whom were tested twice.
# so, 194 sessions total.
#
# participant 1, session 1.
# participant 1, session 2.
# participant 2, session 1,
# ...

n_participants=len(session_list)

for p_idx in range( n_participants ):
    for s_num in [1, 2]:
        # check if there was a session 2.
        # if not, do next participant
        if s_num == 2 and session_list[p_idx] == 0:
            break

        start_time = time.time()

        erp_path = core_path + "/ftonsets_erps_avg/ftonsets_p%d_s%d.mat" % (p_idx+1, s_num)

        print(erp_path)

        f_erp=h5py.File(erp_path, 'r+')
        dataset=f_erp["/"]
        eeg_data = dataset["data"]

        est_onset_a = onsets['onsets'][0][p_idx]

        est_onset = eeg_data['onset_in_ms']
        est_onset = est_onset[0][0]

        print("onset list %0.1f  vs  eeg_data onset_in_ms %0.1f" % ( est_onset_a, est_onset  ))

        times = eeg_data["times"]
        x_times=times[:].flatten()

        ########################################
        # face trials
        ########################################
        # for each paricipant
        #   for each session
        #     for each electrode
        #       calculate and store the avg ERP in its hdf5 file.
        # then do same for noise trials
                                              # trials,  time_pts, electrodes
        face_trials = eeg_data["face_trials"] # ~ (143,       451,      119)

        num_electrodes = face_trials.shape[2]
        print("calc avg for each of the %d electrodes" % num_electrodes)

        erp_avg = np.mean(face_trials, axis=0)
        erp_avg_list = erp_avg.T              # one row for each electrode

        dataset_key = 'face_trials_avg'
        if dataset_key in eeg_data:
            del eeg_data[ dataset_key ]
        eeg_data["face_trials_avg"] = np.vstack(erp_avg_list)

        #######################################
        # noise trials
        #######################################
        #
        # in numpy, axis=0 means operate vertically by row, e.g. calc column means
        #           axis 1 means operate horizontally by col, e.g. calc row means
        #
        noise_trials = eeg_data["noise_trials"]
        erp_avg = np.mean(noise_trials, axis=(0))
        erp_avg_list = erp_avg.T                   # one row for each electrode

        dataset_key = 'noise_trials_avg'
        if dataset_key in eeg_data:
            del eeg_data[ dataset_key ]
        eeg_data["noise_trials_avg"] = np.vstack(erp_avg_list)

                    # close file because we just finished saving the
                    # avg erp Over all trials for each electrode
        f_erp.close()  # for the p_num, s_num session.
        print("p%d, s%d time to avg %d electrodes over trials --- %0.2f seconds ---" %
              (p_idx+1, s_num, num_electrodes, time.time() - start_time))



f1.close()
f2.close()
f_erp.close()


