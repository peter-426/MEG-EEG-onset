#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Jun 27 20:27:13 2024

@author: phebden
'''

import matplotlib.pyplot as plt
import numpy as np
import h5py
import time

######################################
# check the updated dataset's contents
######################################
p_idx=11
s_num=2
elec_idx = 10


core_path = '/home/phebden/Glasgow/DRP-3-code'
ses_path    = core_path + "/ftonsets_demographics/ftonsets_2ses.mat"
onsets_path = core_path + "/ftonsets_demographics/ftonsets_onsets.mat"
print(ses_path)
print(onsets_path)

f_ses=h5py.File(ses_path, 'r')
sessions=f_ses["/"]
sessions.keys()
sessions['ftonsets_2ses'].shape   # (1, 120), an array of ones and zeros
session_list=sessions['ftonsets_2ses'][0]

if s_num == 2 and session_list[p_idx] == 0:
    print("paricipant %d did not do sessions 2" % (p_idx+1) )
else:
    f_onsets=h5py.File(onsets_path, 'r')
    onsets=f_onsets["/"]
    onsets.keys()
    onsets['onsets'][0][0]

    erp_path = core_path + "/ftonsets_erps_avg/ftonsets_p%d_s%d.mat" % (p_idx+1, s_num)
    print(erp_path)

    f_erp    = h5py.File(erp_path, 'r')
    dataset  = f_erp["/"]
    eeg_data = dataset["data"]

    print( eeg_data.keys() )

    times = eeg_data["times"]
    x_times=times[:].flatten()

    # If you need to remove an element
    # that was added due to a typo, for example.
    #
    # dataset_key = "erp_trials_avg"
    # if dataset_key in eeg_data:
    #     del eeg_data[dataset_key]

    face_avg  = eeg_data['face_trials_avg']
    noise_avg = eeg_data['noise_trials_avg']

    plt.figure()
    plt.plot(x_times, face_avg[ elec_idx ],     'b', label="face ERP")
    plt.plot(x_times, noise_avg[ elec_idx ], 'r', label="noise ERP")
    plt.title("p%d, s%d: average ERP for electrode %d" % (p_idx+1, s_num, elec_idx+1) )
    plt.legend(["face ERP", "noise ERP"])

    f_erp.close()
    f_ses.close()
    f_onsets.close()