#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:26:20 2024

@author: phebden
"""



import ruptures as rpt
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import h5py
import csv

'''
Dataset: https://datadryad.org/stash/dataset/doi:10.5061/dryad.46786

Bieniek, Magdalena M.; Bennett, Patrick J.;
Sekuler, Allison B.; Rousselet, Guillaume A. (2015).
Data from: A robust and representative lower bound on object
processing speed in humans [Dataset].
Dryad. https://doi.org/10.5061/dryad.46786


In this datset, EEG data recorded while texture images were
presented is referred to as noise, which may be confusing.
This Python code generates plots with such data labelled as "texture".
Face image data is labelled "face", as expected.
'''



# In[]

data_path="/home/phebden/Glasgow/DRP-3-code/"


# In[] load temporal cluster-based onsets from 2023 paper

# 75 onsets here

fname_2ses = '%s/ftonsets_demographics/ftonsets_2ses.mat' % data_path;

f_ses=h5py.File(fname_2ses, 'r')
sessionsHD5=f_ses["/"]
sessionsHD5.keys()
sessionsHD5['ftonsets_2ses'].shape   # (1, 120), an array of ones and zeros
sessions_list=sessionsHD5['ftonsets_2ses'][0]
f_ses.close

#print(sessions_list) # 120 1s and 0s
# *** 75 1s --> 75 participants did 2 sessions
#

results_folder = 'GAR/ftonsets_results_p1_p120_1000_permutations';

s1_onsets_fn = "participants_s1_onsets.txt";
s2_onsets_fn = "participants_s2_onsets.txt";

fname_s1 = '%s/%s/%s' % ( data_path, results_folder, s1_onsets_fn);
fname_s2 = '%s/%s/%s' % ( data_path, results_folder, s2_onsets_fn);

t_cb_s1=[]
t_cb_s2=[]

# just get onsets if participant did sessions 1 and 2
#
with open(fname_s1) as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        ii=0
        for onset in row:
            if sessions_list[ii] == 1:
                t_cb_s1.append(int(onset))
            ii = ii+1


with open(fname_s2) as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        ii=0
        for onset in row:
            if sessions_list[ii] == 1:
                t_cb_s2.append(int(onset))
            ii = ii+1

assert( len(t_cb_s1) == len(t_cb_s2) )


# In[]   load the session info, participant did two sessions or not

file_path = data_path + "ftonsets_demographics/ftonsets_2ses.mat"
f1=h5py.File(file_path, 'r')
sessionsHD5=f1["/"]
sessionsHD5.keys()
sessionsHD5['ftonsets_2ses'].shape   # (1, 120), an array of ones and zeros
session_list=sessionsHD5['ftonsets_2ses'][0]

# figshare onset files, use 3rd column
file_path_onset_120 = data_path + "figshare/data/onset.txt"
file_path_onset_s1 = data_path + "figshare/data/onset1.txt"  # 74
file_path_onset_s2 = data_path + "figshare/data/onset2.txt"  # 74

# f2=h5py.File(file_path, 'r')
# onsets=f2["/"]
# onsets.keys()
# est_onset_2015_list = onsets['onsets'][0][:]  # len = 120

est_onset_120_2015_list=[]
est_onset_s1_2015_list=[]
est_onset_s2_2015_list=[]

with open(file_path_onset_120) as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for row in csv_reader:
        est_onset_120_2015_list.append( int(row[2]) )

with open(file_path_onset_s1) as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for row in csv_reader:
        est_onset_s1_2015_list.append( int(row[2]) )

with open(file_path_onset_s2) as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for row in csv_reader:
        est_onset_s2_2015_list.append( int(row[2]) )

# In[]
#             trials,  time_pts,     electrodes
# eeg_data  # ~ (143,      451,      119)

# single-trial ERPs from 120 participants, 74 of whom were tested twice.
# so, 194 sessions total.
#
# participant 1, session 1.
#

p_idx_start=0
p_idx_stop=91   # <<< only a subset of participants 1 .. 90 did 2 sessions

num_part=75

class Session:
    def __init__(self):
        self.participant_idx_list=[]
        self.face_avg_list=[]
        self.noise_avg_list=[]
        self.max_t_vals_list=[]
        self.max_t_p_vals_list=[]
        self.est_onset_t2_list=[]     # estimated from t2-vals
        self.max_t_val_electrode_idx_list=[]   # used to build session VE
        self.max_t_val_electrode_name_list=[]  # used for max t2 electrode histogram
        self.face_avg_sum=np.zeros(451)
        self.noise_avg_sum=np.zeros(451)
        self.max_t2_vals_sum=np.zeros(451)


sessions = [Session(), Session()]


for p_idx in range(p_idx_start, p_idx_stop):

    if session_list[p_idx] == 0:
        print("participant number %d, no session 2" % (p_idx + 1))
        continue

    for s_idx in [0,1]:

        sessions[s_idx].participant_idx_list.append(p_idx)

        file_path = data_path + "ftonsets_erps_avg/ftonsets_p%d_s%d.mat" % (p_idx+1, s_idx+1)
        if p_idx % 10 == 0:
            print(file_path)

        f3=h5py.File(file_path, 'r')

        dataset=f3["/"]
        eeg_data = dataset["data"]
        times = eeg_data["times"][:]
        x_times=times[:].flatten()

        face_avg  = eeg_data["face_trials_avg"][:]   # e.g.  (119, 451)
        noise_avg = eeg_data["noise_trials_avg"][:]

        # sessions[s_idx].face_avg_sum += face_avg
        # sessions[s_idx].noise_avg_sum += face_avg

        sessions[s_idx].face_avg_list.append(face_avg)
        sessions[s_idx].noise_avg_list.append(noise_avg)

        face_trials  = eeg_data["face_trials"][:]    # (143, 451, 119)
        noise_trials = eeg_data["noise_trials"][:]

        # loaded data into memory using [:] instead of using ref to data file
        # contents, is faster, can close hdf5 files before using that data

        #
        ### test the null hypothesis that the means of two groups are equal ###
        #

        ### construct one virtual electrode per participant

        #### the max electrode recipe ###
        # compute t-tests across trials at every electrode and time point.
        # Save the max t2 values across electrodes -- that's the virtual electrode.
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

            face_cols  = face_trials[:,  :, elec_idx]   # face_cols (143, 451)
            noise_cols = noise_trials[:, :, elec_idx]
            tt = stats.ttest_ind(a=face_cols, b=noise_cols, equal_var=True)

            t_val_list.append( abs(tt.statistic) ) # tt.statistic ~119 x 451
            p_val_list.append( tt.pvalue )

        t_val_2d_array = np.vstack(t_val_list)  # e.g. (119, 451)
        p_val_2d_array = np.vstack(p_val_list)

        max_index = np.unravel_index(np.argmax(t_val_2d_array), t_val_2d_array.shape)
        sessions[s_idx].max_t_val_electrode_idx_list.append(max_index[0])
        ref=eeg_data['chanlocs/labels'][max_index[0]][0] # reps A10 or A11
        channel_name = [ chr(ch[0]) for ch in eeg_data[ref] ]
        channel_name= "".join(channel_name)
        sessions[s_idx].max_t_val_electrode_name_list.append(channel_name)

        # electrode with max t-val also has max t2-val
        max_t_vals = np.max(t_val_2d_array, axis=0)  # (451, )

        #print(">> s%d  %f" % (s_idx, max_t_vals[0]))  # 1 float

        sessions[s_idx].max_t_vals_list.append(max_t_vals)

        # find p-values for two-tailed test
        deg_free = n_trials
        max_t_p_vals = stats.t.sf( abs(max_t_vals), df=deg_free)*2
        sessions[s_idx].max_t_p_vals_list.append(max_t_p_vals)

        f3.close()

f1.close()


# In[]

# Print the variance of both data groups
# print("variance: face trials=%0.3f, noise trials=%0.3f" %
# ( np.var(face_col), np.var(noise_col)) )

# face_avg is a 2d matrix
# face_avg[0] is the avg ERP of the first electrode

#add a row of the face-texture ERP difference.
#Add a reference line at zero in the ERP and t plots.
#add red dots for significant time points in the t plot.

# elec_num = ?    # plot max t-val electrode for example erp

#
# participants 1 to 120, but only 74 did both sessions
#
# do onset predictions
#

sub_idx=-1

for p_idx in range(p_idx_start, p_idx_stop):

    if session_list[p_idx] == 0:
        continue

    if p_idx % 10 == 0:
        print("p_idx = %d" % p_idx)

    sub_idx = sub_idx + 1   # keep track b/c only some participants did 2 sessions

    for s_idx in [0,1]:

        s_num = s_idx+1

        max_t_vals  = sessions[s_idx].max_t_vals_list[sub_idx]
        max_t2_vals = pow(max_t_vals,2)               # max t2

        max_t_p_vals = sessions[s_idx].max_t_p_vals_list[sub_idx]

        algo_name = 'BinSeg'
        mdl='normal'

        algo=rpt.Binseg(model=mdl).fit(max_t2_vals)
        est_onset2 = algo.predict(n_bkps=1)      # 451 time points --> -300..600
        est_onset2 = (est_onset2[0] * 2) - 300   # map to -300..600ms scale
        sessions[s_idx].est_onset_t2_list.append(est_onset2)   # 75 max

s1_length = len(sessions[0].est_onset_t2_list)
s2_length = len(sessions[1].est_onset_t2_list)

if  s1_length > 75:
    print("ERROR session idx 0, lists too long")

if  s2_length > 75:
    print("ERROR session idx 1, lists too long")



# In[]  save face avg erp from max electrode per participant

# these csv files will be used by Matlab fto_demo for permuations
# amd the one virtual electrode per session

max_elec_face_avg_s1_list=[]
max_elec_noise_avg_s1_list=[]

s_idx=0
for p_ii in range(len(sessions[s_idx].face_avg_list)):
    temp=sessions[s_idx].face_avg_list[p_ii]
    max_t_val_electrode_idx  = sessions[s_idx].max_t_val_electrode_idx_list[p_ii]
    max_elec_face_avg_s1_list.append(temp[max_t_val_electrode_idx])
    temp=sessions[s_idx].noise_avg_list[p_ii]
    max_elec_noise_avg_s1_list.append(temp[max_t_val_electrode_idx])

np.savetxt('max_face_avg_s1.csv', max_elec_face_avg_s1_list,   delimiter=',', fmt='%d')
np.savetxt('max_noise_avg_s1.csv', max_elec_noise_avg_s1_list, delimiter=',', fmt='%d')

####
max_elec_face_avg_s2_list=[]
max_elec_noise_avg_s2_list=[]

s_idx=1
for p_ii in range(len(sessions[s_idx].face_avg_list)):
    temp=sessions[s_idx].face_avg_list[p_ii]
    max_t_val_electrode_idx  = sessions[s_idx].max_t_val_electrode_idx_list[p_ii]
    max_elec_face_avg_s1_list.append(temp[max_t_val_electrode_idx])
    temp=sessions[s_idx].noise_avg_list[p_ii]
    max_elec_noise_avg_s1_list.append(temp[max_t_val_electrode_idx])


np.savetxt('max_face_avg_s2.csv', max_elec_face_avg_s2_list,   delimiter=',', fmt='%d')
np.savetxt('max_noise_avg_s2.csv', max_elec_noise_avg_s2_list, delimiter=',', fmt='%d')


# In[]

### sess 1

s_idx=0
s_num= s_idx+1

sessions[s_idx].face_avg_sum=np.zeros(451)
sessions[s_idx].noise_avg_sum=np.zeros(451)
sessions[s_idx].max_t2_vals_sum=np.zeros(451)

#############################################################################
#
### construct one virtual electrode per session
#
# max electrode not the same for all participants
# max_t_val_electrode_name = sessions[s_idx].max_t_val_electrode_name_list[p_idx]


max_t2_vals_list = []

for p_ii in range(len(sessions[s_idx].face_avg_list)):
    temp=sessions[s_idx].face_avg_list[p_ii]
    max_t_val_electrode_idx  = sessions[s_idx].max_t_val_electrode_idx_list[p_ii]
    sessions[s_idx].face_avg_sum += np.array(temp[max_t_val_electrode_idx])  # idx=0 is first electrode

    temp=sessions[s_idx].noise_avg_list[p_ii]
    sessions[s_idx].noise_avg_sum += np.array(temp[max_t_val_electrode_idx])

    max_t_vals  = sessions[s_idx].max_t_vals_list[p_ii]  # session 2
    max_t2_vals = pow(max_t_vals,2)
    max_t2_vals_list.append(max_t2_vals)
    sessions[s_idx].max_t2_vals_sum += max_t2_vals

face_avg_all_s1 = sessions[s_idx].face_avg_sum/s1_length
noise_avg_all_s1 = sessions[s_idx].noise_avg_sum/s1_length
max_t2_avg_all_s1 = sessions[s_idx].max_t2_vals_sum/s1_length

temp_fn_diff=[]


for p_ii in range(num_part):
    max_idx  = sessions[s_idx].max_t_val_electrode_idx_list[p_ii]
    temp_fn_diff.append(np.array(sessions[s_idx].face_avg_list[p_ii][max_idx]) - np.array(sessions[s_idx].noise_avg_list[p_ii][max_idx]))

fn_diff_std_s1 = np.std(temp_fn_diff,0)
max_t2_vals_std_s1=np.std(max_t2_vals_list, 0)

# make sure that t2 band never goes below zero
xxx=max_t2_avg_all_s1 - max_t2_vals_std_s1
for ii in range(len(xxx)):
    if xxx[ii] < 0:
        max_t2_vals_std_s1[ii]=max_t2_avg_all_s1[ii]


# In[]   session 1
#
# CPD predictions from the avg max t2 time series

s_idx=0
s_num= s_idx+1

min_size=2
jump=1

cpd_onsets_s1 = []

### binseg
algo_name = 'BinSeg'
mdl='normal'
algo = rpt.Binseg(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s1)
temp = algo.predict(n_bkps=1)      # 451 time points --> -300..600
est_onset_binseg_s1_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("Binseg s1 t2 onset", est_onset_binseg_s1_t2)
cpd_onsets_s1.append(est_onset_binseg_s1_t2)

### pelt
mdl='rbf'
algo = rpt.Pelt(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s1)
temp = algo.predict(pen=5)       # no default,  penalty 1 --> early, 10 --> late
est_onset_pelt_s1_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("pelt s1 t2 onset", est_onset_pelt_s1_t2)
cpd_onsets_s1.append(est_onset_pelt_s1_t2)

### Window
mdl='l2'
algo = rpt.Window(width=100, model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s1)
temp = algo.predict(n_bkps=1)
est_onset_window_s1_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("Window s1 t2 onset", est_onset_window_s1_t2)
cpd_onsets_s1.append(est_onset_window_s1_t2)

### Dynp
mdl='rbf'
algo = rpt.Dynp(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s1)
temp = algo.predict(n_bkps=1)
est_onset_dynp_s1_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("Dynp s1 t2 onset", est_onset_dynp_s1_t2)
cpd_onsets_s1.append(est_onset_dynp_s1_t2)

### BottomUp
mdl='rbf'
algo = rpt.BottomUp(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s1)
temp = algo.predict(n_bkps=1, pen=None)
est_onset_bottomup_s1_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("BottomUp s1 t2 onset", est_onset_bottomup_s1_t2)
cpd_onsets_s1.append(est_onset_bottomup_s1_t2)
print(cpd_onsets_s1)


# In[]    session 1 plots
#############################

s_idx=0
s_num= s_idx+1

plt.figure(figsize=(8,12))
plt.subplot(3,1,1)
plt.plot(x_times, face_avg_all_s1, label="face ERP")
plt.plot(x_times, noise_avg_all_s1, label="texture ERP")
plt.axhline(0, color='black', linestyle="--", lw=1, alpha=0.5)

plt.xlabel("time (ms)", fontsize=18)
plt.ylabel("amplitude ($\mu$V)", fontsize=18)
plt.legend(fontsize=18)
plt.suptitle("Average for all participants, s%d" % (s_num), fontsize=22 )
plt.title("ERPs from max electrodes",  fontsize=18)

plt.subplot(3,1,2)
face_noise_diff_s1 = face_avg_all_s1 - noise_avg_all_s1
plt.plot(x_times, face_noise_diff_s1,  label="face minus texture ERPs")
plt.fill_between(x_times, face_noise_diff_s1 - fn_diff_std_s1, face_noise_diff_s1 + fn_diff_std_s1, color='b', alpha=0.25)
plt.axhline(0, color='black', linestyle="--", lw=1, alpha=0.5)

plt.xlabel("time (ms)", fontsize=18)
plt.ylabel("amplitude ($\mu$V)", fontsize=18)
plt.legend(fontsize=18)
plt.title("Face minus texture ERPs",  fontsize=18)



plt.subplot(3,1,3)
#plt.plot(below_threshold_times, below_threshold_t_values, '.')

plt.plot(x_times, max_t2_avg_all_s1, 'g')
plt.fill_between(x_times, max_t2_avg_all_s1 - max_t2_vals_std_s1, max_t2_avg_all_s1 + max_t2_vals_std_s1, color='g', alpha=0.25)
plt.axhline(y = 0, color = 'black', lw=1 )

#supported values are '-', '--', '-.', ':', 'None', ' ', '',
# 'solid', 'dashed', 'dashdot', 'dotted'

# plt.axvline(x = est_onset2_st_cb, color = 'blue', ls="-", lw=3, label='ST CB onset' )

est_onset_t_cb_s1_t2=76 # 1000 permutations

plt.axvline(x = est_onset_t_cb_s1_t2,       color = 'black',   ls=":",  lw=3, label='T CB 1000' )
plt.axvline(x = est_onset_binseg_s1_t2,       color = 'red',   ls=":",  lw=3, label='Binseg onset' )
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("time (ms)",    fontsize=18)
plt.ylabel("t$^2$-value",  fontsize=18)
plt.title("average max t$^2$ virtual electrode face vs texture", fontsize=20)
plt.tight_layout()
fn = "p_all_s%d_avg_t2.png" % (s_num)

plt.savefig(fn, format="png", bbox_inches="tight")

# fn = "p_all_s%d_avg_t2.png" % (s_num)
# plt.savefig(fn, format="png", bbox_inches="tight")


# In[]
###################################################
### session 2
###################################################
s_idx=1
s_num=s_idx+1

sessions[s_idx].face_avg_sum=np.zeros(451)
sessions[s_idx].noise_avg_sum=np.zeros(451)
sessions[s_idx].max_t2_vals_sum=np.zeros(451)

# max electrode not the same for all paricipants
#max_t_val_electrode_name = sessions[s_idx].max_t_val_electrode_name_list[p_idx]

max_t2_vals_list=[]

for p_ii in range(len(sessions[s_idx].face_avg_list)):
    temp=sessions[s_idx].face_avg_list[p_ii]
    max_t_val_electrode_idx  = sessions[s_idx].max_t_val_electrode_idx_list[p_ii]
    sessions[s_idx].face_avg_sum += np.array(temp[max_t_val_electrode_idx])  # idx=0 is first electrode

    temp=sessions[s_idx].noise_avg_list[p_ii]
    sessions[s_idx].noise_avg_sum += np.array(temp[max_t_val_electrode_idx])

    max_t_vals  = sessions[s_idx].max_t_vals_list[p_ii]  # session 2
    max_t2_vals = pow(max_t_vals,2)
    max_t2_vals_list.append(max_t2_vals)
    sessions[s_idx].max_t2_vals_sum += max_t2_vals

face_avg_all_s2 = sessions[s_idx].face_avg_sum/s2_length
noise_avg_all_s2 = sessions[s_idx].noise_avg_sum/s2_length
face_noise_diff_s2 = face_avg_all_s2 - noise_avg_all_s2

max_t2_avg_all_s2 = sessions[s_idx].max_t2_vals_sum/s2_length

# face_avg_all_std = np.std(sessions[s_idx].face_avg_list)
# noise_avg_all_std = np.std(sessions[s_idx].noise_avg_list)

temp_fn_diff=[]

for p_ii in range(num_part):
    max_idx  = sessions[s_idx].max_t_val_electrode_idx_list[p_ii]
    temp_fn_diff.append(np.array(sessions[s_idx].face_avg_list[p_ii][max_idx]) - np.array(sessions[s_idx].noise_avg_list[p_ii][max_idx]))

fn_diff_std_s2 = np.std(temp_fn_diff,0)

max_t2_vals_std_s2=np.std(max_t2_vals_list, 0)

xxx=max_t2_avg_all_s2 - max_t2_vals_std_s2
for ii in range(len(xxx)):
    if xxx[ii] < 0:
        max_t2_vals_std_s2[ii]=max_t2_avg_all_s2[ii]

# In[] save both max t2 lists to file

np.savetxt('face_avg_all_s1.csv', face_avg_all_s1, delimiter=',', fmt='%d')
np.savetxt('noise_avg_all_s1.csv', noise_avg_all_s1, delimiter=',', fmt='%d')

np.savetxt('face_avg_all_s2.csv', face_avg_all_s2, delimiter=',', fmt='%d')
np.savetxt('noise_avg_all_s2.csv', noise_avg_all_s2, delimiter=',', fmt='%d')


np.savetxt('max_t2_avg_all_s1.csv', max_t2_avg_all_s1, delimiter=',', fmt='%d')
np.savetxt('max_t2_avg_all_s2.csv', max_t2_avg_all_s2, delimiter=',', fmt='%d')

# In[]
# CPD predictions from the avg max t2 time series

cpd_onsets_s2 = []

min_size=2
jump=1

### binseg
algo_name = 'BinSeg'
mdl='normal'
algo = rpt.Binseg(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s2)
#algo=rpt.Binseg(model=mdl).fit(max_t2_avg_all)
temp = algo.predict(n_bkps=1)      # 451 time points --> -300..600
est_onset_binseg_s2_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("Binseg s2 t2 onset", est_onset_binseg_s2_t2)
cpd_onsets_s2.append( est_onset_binseg_s2_t2 )

### pelt
mdl='rbf'
algo = rpt.Pelt(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s2)
temp = algo.predict(pen=5)       # no default,  penalty 1 --> early, 10 --> late
est_onset_pelt_s2_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("pelt s2 t2 onset", est_onset_pelt_s2_t2)
cpd_onsets_s2.append( est_onset_pelt_s2_t2)

### Window
mdl='l2'
algo = rpt.Window(width=100, model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s2)
temp = algo.predict(n_bkps=1)
est_onset_window_s2_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("Window s2 t2 onset", est_onset_window_s2_t2)
cpd_onsets_s2.append( est_onset_window_s2_t2)

### Dynp
mdl='rbf'
algo = rpt.Dynp(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s2)
temp = algo.predict(n_bkps=1)
est_onset_dynp_s2_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("Dynp s2 t2 onset", est_onset_dynp_s2_t2)
cpd_onsets_s2.append(est_onset_dynp_s2_t2 )

### BottomUp
mdl='rbf'
algo = rpt.BottomUp(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(max_t2_avg_all_s2)
temp = algo.predict(n_bkps=1, pen=None)
est_onset_bottomup_s2_t2 = (temp[0] * 2) - 300   # map to -300..600ms scale
print("BottomUp s2 t2 onset", est_onset_bottomup_s2_t2)
cpd_onsets_s2.append(est_onset_bottomup_s2_t2  )

print(cpd_onsets_s2)

#  scatter plot of CPD onsets for the all virtual electrode

sym_size=300

# cpd_onsets_s1.append(70)  # T CB 2 permutations
# cpd_onsets_s1.append(76)  # T CB 100
cpd_onsets_s1.append(76)  # T CB 1000

# cpd_onsets_s2.append(70)
# cpd_onsets_s2.append(74)
cpd_onsets_s2.append(76)

mn = min(min(cpd_onsets_s1), min(cpd_onsets_s2)) - 5
mx = max(max(cpd_onsets_s1), max(cpd_onsets_s2)) + 5

cpd_names = ['Binseg', 'Pelt', 'Window', 'Dynp', 'BottomUp', 'T CB 1000']
marker_list = ['o',    's',    '^',      '*',      'o',      'v' ]

plt.figure(figsize=(8,8))
for ii in range(len(cpd_names)):
    plt.scatter(cpd_onsets_s1[ii], cpd_onsets_s2[ii], alpha=.5, edgecolor='k', marker=marker_list[ii], s=sym_size, label=cpd_names[ii])

plt.plot([mn, mx], [mn, mx])
plt.xlabel("session 1 predicted onset (ms)", size=18)
plt.ylabel("session 2 predicted onset (ms)", size=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([mn,mx])
plt.ylim([mn,mx])
plt.legend(prop={'size': 18})

plt.title('Predicted onsets, one VE per session: s1 and s2', fontsize=22)

fn = "p_all_s1_s2_scatter.png"
plt.savefig(fn, format="png", bbox_inches="tight")

# fn = "p_all_s1_s2_scatter.png"
# plt.savefig(fn, format="png", bbox_inches="tight")






# In[]     all
##########################
#       session 2
##########################

s_idx=1
s_num=s_idx+1

plt.figure(figsize=(8,12))
plt.subplot(3,1,1)
plt.plot(x_times, face_avg_all_s2,  label="face ERP")
plt.plot(x_times, noise_avg_all_s2, label="texture ERP")
plt.axhline(0, color='black', linestyle="--", lw=1, alpha=0.5)

plt.xlabel("time (ms)", fontsize=18)
plt.ylabel("amplitude ($\mu$V)", fontsize=18)
plt.legend(fontsize=18)
plt.suptitle("Average for all participants, s%d" % (s_num), fontsize=22 )
plt.title("ERPs from max electrodes",  fontsize=18)

plt.subplot(3,1,2)
face_noise_diff = face_avg_all_s2 - noise_avg_all_s2
plt.plot(x_times, face_noise_diff_s2,  label="face minus texture ERPs")
plt.fill_between(x_times, face_noise_diff_s2 - fn_diff_std_s2, face_noise_diff_s2 + fn_diff_std_s2, color='b', alpha=0.25)
plt.axhline(0, color='black', linestyle="--", lw=1, alpha=0.5)

plt.xlabel("time (ms)", fontsize=18)
plt.ylabel("amplitude ($\mu$V)", fontsize=18)
plt.legend(fontsize=18)
plt.title("Face minus texture ERPs",  fontsize=18)


plt.subplot(3,1,3)
plt.plot(x_times, max_t2_avg_all_s2, 'g')
plt.fill_between(x_times, max_t2_avg_all_s2 - max_t2_vals_std_s2, max_t2_avg_all_s2 + max_t2_vals_std_s2, color='g', alpha=0.25)

#plt.plot(below_threshold_times, below_threshold_t_values, '.')

plt.axhline(y = 0, color = 'black', lw=1 )

#supported values are '-', '--', '-.', ':', 'None', ' ', '',
# 'solid', 'dashed', 'dashdot', 'dotted'

# plt.axvline(x = est_onset2_st_cb, color = 'blue', ls="-", lw=3, label='ST CB onset' )
est_onset_t_cb_s2_t2=76 # 1000 permutations

plt.axvline(x = est_onset_t_cb_s2_t2,    color='black', ls=":", lw=3, label='T CB 1000' )
plt.axvline(x = est_onset_binseg_s2_t2,  color='red',   ls=":", lw=3, label='Binseg onset' )
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("time (ms)", fontsize=18),
plt.ylabel("t$^2$-value",  fontsize=18)
plt.title("average max t$^2$ virtual electrode face vs texture", fontsize=20)
plt.tight_layout()
fn = "p_all_s%d_avg_t2.png" % (s_num)

plt.savefig(fn, format="png", bbox_inches="tight")

# fn = "p_all_s%d_avg_t2.png" % (s_num)
# plt.savefig(fn, format="png", bbox_inches="tight")




# In[]  plot EXAMPLE ERPs, onset times for sessions 1 & 2

###############################################################
# p_num accounts for zero based index and skipped participants
# (those that did not do both sessions
###############################################################
p_idx = 1

# get index of participant that did 2 sessions, +1
p_num = sessions[0].participant_idx_list[p_idx] + 1


###########
# session 1
###########

s_num = 1

face_avg  = sessions[0].face_avg_list[p_idx]
noise_avg = sessions[0].noise_avg_list[p_idx]
max_t_val_electrode_name = sessions[0].max_t_val_electrode_name_list[p_idx]
max_t_val_electrode_idx  = sessions[s_idx].max_t_val_electrode_idx_list[p_idx]

# max t2 electrode ERPs
plt.figure(figsize=(8,12))
temp="Participant %d, session %d" % (p_num, s_num)
plt.suptitle(temp, fontsize=22)

plt.subplot(3,1,1)
plt.plot(x_times, face_avg[max_t_val_electrode_idx ],  label = "face ERP"  )   # idx 0 is first electrode
plt.plot(x_times, noise_avg[max_t_val_electrode_idx ], label = "texture ERP" )
plt.axhline(y = 0, color = 'black', lw=1)
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("amplitude ($\mu$V)", fontsize=18)
plt.title("average ERPs: max t$^2$ electrode %s" % (max_t_val_electrode_name), fontsize=20)

#
# ERPs difference
#
plt.subplot(3,1,2)
fn_diff = face_avg[0] - noise_avg[0]
plt.plot(x_times, fn_diff, label="face minus texture"  )
plt.axhline(y = 0, color = 'black', lw=1 )
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("amplitude ($\mu$V)", fontsize=18)
plt.title("face minus texture ERPs: max t$^2$ electrode %s" % (max_t_val_electrode_name), fontsize=20 )


plt.subplot(3,1,3)
# t-vals
max_t_vals  = sessions[0].max_t_vals_list[p_idx]
max_t2_vals = pow(max_t_vals,2)
# onsets predicted from t-vals
est_onset2=sessions[0].est_onset_t2_list[p_idx]
est_onset2_st_cb = est_onset_s1_2015_list[p_idx]
est_onset2_t_cb = t_cb_s1[p_idx]

plt.plot(x_times, max_t2_vals, 'g')

#plt.plot(below_threshold_times, below_threshold_t_values, '.')

plt.axhline(y = 0, color = 'black', lw=1 )

#supported values are '-', '--', '-.', ':', 'None', ' ', '',
# 'solid', 'dashed', 'dashdot', 'dotted'

plt.axvline(x = est_onset2_st_cb, color = 'blue', ls="-", lw=3, label='ST CB onset' )
plt.axvline(x = est_onset2_t_cb,  color = 'orange', ls="--", lw=3, label='T CB onset' )
plt.axvline(x = est_onset2,       color = 'red',   ls=":", lw=3, label='Binseg onset' )
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("time (ms)", fontsize=18),
plt.ylabel("t$^2$-value",  fontsize=18)
plt.title("max t$^2$ virtual electrode face vs texture", fontsize=20)
plt.tight_layout()

fn = "p%d_s%d.png" % (p_num, s_num)

plt.savefig(fn, format="png", bbox_inches="tight")


###########
# session 2
###########
s_num=2


face_avg  = sessions[1].face_avg_list[p_idx]
noise_avg = sessions[1].noise_avg_list[p_idx]
max_t_val_electrode_name = sessions[1].max_t_val_electrode_name_list[p_idx]
max_t_val_electrode_idx  = sessions[s_idx].max_t_val_electrode_idx_list[p_idx]

plt.figure(figsize=(8,12))
temp="Participant %d, session %d" % (p_num, s_num)
plt.suptitle(temp, fontsize=22)

plt.subplot(3,1,1)
plt.plot(x_times, face_avg[max_t_val_electrode_idx],  label="face ERP"  )
plt.plot(x_times, noise_avg[max_t_val_electrode_idx], label="texture ERP" )
plt.axhline(y = 0, color = 'black', lw=1 )
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("amplitude ($\mu$V)", fontsize=18)
plt.title("average ERPs: max t$^2$ electrode %s" %  (max_t_val_electrode_name), fontsize=20)


# example ERPs difference
#
plt.subplot(3,1,2)
fn_diff = face_avg[0] - noise_avg[0]
plt.plot(x_times, fn_diff, label="face minus texture"  )

plt.axhline(y = 0, color = 'black', lw=1 )
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("amplitude ($\mu$V)", fontsize=18)
plt.title("face minus texture ERPs: max t$^2$ electrode %s " % (max_t_val_electrode_name), fontsize=20)


plt.subplot(3,1,3)
# t-vals
max_t_vals  = sessions[1].max_t_vals_list[p_idx]  # session 2
max_t2_vals = pow(max_t_vals,2)
# onsets predicted from t-vals
est_onset2=sessions[1].est_onset_t2_list[p_idx]
est_onset2_st_cb = est_onset_s2_2015_list[p_idx]
est_onset2_t_cb = t_cb_s2[p_idx]

#supported values are '-', '--', '-.', ':', 'None', ' ', '',
# 'solid', 'dashed', 'dashdot', 'dotted'

plt.plot(x_times, max_t2_vals , 'g')
plt.axhline(y = 0, color = 'black', lw=1 )
plt.axvline(x = est_onset2_st_cb, color = 'blue',   ls="-", lw=3, label='ST CB onset' )
plt.axvline(x = est_onset2_t_cb,  color = 'orange', ls="--", lw=3, label='T CB onset' )
plt.axvline(x = est_onset2,       color = 'red',    ls=":", lw=3, label='Binseg onset' )
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("time (ms)", fontsize=18)
plt.ylabel("t$^2$-value", fontsize=18)
plt.title("max t$^2$ virtual electrode face vs texture", fontsize=20)
plt.tight_layout()

fn = "p%d_s%d.png" % (p_num, s_num)

plt.savefig(fn, format="png", bbox_inches="tight")


# In[]

# Example
p_idx=0
s_idx=0
file_path = data_path + "ftonsets_erps_avg/ftonsets_p%d_s%d.mat" % (p_idx+1, s_idx+1)
print(file_path)

f3=h5py.File(file_path, 'r')
dataset=f3["/"]
eeg_data = dataset["data"]
ref=eeg_data['chanlocs/labels'][0][0] # A11
ch_name = [ chr(ch[0]) for ch in eeg_data[ref] ]
channel_name= "".join(ch_name)
print(f"eeg_data['chanlocs/labels'][0][0] electrode channel name={channel_name}")
f3.close()


# In[]

temp1=np.array(sessions[0].max_t_val_electrode_name_list)
temp2=np.array(sessions[1].max_t_val_electrode_name_list)
data = np.concatenate([temp1, temp2])
kk = np.unique(data)
print("There are %d unique predicted onset times" % len(kk))

# Initialize a dictionary to store counts
count_dict = {i: 0 for i in kk}

# Count occurrences of each value
for ch_name in data:
    count_dict[ch_name] += 1

# Sort the counts in descending order
sorted_counts = sorted(count_dict.items(), key=lambda item: item[1], reverse=True)

total_count=0

# Print the sorted counts
for value, count in sorted_counts:
        # ref=eeg_data['chanlocs/labels'][value][0] # A11
        # ch_name = [ chr(ch[0]) for ch in eeg_data[ref] ]
        # channel_name= "".join(ch_name)
        print(f"electrode {value}: {count} times")
        total_count = total_count + count

print("total count = %d" % total_count)

labels = [pair[0] for pair in sorted_counts]
counts = [pair[1] for pair in sorted_counts]

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color='blue')
plt.xlabel("electrode channel names", fontsize=14)
plt.ylabel("count", fontsize=14)
plt.title("Max t-value electrode counts", fontsize=16)

plt.savefig("max-t-val-electrode-counts.png", format="png", bbox_inches="tight")
#plt.savefig("max-t-val-electrode-counts.png", format="png", bbox_inches="tight")
##########################################################
# Using channel names, zero index into eeg_data channel loc
# electrode B8: 33 times
# electrode B10: 19 times
# electrode B7: 17 times
# electrode B9: 16 times
# electrode A11: 11 times
# electrode A28: 9 times
# electrode A10: 7 times
# electrode A14: 6 times
# electrode B6: 5 times
# electrode D32: 4 times
# electrode A29: 3 times
# electrode B11: 3 times
# electrode A15: 2 times
# electrode A16: 2 times
# electrode A23: 2 times
# electrode A25: 2 times
# electrode A27: 2 times
# electrode A12: 1 times
# electrode A13: 1 times
# electrode A18: 1 times
# electrode A22: 1 times
# electrode A24: 1 times
# electrode A31: 1 times
# electrode A7: 1 times







