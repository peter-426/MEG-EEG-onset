#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Jul  4 00:57:21 2024

@author: peter
'''

# the ERP data for the ftonsets project are available here:
# https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.46786

# 2 systems: 128 electrodes and 256 electrodes

# https://figshare.com/articles/dataset/Face_noise_ERP_onsets_from_194_recording_sessions/1588513

import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import numpy as np
import scipy.stats as sp
#from scipy import stats
import csv
import h5py

# In[] onsets for sessions 1 and 2

class session_onsets:
    def __init__(self):
        self.s1=[]
        self.s2=[]

spatio_temporal_cb_obj = session_onsets()
temporal_cb_obj = session_onsets()
bs_obj          = session_onsets()   # binseg

data_path = '/home/phebden/Glasgow/DRP-3-code';


# In[] load spatio-temporal onsets from 2016 paper

# 74 onsets here

file_path_onset_s1 = data_path + "/figshare/data/onset1.txt"  # 74
file_path_onset_s2 = data_path + "/figshare/data/onset2.txt"  # 74

with open(file_path_onset_s1) as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for row in csv_reader:
        spatio_temporal_cb_obj.s1.append( int(row[2]) )

with open(file_path_onset_s2) as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for row in csv_reader:
        spatio_temporal_cb_obj.s2.append( int(row[2]) )


# In[] load temporal cluster-based onsets from 2023 paper

# 75 onsets here

fname_2ses = '%s/ftonsets_demographics/ftonsets_2ses.mat' % data_path;

f_ses=h5py.File(fname_2ses, 'r')
sessions=f_ses["/"]
sessions.keys()
sessions['ftonsets_2ses'].shape   # (1, 120), an array of ones and zeros
sessions_list=sessions['ftonsets_2ses'][0]
f_ses.close

#print(sessions_list) # 120 1s and 0s
# *** 75 1s --> 75 participants did 2 sessions
#

results_folder = 'GAR/ftonsets_results_p1_p120_1000_permutations';

s1_onsets_fn = "participants_s1_onsets.txt";
s2_onsets_fn = "participants_s2_onsets.txt";

fname_s1 = '%s/%s/%s' % ( data_path, results_folder, s1_onsets_fn);
fname_s2 = '%s/%s/%s' % ( data_path, results_folder, s2_onsets_fn);

# just get onsets if participant did sessions 1 and 2
#
with open(fname_s1) as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        ii=0
        for onset in row:
            if sessions_list[ii] == 1:
                temporal_cb_obj.s1.append(int(onset))
            ii = ii+1


with open(fname_s2) as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        ii=0
        for onset in row:
            if sessions_list[ii] == 1:
                temporal_cb_obj.s2.append(int(onset))
            ii = ii+1


# In[]  ********** BinSeg predicted onsets ****
###############################################
# compare predicted onsets for sessions 1 vs 2.
###############################################


do_plots=False

time_ms=list(range(-300,601,2)) # length = 451

mdl="normal"

print("*** Binseg cost function = %s ***" % mdl)

for p_idx in range(120):
    p_num = p_idx+1

    for s_num in [1,2]:

        # did this participant do both sessions?
        if sessions_list[p_idx] == 0:
            break;

        # 451x1
        max_t2=[]

        fn = "participant_p%d_s%d_maxt2.txt" % (p_num, s_num);
        fname_max_t2 = '%s/%s/%s' % (data_path, results_folder, fn);
        with open(fname_max_t2) as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                max_t2.append(float(row[0]))

        n_bkps=1
        temp = np.array(max_t2)
        algo = rpt.Binseg(model=mdl, custom_cost=None, min_size=2, jump=1, params=None).fit(temp)
        est_onset = algo.predict(n_bkps=n_bkps) # 451 time points --> -300..600ms
        est_onset = (est_onset[0] * 2) - 300    # map to -300..600ms scale
                                                # * 2 to inflate to the -300..600 scale
                                                # -300 for baseline

        # there were two sessions for this participant (see check done above)
        if s_num==1:
            bs_obj.s1.append(est_onset)
        else:
            bs_obj.s2.append(est_onset) # must be session 2

        if do_plots:
            plt.figure(figsize=(8,6));
            plt.plot(time_ms, max_t2, color='red');
            plt.axvline(est_onset)
            plt.title(fn)

# In[]

# convert lists of onsets to numpy arrays

t_cb_s1 = np.array(temporal_cb_obj.s1)
t_cb_s2 = np.array(temporal_cb_obj.s2)

bs_s1 = np.array(bs_obj.s1)
bs_s2 = np.array(bs_obj.s2)

st_cb_s1 = np.array(spatio_temporal_cb_obj.s1)
st_cb_s2 = np.array(spatio_temporal_cb_obj.s2)

mn1 = min(min(t_cb_s1), min(t_cb_s2))
mx1 = max(max(t_cb_s1), max(t_cb_s2))


mn2 = min(min(bs_s1), min(bs_s2))
mx2 = max(max(bs_s1), max(bs_s2))

mn3 = min(min(st_cb_s1), min(st_cb_s2))
mx3 = max(max(st_cb_s1), max(st_cb_s2))

mn=min(mn1, mn2, mn3) - 5
mx=max(mx1, mx2, mx3) + 5


# In[]
###################### scatter plots ########################################

sym_size=50

# marker="X"

import scipy.stats as stats

###
### 2016 spatio-temporal figshare data onsets
###
st_pred_diff = abs(st_cb_s1 - st_cb_s2)
st_mad = np.mean(st_pred_diff)
st_median_ad =stats.median_abs_deviation(st_pred_diff)

med1a = np.median(st_cb_s1)
med2a = np.median(st_cb_s2)

plt.figure(figsize=(8,8))
plt.scatter(st_cb_s1, st_cb_s2, color='red',
            alpha=.35, edgecolor='k', s=sym_size, label='spatio-temporal CB')
plt.plot([mn, mx], [mn, mx])

plt.plot([mn, med1a], [med1a, med1a], color='black', label='median')
plt.plot([med2a, med2a], [mn, med2a], color='black')


plt.xlabel("session 1 predicted onset (ms)", size=18)
plt.ylabel("session 2 predicted onset (ms)", size=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([mn,mx])
plt.ylim([mn,mx])
plt.legend(prop={'size': 18})

plt.title('Spatio-temporal cluster onsets', fontsize=22)
plt.savefig("t2_analysis_st_cb_scatter.png", format="png", bbox_inches="tight")


print('Spatio-temporal cluster-based onsets: MAD=%0.2f' % st_mad)
print('Spatio-temporal cluster-based onsets: std=%0.2f' % np.std(st_pred_diff))
print('Spatio-temporal cluster-based onsets: median_ad=%0.2f' % st_median_ad)
###
### temporal cluster-based
###
t_pred_diff = abs(t_cb_s1 - t_cb_s2)
t_mad = np.mean(t_pred_diff)
t_median_ad =stats.median_abs_deviation(t_pred_diff)

med1b = np.median(t_cb_s1)
med2b = np.median(t_cb_s2)

plt.figure(figsize=(8,8));
plt.scatter(t_cb_s1, t_cb_s2, color='green',
            alpha=.35,  edgecolor='k', s=sym_size, label='temporal CB');
plt.plot([mn, mx], [mn, mx])
plt.plot([mn, med1b], [med1b, med1b], color='black', label='median')
plt.plot([med2b, med2b], [mn, med2b], color='black')

plt.xlabel("session 1 onset (ms)", size=18)
plt.ylabel("session 2 onset (ms)", size=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([mn,mx])
plt.ylim([mn,mx])
plt.legend(prop={'size': 18})

plt.title("Temporal cluster-based onsets", fontsize=22)
plt.savefig("t2_analysis_t_cb_scatter.png", format="png", bbox_inches="tight")

print('Temporal cluster-based onsets: MAD=%0.2f' % t_mad)
print('Temporal cluster-based onsets: std=%0.2f' % np.std(t_pred_diff))
print('Temporal cluster-based onsets: median_ad=%0.2f' % t_median_ad)
###
### binseg
###
bs_pred_diff = abs(bs_s1 - bs_s2)
bs_mad = np.mean(bs_pred_diff)
bs_median_ad =stats.median_abs_deviation(bs_pred_diff)

med1c = np.median(bs_s1)
med2c = np.median(bs_s2)

plt.figure(figsize=(8,8))
plt.scatter(bs_s1, bs_s2, color='blue', alpha=.35, edgecolor='k',
            s=sym_size, label='Binseg')
plt.plot([mn, mx], [mn, mx])

plt.plot([mn, med1c],    [med1c, med1c], color='black', label='median')
plt.plot([med2c, med2c], [mn, med2c],    color='black')

plt.xlabel("session 1 predicted onset (ms)", size=18)
plt.ylabel("session 2 predicted onset (ms)", size=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([mn,mx])
plt.ylim([mn,mx])
plt.legend(prop={'size': 18})

plt.title('BinSeg predicted onsets', fontsize=22)
plt.savefig("t2_analysis_binseg_scatter.png", format="png", bbox_inches="tight")

print('BinSeg predicted: MAD=%0.2f' % bs_mad)
print('BinSeg predicted: std=%0.2f' % np.std(bs_pred_diff))
print('BinSeg predicted: median_ad=%0.2f' % bs_median_ad)


# In[]

import seaborn
#import pandas as pd

seaborn.set(style = 'whitegrid')

st_pred_diff = st_cb_s1 - st_cb_s2
t_pred_diff  = t_cb_s1 - t_cb_s2
bs_pred_diff = bs_s1 - bs_s2


labels = ["ST CB", "T CB", "Binseg"]
data= [st_pred_diff, t_pred_diff, bs_pred_diff]

#plt.axhline(algo_obj.true_onset)

# Plot the violin plots
plt.figure(figsize=(10, 8))
violin = sns.violinplot(data=data, palette=['red', 'green', 'blue'])

# Set the alpha (transparency) for each violin
for patch in violin.collections:
    patch.set_alpha(0.35)  # Adjust alpha value

# Set the x-axis labels
plt.xticks(ticks=range(len(labels)), labels=labels, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("time (ms)", fontsize=22)
plt.xlabel("onset detection method", fontsize=22)

# Add a title
plt.suptitle('Distribution of prediction differences\n s1 - s2', fontsize=24)
#plt.title('dataset %d' % dataset_num, fontsize=20)
fn = "session_pred_diffs_violins.png"

plt.savefig(fn, format="png", bbox_inches="tight")


# In[]


#######################################################

##################### super-imposed
# spatio-temporal

for ii in range(3):
    plt.figure(figsize=(8,8))

    if ii >= 0:
        plt.scatter(st_cb_s1, st_cb_s2, color='red',
                    alpha=.35, edgecolor='k', s=sym_size, label='spatio-temporal CB')
        plt.plot([mn, med1a],    [med1a, med1a],  color='red', label='ST CB median')
        plt.plot([med2a, med2a], [mn, med2a], color='red'  )

    if ii >= 1:
        # temporal
        plt.scatter(t_cb_s1, t_cb_s2, color='green',
                    alpha=.35,  edgecolor='k', s=sym_size, label='temporal CB');
        plt.plot([mn, med1b],    [med1b, med1b],  color='green', label='T CB median')
        plt.plot([med2b, med2b], [mn, med2b], color='green'  )

    if ii >= 2:
        # binseg
        plt.scatter(bs_s1, bs_s2, color='blue', alpha=.35, edgecolor='k',
                    s=sym_size, label='Binseg')
        plt.plot([mn, med1c],    [med1c, med1c],  color='blue', label='Binseg median')
        plt.plot([med2c, med2c], [mn, med2c], color='blue'  )

    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("session 1 predicted onset (ms)", size=20)
    plt.ylabel("session 2 predicted onset (ms)", size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([mn,mx])
    plt.ylim([mn,mx])
    plt.legend(prop={'size': 18})

    plt.title('Predicted onset times', fontsize=24)
    plt.savefig("t2_analysis_scatter_superimposed_%d.png" % ii, format="png", bbox_inches="tight")


# In[]
################## kernel plots #########################
#
########## spatio-temporal kernel plot

n1=len(st_cb_s1)
n2=len(st_cb_s2)

mu1 =np.mean(st_cb_s1)
med1=np.median(st_cb_s1)
std1=np.std(st_cb_s1)

mu2 =np.mean(st_cb_s2)
med2=np.median(st_cb_s2)
std2=np.std(st_cb_s2)

# centered so can't align floats
ses1=f"session 1: n={n1}, mean={mu1:>4.2f}, median={med1}, std={std1:>4.2f} ms"
ses2=f"session 2: n={n2}, mean={mu2:>4.2f}, median={med2}, std={std2:>4.2f} ms"

print("ST CB: " + ses1)
print("ST CB: " + ses2)

plt.figure(figsize=(10,8))
sns.kdeplot(st_cb_s1, bw_adjust=0.5, fill=True, label="session 1")
sns.kdeplot(st_cb_s2, bw_adjust=0.5, fill=True, label="session 2")
ax1 = sns.kdeplot(st_cb_s1, bw_adjust=0.5, fill=False)
ax2 = sns.kdeplot(st_cb_s2, bw_adjust=0.5, fill=False)
plt.axvline(med1, color='blue',   linestyle="--", label='session 1 median', lw=5)
plt.axvline(med2, color='orange', linestyle=":",  label='session 2 median', lw=5)
# Extract the line data
x1, y1 = ax1.get_lines()[0].get_data();
x2, y2 = ax2.get_lines()[0].get_data();
# Find the x value corresponding to the maximum y value
x1_peak = x1[np.argmax(y1)]
x2_peak = x2[np.argmax(y2)]

print(' ')
print("ST CB Session 1 peak x-value: %0.1f" % x1_peak)
print("ST CB Session 2 peak x-value: %0.1f" % x2_peak)

plt.legend(prop={'size': 18})
plt.xlim([30,180])
plt.ylim([0,0.04])
plt.xlabel("time (ms)", size=18)
plt.ylabel("density",   size=18)
plt.xticks(fontsize=16)
plt.yticks([])

plt.title("Spatio-temporal cluster-based onsets", fontsize=20)
plt.savefig("t2_analysis_st_cb_kernel.png", format="png", bbox_inches="tight")

# In[]
##### temporal cluster-based kernel plot

n1=len(t_cb_s1)
n2=len(t_cb_s2)

med1=np.median(t_cb_s1)
mu1 =np.mean(  t_cb_s1)
std1 = np.std( t_cb_s1)

med2 =np.median(t_cb_s2)
mu2  =np.mean(  t_cb_s2)
std2 =np.std(   t_cb_s2)

# centered so can't align floats
ses1=f"session 1: n={n1}, mean={mu1:>4.2f}, median={med1}, std={std1:>4.2f} ms"
ses2=f"session 2: n={n2}, mean={mu2:>4.2f}, median={med2}, std={std2:>4.2f} ms"

print("T CB: " + ses1)
print("T CB: " + ses2)

plt.figure(figsize=(10,8))

sns.kdeplot(t_cb_s1, bw_adjust=0.5, fill=True, label="session 1")
sns.kdeplot(t_cb_s2, bw_adjust=0.5, fill=True, label="session 2")
ax1 = sns.kdeplot(t_cb_s1, bw_adjust=0.5, fill=False)
ax2 = sns.kdeplot(t_cb_s2, bw_adjust=0.5, fill=False)
plt.axvline(med1, color='blue', linestyle="--", label='session 1 median', lw=5)
plt.axvline(med2, color='orange', linestyle=':', label='session 2 median', lw=5)
# Extract the line data
x1, y1 = ax1.get_lines()[0].get_data();
x2, y2 = ax2.get_lines()[0].get_data();
# Find the x value corresponding to the maximum y value
x1_peak = x1[np.argmax(y1)]
x2_peak = x2[np.argmax(y2)]

print("T CB Session 1 peak x-value: %0.1f" % x1_peak)
print("T CB Session 2 peak x-value: %0.1f" % x2_peak)

plt.legend(prop={'size': 18})
plt.xlim([30,180])
plt.ylim([0,0.04])
plt.xlabel("time (ms)", size=18)
plt.ylabel("density",   size=18)
plt.xticks(fontsize=16)
plt.yticks([])

plt.title("Temporal cluster-based onsets", fontsize=22)
plt.savefig("t2_analysis_t_cb_kernel.png", format="png", bbox_inches="tight")


# In[]
######### binseg kernel plot

n1=len(bs_s1)
n2=len(bs_s2)
mu1 =np.mean(bs_s1)
med1=np.median(bs_s1)
std1=np.std(bs_s1)

mu2 =np.mean(bs_s2)
med2=np.median(bs_s2)
std2=np.std(bs_s2)

# centered so can't align floats
ses1=f"session 1: n={n1}, mean={mu1:>4.2f}, median={med1}, std={std1:>4.2f} ms"
ses2=f"session 2: n={n2}, mean={mu2:>4.2f}, median={med2}, std={std2:>4.2f} ms"

print("BS: " + ses1)
print("BS: " + ses2)

algo_name="BinSeg"
mdl="normal"

### get peaks too

plt.figure(figsize=(10, 8))
sns.kdeplot(bs_s1, bw_adjust=0.5, fill=True, label="session 1")
sns.kdeplot(bs_s2, bw_adjust=0.5, fill=True, label="session 2")
ax1 = sns.kdeplot(bs_s1, bw_adjust=0.5, fill=False)
ax2 = sns.kdeplot(bs_s2, bw_adjust=0.5, fill=False)
plt.axvline(med1, color='blue',   linestyle="--",  label='session 1 median', lw=5)
plt.axvline(med2, color='orange', linestyle=":", label='session 2 median', lw=5)
# Extract the line data
x1, y1 = ax1.get_lines()[0].get_data();
x2, y2 = ax2.get_lines()[0].get_data();
# Find the x value corresponding to the maximum y value
x1_peak = x1[np.argmax(y1)]
x2_peak = x2[np.argmax(y2)]

print("BS Session 1 peak x-value: %0.1f" % x1_peak)
print("BS Session 2 peak x-value: %0.1f" % x2_peak)

plt.xlim([30,180])
plt.ylim([0,0.04])
plt.xlabel("time (ms)", size=18)
plt.ylabel("density",   size=18)
plt.xticks(fontsize=16)
plt.yticks([])
plt.legend(prop={'size': 18})

plt.title("%s predicted onsets, cost function=%s" % (algo_name, mdl), fontsize=20)
plt.savefig("t2_analysis_binseg_kernel.png", format="png", bbox_inches="tight")

print('\n')

# In[]
#
### three methods with combined sessions
#

algo_name="BinSeg"
mdl="normal"

### get peaks too

st_cb_s1_s2 = np.concatenate([st_cb_s1, st_cb_s2])
t_cb_s1_s2  = np.concatenate([ t_cb_s1,  t_cb_s2])
bs_s1_s2    = np.concatenate([   bs_s1,    bs_s2])

st_cb_median_s1_s2 = np.median(st_cb_s1_s2)
t_cb_median_s1_s2  = np.median(t_cb_s1_s2)
bs_median_s1_s2 = np.median(bs_s1_s2)

mu1 =np.mean(  st_cb_s1_s2)
med1=np.median(st_cb_s1_s2)
std1=np.std(   st_cb_s1_s2)

mu2 =np.mean(  t_cb_s1_s2)
med2=np.median(t_cb_s1_s2)
std2=np.std(   t_cb_s1_s2)

mu3 =np.mean(  bs_s1_s2)
med3=np.median(bs_s1_s2)
std3=np.std(   bs_s1_s2)

meth1=f"spatio-temporal: n={148}, mean={mu1:>4.2f}, median={med1}, std={std1:>4.2f} ms\n"
meth2=f"temporal:        n={150}, mean={mu2:>4.2f}, median={med2}, std={std2:>4.2f} ms\n"
meth3=f"Binseg:          n={150}, mean={mu3:>4.2f}, median={med3}, std={std3:>4.2f} ms"

print("both sessions combined")
print(meth1 + meth2 + meth3)

plt.figure(figsize=(10, 8))
sns.kdeplot(st_cb_s1_s2, bw_adjust=0.5, color='red', fill=True, label="ST CB s1 & s2")
sns.kdeplot( t_cb_s1_s2, bw_adjust=0.5, color='green', fill=True, label="T CB s1 & s2")
sns.kdeplot(   bs_s1_s2, bw_adjust=0.5, color='blue', fill=True, label="Binseg s1 & s2")

plt.axvline(st_cb_median_s1_s2, color='red',    linestyle="--", label='ST CB median', lw=5)
plt.axvline(t_cb_median_s1_s2,  color='green', linestyle=":",  label='T CB median', lw=5)
plt.axvline(bs_median_s1_s2,  color='blue', linestyle=":",  label='Binseg median', lw=5)


ax1 = sns.kdeplot(st_cb_s1_s2, bw_adjust=0.5, fill=False)
ax2 = sns.kdeplot( t_cb_s1_s2, bw_adjust=0.5, fill=False)
ax3 = sns.kdeplot(   bs_s1_s2, bw_adjust=0.5, fill=False)


# ax2 = sns.kdeplot(bs_s2, bw_adjust=0.5, fill=False)
# plt.axvline(med1, color='blue',   label='session 1 median', lw=5)
# plt.axvline(med2, color='orange', label='session 2 median', lw=5)
# Extract the line data

x1, y1 = ax1.get_lines()[0].get_data();
x2, y2 = ax2.get_lines()[1].get_data();
x3, y3 = ax3.get_lines()[2].get_data();

# Find the x value corresponding to the maximum y value

x1_peak = x1[np.argmax(y1)]
x2_peak = x2[np.argmax(y2)]
x3_peak = x3[np.argmax(y3)]

print("ST CB:  Session 1 & 2 peak x-value: %0.1f" % x1_peak)
print("T CB:   Session 1 & 2 peak x-value: %0.1f" % x2_peak)
print("Binseg: Session 1 & 2 peak x-value: %0.1f" % x3_peak)

plt.xlim([30,180])
plt.ylim([0,0.04])
plt.xlabel("time (ms)", fontsize=18)
plt.ylabel("density",   fontsize=18)
plt.xticks(fontsize=16)
plt.yticks([])
plt.legend(prop={'size': 18})

plt.title("Predicted onsets, both sessions", fontsize=22)
plt.savefig("t2_analysis_3_methods_kernel.png", format="png", bbox_inches="tight")

print('\n')

# In[]

#########################################################################
###################### stats ############################################
#########################################################################

# see below
# sp.t.ppf
#
# Percent point function (inverse of `cdf`) at q of the given RV.
#
# Parameters
# ----------
# q : array_like
#     lower tail probability
# arg1, arg2, arg3,... : array_like
#     The shape parameter(s) for the distribution (see docstring of the
#     instance object for more information)
# loc : array_like, optional
#     location parameter (default=0)
# scale : array_like, optional
#     scale parameter (default=1)

# Returns
# -------
# x : array_like
#     quantile corresponding to the lower tail probability q.

# In[]
##################
### cluster based

# bias
def bias_func(s1, s2):
      pred_diff = s1 - s2
      bias = np.mean(pred_diff)
      return bias

# mean squared diff
def msd_func(s1, s2):
    pred_diff = s1 - s2
    msd = sum(np.pow(pred_diff, 2))/len(pred_diff)
    return msd

# CI
def CI_func(s1, s2):
    # Standard deviation of the differences
    std_diff = np.std(s1 - s2, ddof=1)
    n = len(t_cb_s1)

    # Confidence interval for the mean difference (Bias)
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_crit = sp.t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_crit * (std_diff / np.sqrt(n))
    confidence_interval = (bias - margin_of_error, bias + margin_of_error)
    return confidence_interval


# In[]
### spatio-temporal cluster-based  predictions #########################

#########################
# mean squared difference
msd = msd_func(st_cb_s1, st_cb_s2)
print("Spatio-temporal cluster-based mean squared difference (s1 - s2), MSD = %0.2f" % msd)

########################
# mean difference (bias)
bias = bias_func(st_cb_s1, st_cb_s2)
print("Spatio-emporal cluster-based predictions for s1 v s2, bias = %0.2f" % bias)

#########################
# CI
confidence_interval = CI_func(st_cb_s1, st_cb_s2)
print("95%% Confidence Interval for bias: (%0.2f, %0.2f)" %
      (confidence_interval[0], confidence_interval[1]) )
print("===============================================")


# In[]
#########################
# mean squared difference
msd = msd_func(t_cb_s1, t_cb_s2)
print("Temporal cluster-based mean squared difference (s1 - s2), MSD = %0.2f" % msd)

########################
# mean difference (bias)
bias = bias_func(t_cb_s1, t_cb_s2)
print("Temporal cluster-based predictions for s1 v s2, bias = %0.2f" % bias)

#########################
# CI
confidence_interval = CI_func(t_cb_s1, t_cb_s2)
print("95%% Confidence Interval for bias: (%0.2f, %0.2f)" %
      (confidence_interval[0], confidence_interval[1]) )

print("===============================================")


# In[]
### Binseg predictions ##################################################

#########################
# mean squared difference
msd = msd_func(bs_s1, bs_s2)
print("Binseg mean squared difference (s1 - s2), MSD = %0.2f" % msd)

########################
# mean difference (bias)
bias = bias_func(bs_s1, bs_s2)
print("Binseg predictions for s1 v s2, bias = %0.2f" % bias)

#########################
# CI
confidence_interval = CI_func(bs_s1, bs_s2)
print("95%% Confidence Interval for bias: (%0.2f, %0.2f)" %
      (confidence_interval[0], confidence_interval[1]) )
print("===============================================")


# In[]

# Variances of onset predictions: temporal cluster-based vs binseg

# two ways to calc confidence interval

# here using both s1 and s2 onsets for each method


# In[]
from scipy.stats import f

# Chi-square distribution is used for constructing confidence intervals for
# variances of normally distributed data, not just categorical variables.
# Confidence intervals for variances help quantify the uncertainty around
# your variance estimates.

def compare_variance(measurements_A, measurements_B, method_A, method_B):

    # Calculate sample variances
    s_A2 = np.var(measurements_A, ddof=1)
    s_B2 = np.var(measurements_B, ddof=1)

    # Calculate F-statistic
    F = s_A2 / s_B2

    # Degrees of freedom
    df_A = len(measurements_A) - 1
    df_B = len(measurements_B) - 1

    # Critical value for F at alpha = 0.05 (one-tailed test)
    alpha = 0.05
    critical_value = f.ppf(1 - alpha, df_A, df_B)

    # P-value
    p_value = 1 - f.cdf(F, df_A, df_B)

    print("Variances of onset predictions: %s var=%0.3f, %s var=%0.3f" % (method_A, s_A2, method_B, s_B2))
    print("F-statistic:    %4.3f" % F)
    print("Critical value: %4.3f" % critical_value)
    print("P-value:        %4.3f" % p_value)

    print("The null hypothesis is that the variances are equal.")
    if F > critical_value:
        print("Reject the null hypothesis: colnclude that the variances are statistically different.")
    else:
        print("Fail to reject the null hypothesis: No evidence that the variances are not equal.")


# Combine measurements from each method

measurements_A = np.concatenate([st_cb_s1, st_cb_s2])
measurements_B = np.concatenate([t_cb_s1, t_cb_s2])
measurements_C = np.concatenate([bs_s1, bs_s2])


compare_variance(measurements_A, measurements_B, "ST CB", "T CB")
print("=================================================================")
compare_variance(measurements_A, measurements_C, "ST CB", "Binseg")
print("=================================================================")
compare_variance(measurements_B, measurements_C, "T CB", "Binseg")

# In[]


from scipy.stats import chi2

# predicted onsets for 75 participants, 2 sessions, 2 methods

# A refers to cluster-based method
# B refers to BinSeg method

use_diffs=0

if use_diffs:
    st_cb_s1_s2 = st_cb_s1 - st_cb_s2
    t_cb_s1_s2 = t_cb_s1 - t_cb_s2
    bs_s1_s2 = bs_s1 - bs_s2
else:
    st_cb_s1_s2 = np.concatenate([st_cb_s1, st_cb_s2])
    t_cb_s1_s2 = np.concatenate([t_cb_s1, t_cb_s2])
    bs_s1_s2 = np.concatenate([bs_s1, bs_s2])

# Calculate sample variances,
# ddof: “Delta degrees of freedom”: adjustment to the degrees of freedom for the p-value.
s_A2 = np.var(st_cb_s1_s2, ddof=1)
s_B2 = np.var(t_cb_s1_s2, ddof=1)
s_C2 = np.var(bs_s1_s2, ddof=1)

# Sample sizes
n_A = len(st_cb_s1_s2)
n_B = len(t_cb_s1_s2)
n_C = len(bs_s1_s2)

# Degrees of freedom
df_A = n_A - 1
df_B = n_B - 1
df_C = n_C - 1

# Confidence level
alpha = 0.05

# Critical values from Chi-square distribution
# ppf: Percent point function (inverse of `cdf`) at q of the given RV.
chi2_lower_A = chi2.ppf(alpha / 2, df_A)
chi2_upper_A = chi2.ppf(1 - alpha / 2, df_A)

chi2_lower_B = chi2.ppf(alpha / 2, df_B)
chi2_upper_B = chi2.ppf(1 - alpha / 2, df_B)

chi2_lower_C = chi2.ppf(alpha / 2, df_C)
chi2_upper_C = chi2.ppf(1 - alpha / 2, df_C)

# Confidence intervals for the variances
ci_A = ((df_A * s_A2) / chi2_upper_A, (df_A * s_A2) / chi2_lower_A)
ci_B = ((df_B * s_B2) / chi2_upper_B, (df_B * s_B2) / chi2_lower_B)
ci_C = ((df_C * s_C2) / chi2_upper_C, (df_C * s_C2) / chi2_lower_C)

# Determine which methods has lower variance

print("Sample variance of ST cluster-based: %4.3f" % s_A2)
print("Sample variance of T cluster-based: %4.3f" % s_B2)
print("Sample variance of BinSeg: %4.3f" % s_C2)

print("95%% confidence interval for the variance of A: (%4.3f, %4.3f)" % ci_A)
print("95%% confidence interval for the variance of B: (%4.3f, %4.3f)" % ci_B)
print("95%% confidence interval for the variance of B: (%4.3f, %4.3f)" % ci_C)



# In[] ### correlation #######################

# % default alpha = 0.05
# % corrcoef(A,'Alpha',0.1) specifies a 90% confidence interval

#
### cluster-based method:
#

# corrcoef and p-value
pr = sp.pearsonr(t_cb_s1, t_cb_s2)
sr = sp.spearmanr(t_cb_s1, t_cb_s2)
kt = sp.kendalltau(t_cb_s1, t_cb_s2)

print(' ')
print("cluster-based predicted onsets: Pearson correlation of s1 with s2: r=%0.3f, p-val=%0.3f" %
      (pr.statistic, pr.pvalue))
print("cluster-based predicted onsets: Spearman correlation of s1 with s2: r=%0.3f, p-val=%0.3f" %
      (sr.statistic, sr.pvalue))
print("cluster-based predicted onsets: Kendall's tau correlation of s1 with s2: r=%0.3f, p-val=%0.3f" %
      (kt.statistic, kt.pvalue))

# [R,P,RL,RU]=corrcoef(s1_temp, s2_temp);
# fprintf("correlation=%0.3f\n\n", R(1,2));
# fprintf("p-value=%0.3f\n\n", P(1,2));
# fprintf("RL=%0.3f\n\n", RL(1,2));
# fprintf("RU=%0.3f\n\n", RU(1,2));
# bins=list(range(100,400,10))


# In[]

# plt.figure(figsize=(10,6))
# plt.subplot(1,2,1)
# plt.hist(bs_s1, bins=bins)
# plt.xlim(100, 400)
# plt.subplot(1,2,2)
# plt.hist(bs_s2, bins=bins)
# plt.xlim(100, 400)

pr = sp.pearsonr(bs_s1, bs_s2)
sr = sp.spearmanr(bs_s1, bs_s2)
kt = sp.kendalltau(bs_s1, bs_s2)


print(' ')
print("BinSeg predicted onsets: Pearson R correlation of s1 with s2: r=%0.3f,  p-val=%0.3f" %
      (pr.statistic, pr.pvalue))
print("BinSeg predicted onsets: Spearman R correlation of s1 with s2: r=%0.3f, p-val=%0.3f" %
      (sr.statistic, sr.pvalue))
print("BinSeg predicted onsets: Kendall's tau correlation of s1 with s2: r=%0.3f, p-val=%0.3f" %
      (kt.statistic, kt.pvalue))



