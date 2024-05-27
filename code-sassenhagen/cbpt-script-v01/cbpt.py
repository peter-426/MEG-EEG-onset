#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:34:29 2024

author: peter

# code from:  https://osf.io/xf53t

Sassenhagen J, Draschkow D.
Cluster-based permutation tests of MEG/EEG data do not establish significance
of effect latency or location. Psychophysiology. 2019; 56:e13335.
 https://doi.org/10.1111/psyp.13335
"""

# python -c "import mne; print(mne.__version__)"
#conda --version && python --version

# https://github.com/jona-sassenhagen/mne_workshop_amsterdam/tree/master

import os
import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.preprocessing import minmax_scale
import pandas as pd
import seaborn as sns

from multiprocessing import Pool
import tqdm

import mne
mne.set_log_level(False)

# conda --version && python --version
# conda 23.10.0
# Python 3.11.5
# python -c "import mne; print(mne.__version__)"
# 1.6.1
import platform
print("using python version: ", platform.python_version())
print("using mne version: ", mne.__version__)


# In[2]:

from mne.datasets import testing
data_path = testing.data_path()   # PosixPath('/home/phebden/mne_data/MNE-testing-data')
fname   = data_path /  "EEGLAB/test_raw.set"
locs_info_path = data_path / "EEGLAB/test_chans.locs"

#fname =  "sample_data/eeglab_data.set"
#montage ="sample_data/eeglab_chan32.locs"

raw = mne.io.read_raw_eeglab(fname, preload=True, eog=["REF"])

montage = mne.channels.read_custom_montage(locs_info_path)
# import correct channel names
new_chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)
# get old channel names
old_chan_names = raw.info["ch_names"]
# create a dictionary to match old channel names and new (correct) channel names
chan_names_dict = {old_chan_names[i]:new_chan_names[i] for i in range(32)}
# update the channel names in the dataset
raw.rename_channels(chan_names_dict)

raw.set_montage(montage)
ica = mne.preprocessing.ICA(n_components=20, random_state=0)
ica.fit(raw.copy().filter(20, 50))
ica.plot_components() # outlines="skirt");
ica.exclude = [3, 13, 16]
raw = ica.apply(raw, exclude=ica.exclude).filter(.1, 30)
raw.resample(100)
#raw.drop_channels(["STI 014"])
# change: from connectivity(raw.info, "eeg")
conn, names = mne.channels.find_ch_adjacency(raw.info, "eeg")

# In[3]:


topo = ica.get_components()[:, 1]


# In[4]:


pre_stim = np.zeros(15)
post_stim = np.zeros(15)
erp = minmax_scale(norm.pdf(np.linspace(-1.5, 1.5, 21)))

erp = np.hstack((pre_stim, erp, post_stim)) * 1e-5 * 1.5
erp = np.array([erp] * 32) * -topo[:, np.newaxis]

plt.plot(erp.T)
plt.title("our ERP")  # the sharp boundaries are intentional to make the onset objective


# In[16]:


def make_epochs(effsize=1):
    try:
        onset = np.random.uniform()
        raw_ = raw.copy().crop(onset)
        epochs_onset = np.random.choice((0, 1, 2))
        events = mne.make_fixed_length_events(raw_, duration=.5)[epochs_onset::3]
        events = events[sorted(np.random.choice(len(events), size=100, replace=False))]
        events[::2, -1] += 1
        epochs = mne.Epochs(raw_, events, preload=True).apply_baseline().crop(0, .5)
        data = epochs.get_data(copy=True)
        data += (np.array([erp if ii % 2 else np.zeros(erp.shape) for ii, _ in enumerate(events)]) * effsize)
        return epochs
    except Exception:
        return make_epochs(effsize=effsize)


# In[17]:


make_epochs().average().plot_joint();


# In[18]:


make_epochs()["2"].average().plot_joint();


# In[19]:


make_epochs()["1"].average().plot_joint();


# In[77]:


def my_find_cluster_characteristics(epochs):
    data_1,  data_2 = epochs["1"].get_data(copy=False), epochs["2"].get_data(copy=False)
    res = mne.stats.permutation_cluster_test(
        [data_1.swapaxes(1, 2), data_2.swapaxes(1, 2)], n_permutations=1000,
        tail=1, adjacency=None)             #### conn is sparse matrix
    t_obs, clusters, cluster_pv, H0 = res   #### clusters is a list of arrays


    sign_clusters = cluster_pv < .10 # .05
    if not sign_clusters.sum():
        return None  # no sign clusters

    # this is terribly convoluted, but it#s really only looking for the first datapoint
    # included in any significant cluster

    # earliest_sign_datapoint = np.where((np.sum(np.array(clusters)[
    #     np.where(sign_clusters)[0]], 0).sum(1) > 0))[0].min()

    earliest_sign_datapoint = []
    for ii in range(len(sign_clusters)):
        if (sign_clusters[ii]):
            earliest_sign_datapoint.append( min((clusters[ii][0])) )
            earliest_sign_datapoint.append( min((clusters[ii][1])) )


    return earliest_sign_datapoint, sign_clusters.sum()

def make_one_run(ignore, effsize=1):

    #print("effsize=", effsize)

    return my_find_cluster_characteristics(make_epochs(effsize=effsize))



# In[78]:

np.random.seed(seed=426)

# check alpha threshold for p-values.
# given n simulations, a higher alpha allows for more time points

no_pool=True
n = 300
#
# Using laptop:
# if not using Pool, Elapsed time = 546.6 seconds for n=300
# if using Pool, 10000 takes much too long
#
all_results = dict()
effsizes = (.25, .5, .75, 1) # .5, .75, 1, 1.25, 1.5)

start=time.time()

for ii, effsize in enumerate(effsizes):
    def make_one_run_(x):
        return make_one_run(x, effsize=effsize)

    print("processing effect size %0.2f", effsize)

    if no_pool:                                      # bug: n was ignored, fixed PJH
        all_results[ii] = [make_one_run(1, effsize=effsize) for _ in range(n)]
    else:
        with Pool() as p:
            all_results[ii] = list(tqdm.tqdm(p.imap(make_one_run_, range(n)), total=n))

end=time.time()
print("Elapsed time = %0.1f sec, %0.1f sec per run" % (end-start), ((end-start)/n) )

# In[80]:

total=0

for ii in range(len(all_results)):
    result = all_results[ii]
    if result != [None]:
        for r in result:
            if r:
                total += len(r[0])

print("total number of time points = %d" % total)


# In[]

def plot_results(claims, effect_size, fig_num):

    fig, [ax1, ax2] = plt.subplots(ncols=2)
    fig.set_size_inches(12, 4)

    # smallest effect size [0], next smallest [1], ...
    # for smaller effect size, event detection tends to be later (and after real onset)??????

    # 10 ms time steps
    sns.histplot([r * 10 for r in claims ], ax=ax1, kde_kws=dict(bw=10), bins=20, stat='percent')
    ax1.axvline(16 * 10, c="crimson", label="real onset")
    ax1.set_ylabel("cases (%)")
    ax1.set_xlabel("earliest point included in any significant cluster (msec)")

    #ax1.set_yticks(ax1.get_yticks())

    #ax1.set_yticklabels([round(label * 100, 2) for label in ax1.get_yticks()])

    ax1.legend()
    sns.despine()

    ax2.axhline(.05, c="r", label="nominal 5% level")
    ax2.set_ylabel("underestimations (cumulative %)")
    ax2.set_xlabel("degree of underestimation of onset (msec)")

    #s is a []
    s = ((pd.Series(([-min(r - 16, 0) * 10 for r in claims ])).value_counts() / total_claims)
          .sort_index(ascending=False).cumsum().sort_index())[1:]

    #error (where detected onset is before true onset time)
    if s.any():
        s.plot(ax=ax2, kind="bar", color="C1")

    ax2.set_yticks(ax2.get_yticks())
    ax2.set_yticklabels([int(label * 100) for label in ax2.get_yticks()])

    sns.despine()

    fig.suptitle('Onset Detection: Effect Size %0.2f' % effect_size, fontsize=16)

    if (os.path.isdir("figs") == False):
        os.mkdir("figs")
    fig.savefig("figs/clusterf%d.pdf"%fig_num, bbox_inches="tight")

# In[]

# plot results for each effect size to compare
#
for ii in range(len(all_results)):
    result = all_results[ii]     # use index 0, 1, ... but some will be None
    claims = [r[0] for r in result if r]
    total_claims = 0
    temp=[]
    if claims:
        for claim in claims:
            total_claims += len(claim)
            for jj in range(len(claim)):
                if claim[jj] > 0:
                    x=int(claim[jj])
                    temp.extend([x])                # want flat list of time points
        print("total claims =", total_claims)
        if (total_claims > 1):
            #print("rate=", sum([sum(r[0] < 16) for r in claims if r]) / total_claims)
            plot_results(temp, effsizes[ii], ii)

# In[84]:


# n_clus = [r[1] for r in results if r]
# pd.Series(n_clus).value_counts() / len(n_clus) * 100


# In[85]:   ### Errors ###

errors = []
my_effsizes=[]

for ii, effsize in enumerate(effsizes):
    results = all_results[ii]
    if results != [None]:
        my_effsizes.append(effsize)
        claims = [r[0] for r in results if r]
        total_claims = len(claims[0])
        print("len claims = ", total_claims)
        early = sum([r < 16 for r in claims[0] if r]) / total_claims
        late = sum([r > 17 for r in claims[0] if r]) / total_claims
        errors.append((early, late))

# In[86]:

if errors:
    fig=plt.figure()
    fig.set_size_inches(6, 4)
    plt.plot(my_effsizes, errors, label=["early", "late"]);
    plt.legend(loc="best");
    plt.xlabel("effect size");
    plt.title("Errors")

# In[ ]:

