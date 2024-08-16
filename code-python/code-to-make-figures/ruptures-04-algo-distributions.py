#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Jul 10 02:55:13 2024

@author: peter
'''


import matplotlib.pylab as plt
import ruptures as rpt
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import minmax_scale
from scipy.stats import norm

import time
import math

import data_gen

# In[]

# set simulation parameters
#
dataset_num=4 # <<<<<<<<<<--------------------------------------------

PI = math.pi
my_rand_seed = 12345

n_samples=451

margin=10   # margin of error for false discovery rate (fdr): FP/(TP+FP)
            # when detection is too late, e.g. > 10ms after true onset time

num_signals = 100  # number of trials in simulation

n_bkps=1
true_onset=160

times = np.arange(0, n_samples, 1).tolist()

# Note: A new Pelt algorithm object must created for each trial.
# This might also be true for other algorithm implementations.
# Negligible performance hit, so no need to optimise.

# algorithms
algo_dict= {'BinSeg':1, 'Pelt':2, 'Window':3, 'Dynp':4, 'BottomUp': 5}

#
# Parameters for grid search
#
# linear cost function req > 1 dimension.
# mdl_list =['l1', 'l2', 'normal', 'rbf', 'cosine', 'clinear', 'rank', 'mahalanobis', 'ar']

# minimum segment size
min_size_list = [2]    # 2, 5, 10 ... 50


class Algo:
    def __init__(self):
        self.algo_name=""
        self.model = 'l2'
        self.min_size = 2
        self.jump = 1
        self.pred_list = []
        self.MAE=10000000
        self.rmse=10000000
        self.bias=10000000
        self.fdr=10000000
        self.pct_too_early=100          #
        self.true_onset=true_onset    # default


def run_algo(algo_name, dataset_num):
    obj_list=[]

    print('model name= ', algo_name)

    if algo_name == 'BinSeg':
        mdl_list =['l1', 'l2', 'rbf', 'normal' ] # , 'cosine', 'clinear', 'rank', 'mahalanobis', 'ar']
    else:
        mdl_list = ['l1', 'l2', 'rbf'] # dims must be > 1 to use linear


    for mdl in mdl_list:   # 9 models (or 10 if add linear)
        print("model = ", mdl)
        for ii in range(len(min_size_list)):

            algo_obj = Algo()
            algo_obj.algo_name=algo_name
            algo_obj.model=mdl
            algo_obj.pred_list = []
            min_size = algo_obj.min_size
            predictions = []

            rng = np.random.default_rng(my_rand_seed)
            n_samples=451
            true_onset=160

            erp_len = 451-(true_onset-1)
            time_points = np.arange(erp_len)/500

            for kk in range(num_signals):
                # generate signal
                if dataset_num==1:
                    signal, bkps = data_gen.make_data_1(n_samples)
                    algo_obj.true_onset=bkps[0]
                    if False:
                        plt.plot(signal)
                        plt.axvline(algo_obj.true_onset)

                if dataset_num==2:
                    mu, sigma = 0, 1 # mean and standard deviation for simulated data generation
                    mu2, sigma2 = mu + 1, sigma + 0.5
                    signal = data_gen.make_data_2(n_samples, algo_obj.true_onset, mu, sigma, mu2, sigma2, rng)
                    if False:
                        plt.plot(signal)
                        plt.axvline(algo_obj.true_onset)

                if dataset_num==3:
                    # mu, sigma = 0, 1 # mean and standard deviation
                    # inc = rng.uniform(0.75,1.25)
                    # mu2, sigma2 = mu + inc, sigma + 0.5
                    scale=0.05
                    signal = data_gen.make_data_3(time_points, algo_obj.true_onset, rng, scale)
                    #signal = abs(signal)
                    if False:
                       plt.plot(signal)
                       plt.axvline(algo_obj.true_onset)

                if dataset_num==4:
                    lowcut = 1.0  # Low frequency cut-off
                    highcut = 30.0  # High frequency cut-off
                    fs = 500.0  # Sampling frequency
                    order = 5   # Filter order

                    scale = 0.05
                    signal = data_gen.make_data_4(time_points, algo_obj.true_onset, rng, scale)
                    signal = data_gen.bandpass_filter_causal(signal, lowcut, highcut, fs, order)
                    #signal=abs(signal)
                    # if kk==0:
                    #     plt.plot(signal)


                jump=algo_obj.jump   # == 1

                if algo_name == 'BinSeg':
                    algo = rpt.Binseg(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(signal)
                    est_onset = algo.predict(n_bkps=n_bkps)
                    predictions.append(est_onset[0] )    # *** off by 1?, add 1 to onset for this data?
                elif algo_name == 'Pelt':
                    algo = rpt.Pelt(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(signal)
                    est_onset = algo.predict(pen=5)       # no default,  penalty 1 --> early, 10 --> late
                    predictions.append(est_onset[0])
                elif algo_name == 'Window':
                    algo = rpt.Window(width=100, model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(signal)
                    est_onset = algo.predict(n_bkps=n_bkps)
                    predictions.append(est_onset[0])
                elif algo_name == 'Dynp':
                    algo = rpt.Dynp(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(signal)
                    est_onset = algo.predict(n_bkps=n_bkps)
                    predictions.append(est_onset[0])
                elif algo_name == 'BottomUp':
                    algo = rpt.BottomUp(model=mdl, custom_cost=None, min_size=min_size, jump=jump, params=None).fit(signal)  # written in C
                    est_onset = algo.predict(n_bkps=n_bkps, pen=None)
                    predictions.append(est_onset[0])
                else:
                    print("algo not found")
                    #return -1

            algo_obj.pred_list.append(predictions)
            obj_list.append(algo_obj)
    algo_dict[algo_name]=obj_list


#
#

def get_too_early(predictions, true_onset):
    too_early = predictions[predictions < true_onset]
    too_early = np.abs(too_early - true_onset)
    return too_early


def get_pct_too_early(predictions, true_onset):
    too_early = predictions[predictions < true_onset]
    pct_too_early = 100 * (len(too_early)/len(predictions))
    return pct_too_early


def get_MAE(predictions, true_onset):
    MAE = np.mean(np.abs(predictions - true_onset))
    return MAE


def get_std(predictions, true_onset):
    std = np.std(predictions)
    return std

def get_rmse(predictions, true_onset):
    rmse = np.sqrt(np.mean(np.square(predictions - true_onset)))
    return rmse

def get_fdr(predictions, true_onset, margin):
    errors = predictions - true_onset
    early_count = sum(errors < 0)      # FP
    late_count = sum(errors > margin)  # FP
    fdr = (early_count + late_count)/len(predictions)
    return fdr

def get_bias(predictions, true_onset):
    bias=np.mean(predictions) - true_onset
    return bias


# precision = TP/(TP + FP)
def get_precision(predictions, true_onset, margin):
    errors = predictions - true_onset
    early_count = sum(errors < 0)      # FP
    late_count = sum(errors > margin)  # FP
    precision = (len(predictions) - (early_count + late_count))/len(predictions)
    return precision


# Same as precision if one onset prediction for each
# time series where each time series has one true onset.
# Not really any FNs by definition because FNs == FPs.
#
# recall = TP/(TP+FN)
def get_recall(predictions, margin):
    errors = predictions - true_onset
    early_count = sum(errors < 0)      # FP
    late_count = sum(errors > margin)  # FP
    recall = (len(predictions) - (early_count + late_count))/len(predictions)
    return recall

# F1 = 2 * (precision * recall) / (precision + recall).
def get_f1(precision, recall):
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1

def plot_predictions(predictions, title):
    pass

# BinSeg
#
# https://dev.ipol.im/~truong/ruptures-docs/build/html/_modules/ruptures/detection/binseg.html
# algorithm defaults: model='l2', custom_cost=None, min_size=2, jump=5, params=None

# Pelt
#
# https://dev.ipol.im/~truong/ruptures-docs/build/html/detection/pelt.html
# algorithm defaults: model='l2', custom_cost=None, min_size=2, jump=5, params=None


# Window
#
# https://dev.ipol.im/~truong/ruptures-docs/build/html/detection/window.html
# algorithm defaults: width=100, model='l2', custom_cost=None, min_size=2, jump=5, params=None


# Dynp
#
# https://dev.ipol.im/~truong/ruptures-docs/build/html/detection/dynp.html
# algorithm defaults: model='l2', custom_cost=None, min_size=2, jump=5, params=None

# KernelCPD
#
# Efficient kernel change point detection
# Available kernels: linear, rbf, cosine
#
# algorithm defaults: kernel='linear', min_size=2, jump=5, params=None
#
# KernelCPD is implemented in C and is fast, so jump=1 and cannot be changed
# because increasing jump does not improve performance. KernelCPD is uses the
# same algorithm as Pelt?

# BottomUP
#
# https://dev.ipol.im/~truong/ruptures-docs/build/html/detection/bottomup.html
# available models = l1, l2, rbf
# algorithm defaults: model='l2', custom_cost=None, min_size=2, jump=5, params=None
# predict defaults: n_bkps=None, pen=None, epsilon=None


start_time = time.time()

start=time.time()

run_algo('BinSeg', dataset_num)
print("BinSeg %0.2f minutes\n" % ((time.time() - start)/60))

start=time.time()
run_algo('Pelt', dataset_num)
print("Pelt %0.2f minutes\n" % ((time.time() - start)/60))

start=time.time()
run_algo('BottomUp', dataset_num)
print("BottomUp %0.2f minutes\n" % ((time.time() - start)/60))

start=time.time()
run_algo('Window', dataset_num)
print("Window %0.2f minutes\n" % ((time.time() - start)/60))

start=time.time()
run_algo('Dynp', dataset_num)
print("Dynp %0.2f minutes\n" % ((time.time() - start)/60))

print("---TOTAL %s minutes ---\n" % ((time.time() - start_time)/60))


# In[]

algo_dict.keys()

# In[]

verbose = False

for alg in algo_dict.keys():
    obj_list = algo_dict[alg]

    if type(obj_list) == int:
        print("obj_list == int ", obj_list)
        continue

    #print("\n>>> " + obj_list[0].algo_name + " <<<")
    for ii in range(len(obj_list)):
        algo_obj=obj_list[ii]
        preds= np.array(algo_obj.pred_list)[0]
        algo_obj.MAE  = get_MAE(preds,      algo_obj.true_onset);
        algo_obj.rmse  = get_rmse(preds,     algo_obj.true_onset)
        algo_obj.bias = get_bias(preds,     algo_obj.true_onset)
        #algo_obj.fdr  = get_fdr(preds,      algo_obj.true_onset, margin)
        algo_obj.pct_too_early = get_pct_too_early(preds, algo_obj.true_onset)
        if verbose==True:
            print("model = %-12s min_size=%d, MAE=%6.2f,  pct_too_early=%6.2f" % \
                  (algo_obj.model, algo_obj.min_size, algo_obj.MAE,  algo_obj.pct_too_early ))
    print(len(obj_list) )

# In[]
verbose = False

best_algo_dict = { }

best_algo_dict['BinSeg']    =Algo()
best_algo_dict['Pelt']      =Algo()
best_algo_dict['Window']    =Algo()
best_algo_dict['Dynp']      =Algo()
best_algo_dict['BottomUp']  =Algo()


for alg in algo_dict.keys():
    obj_list = algo_dict[alg]

    if type(obj_list) == int:
        print("obj_list == int ", obj_list)
        continue

    #print("\n>>> " + obj_list[0].algo_name + " <<<")
    for ii in range(len(obj_list)):
        algo_obj = obj_list[ii]
        preds = np.array(algo_obj.pred_list)[0]
        algo_obj.MAE = get_MAE(preds,  algo_obj.true_onset);
        algo_obj.rmse = get_rmse(preds, algo_obj.true_onset)
        algo_obj.bias= get_bias(preds, algo_obj.true_onset)
        algo_obj.fdr = get_fdr(preds,  algo_obj.true_onset, margin)

        if best_algo_dict[alg].MAE > algo_obj.MAE:
            best_algo_dict[alg]=algo_obj

        if verbose==True:
            print("model=%10s, min_size=%d, MAE=%6.2f, fdr=%6.2f" % \
                  (algo_obj.model, algo_obj.min_size, algo_obj.MAE, algo_obj.fdr) )
    print("number of algorithm objects=", len(obj_list) )


# In[]

# dictionary of best algorithm objects
#
for alg in best_algo_dict.keys():
    name=best_algo_dict[alg].algo_name
    model = best_algo_dict[alg].model
    min_size = best_algo_dict[alg].min_size
    jump = best_algo_dict[alg].jump
    MAE = best_algo_dict[alg].MAE
    rmse = best_algo_dict[alg].rmse
    bias = best_algo_dict[alg].bias
    fdr = best_algo_dict[alg].fdr
    print("\n %9s  model=%-8s  min_size=%3d, MAE=%6.2f, RMSE=%8.2f, bias=%6.2f, fdr=%5.2f " \
          % (name, model, min_size, MAE, rmse, bias, fdr) )


# In[]

fig, ((ax1, ax2), (ax3, ax4),  (ax5, ax6)) = plt.subplots(3, 2, sharex=True, sharey=True)
fig.suptitle('Dataset %d: optimised CPD Algorithm predictions' % dataset_num, y=1, fontsize=16)

fig.set_figwidth(10)
fig.set_figheight(6)

plt.subplots_adjust(hspace = 0.3)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1,  decimals=0))

binwidth=10
bins = np.linspace(0, 450, 46)

algo_obj= best_algo_dict['BinSeg']
mdl = algo_obj.model
preds =  algo_obj.pred_list[0]
preds_bs=np.array(preds)

ax1.hist(preds, color='blue', bins=bins,  weights=np.ones(len(preds)) / len(preds))
ax1.set_title(("BinSeg %s" % mdl), fontsize=14)
ax1.axvline(algo_obj.true_onset, c="crimson", label="real onset")

algo_obj= best_algo_dict['Pelt']
mdl = algo_obj.model
preds =  algo_obj.pred_list[0]
preds_pelt=np.array(preds)

ax2.hist(preds, color='orange', bins=bins,  weights=np.ones(len(preds)) / len(preds))
ax2.set_title(("PELT %s" % mdl), fontsize=14)
ax2.axvline(algo_obj.true_onset, c="crimson", label="real onset")

algo_obj = best_algo_dict['Window']
mdl = algo_obj.model
preds =  algo_obj.pred_list[0]
preds_win=np.array(preds)

ax3.hist(preds, color='gray', bins=bins,  weights=np.ones(len(preds)) / len(preds))
ax3.set_title(("Window %s" % mdl), fontsize=14)
ax3.set_xlabel("time (ms)")
ax3.axvline(algo_obj.true_onset, c="crimson", label="real onset")

algo_obj = best_algo_dict['Dynp']
mdl = algo_obj.model
preds =  algo_obj.pred_list[0]
preds_dyn=np.array(preds)

ax4.hist(preds, color='yellow', bins=bins,  weights=np.ones(len(preds)) / len(preds))
ax4.set_title( ("Dynamic Programming %s" % mdl), fontsize=14)
ax4.set_xlabel("time (ms)")
ax4.axvline(algo_obj.true_onset, c="crimson", label="real onset")

algo_obj = best_algo_dict['BottomUp']
mdl = algo_obj.model
preds =  algo_obj.pred_list[0]
preds_bup=np.array(preds)

ax5.hist(preds, color='gray', bins=bins,  weights=np.ones(len(preds)) / len(preds))
ax5.set_title(("Bottomup %s" % mdl), fontsize=14)
ax5.set_xlabel("time (ms)")
ax5.axvline(algo_obj.true_onset, c="crimson", label="real onset")

for ax in fig.get_axes():
    ax.label_outer()

# In[]

#preds_pelt=[algo_obj.true_onset]

data = [preds_bs, preds_pelt, preds_win, preds_dyn, preds_bup]

# Create labels for the data
labels = ['Binseg', 'Pelt', 'Window', 'Dynp', 'BottomUp']

# Plot the violin plots
plt.figure(figsize=(10, 8))
sns.violinplot(data=data)
plt.axhline(algo_obj.true_onset)

# Set the x-axis labels
plt.xticks(ticks=range(len(labels)), labels=labels, fontsize=18)
plt.yticks(fontsize=18)

# Add a title
plt.suptitle('Distribution of predictions for five CPD algorithms\n dataset %d' % dataset_num, fontsize=22)
#plt.title('dataset %d' % dataset_num, fontsize=20)
fn = "dataset_%d_violins.pdf" % (dataset_num)

plt.savefig(fn, format="pdf", bbox_inches="tight")


# In[]

verbose = False

if verbose:
    fig, [ax1, ax2] = plt.subplots(ncols=2)
    fig.set_figwidth(12)
    fig.set_figheight(6)

    algo_obj = best_algo_dict['BinSeg']
    mdl = algo_obj.model
    predictions =  algo_obj.pred_list[0]

    # 10 ms time steps, 10ms bin width
    binwidth=10
    bins = np.linspace(0, 450, 46)

    plt.suptitle( ("Optimised BinSeg, %s" % mdl), fontsize=16)

    ax1.hist(predictions,  edgecolor='black', bins=bins,  weights=np.ones(len(predictions)) / len(predictions))
    ax1.set_title(("Onset predictions"), fontsize=14)
    ax1.axvline(200, c="crimson", label="real onset")
    ticks=ax1.get_yticks()
    ax1.set_yticks(ticks)
    ticks=ticks*100
    ticks = [int(v)for v in ticks]
    ax1.set_yticklabels(ticks);
    ax1.set_xlabel("detection time (ms)")
    ax1.set_ylabel("percent")

    plt.subplots_adjust(wspace = 0.3)
    total_predictions = len(predictions) # total number of predictions (predictions)

    ax2.axhline(0.05, c="r", label="nominal 5% level")
    ax2.set_ylabel("underestimations (cumulative %)")
    ax2.set_xlabel("underestimation of onset (ms)")
    ax2.set_xlim([0, 201])
    ax2.set_xticks(range(10,201,20))

    # this works better than sns histplot, etc, for reverse cumsum histogram
    bin_width=10
    bin_list = list(range(0, 201, bin_width))
    too_early = []

    for data_point in predictions:
        if data_point < algo_obj.true_onset:
            data_point = abs(data_point - algo_obj.true_onset)
            too_early.append(data_point)

    plt.hist(too_early, edgecolor='black', align="mid", cumulative=-1, bins=bin_list, weights=np.ones(len(too_early)) / total_predictions)
    ax2.set_title(("Too Early"), fontsize=14)
    ticks=ax2.get_yticks()
    ax2.set_yticks(ticks)
    ticks=ticks*100
    ticks = [int(v)for v in ticks]
    ax2.set_yticklabels(ticks);



# In[]

headers = [ 'cost function',  'MAE', 'RMSE', 'bias', 'pct too early']
data = dict()
for alg in best_algo_dict.keys():
    name=best_algo_dict[alg].algo_name
    model = best_algo_dict[alg].model
    MAE  = round(best_algo_dict[alg].MAE, 2)
    rmse  = round(best_algo_dict[alg].rmse, 2)
    bias = round(best_algo_dict[alg].bias, 2)
    pct_too_early = round(best_algo_dict[alg].pct_too_early, 2)
    data[name]   = [model, MAE, rmse, bias, pct_too_early ]

num_cols=len(headers)-1
textabular = f"l|c{'r'*num_cols}"
texheader = " & " + " & ".join(headers) + "\\\\"
texdata = "\\hline\n"
for label in data:
    texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"
    if label == "BottomUp":
        texdata += "\\hline\n"

title="\\caption{Synthetic dataset %d: CPD algorithms optimised for mean absolute error (MAE)}\n" % dataset_num

print("\\begin{table}[htbp]")
print("\\centering")
print(title)
print("\label{mae-table}")
print("\\begin{tabular}{"+textabular+"}")
print(texheader)
print(texdata, end="")
print("\\end{tabular}")
print("\\end{table}")


