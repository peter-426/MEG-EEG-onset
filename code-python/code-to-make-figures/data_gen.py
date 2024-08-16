#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np
import math
#from scipy.stats import norm
from scipy.signal import butter, filtfilt, lfilter


PI = math.pi

# In[]
#
def make_data_1(n_samples):
    #print("hello 1")
    signal, bkps = rpt.pw_wavy(n_samples=n_samples, n_bkps=1, noise_std=0.5)
    s1 = np.array(signal[0:bkps[0]] )
    s2 = np.array(signal[ bkps[0]: ]) + 5
    signal = np.concatenate((s1,s2))
    return signal, bkps


def make_data_2(n_samples, true_onset, mu, sigma, mu2, sigma2, rng):
    #print("hello 2")
    n = n_samples - true_onset
    s1 = rng.normal(mu, sigma, (true_onset-1))
    s2 = rng.normal(mu2, sigma2, n)
    signal= np.array(list(s1) + list(s2))
    return signal


#
# large variation is erp shapes
# frequent rebound overshoot is apparent
def make_data_3(times, true_onset, rng, scale):
    #print("hello 3")
    erp = ( 1 * np.sin(30 * times) * np.exp(-((times - 0.15 + 0.05 * rng.normal(1)) ** 2) / 0.01) )
    baseline=np.zeros(true_onset-1)
    erp = erp + rng.normal(loc=0.0, scale=scale, size=len(erp))
    signal = np.concatenate( (baseline, erp), axis=0)
    return signal

def make_data_4(times, true_onset, rng, scale):
    #print("hello 4")
    erp = ( 1 * np.sin(30 * times) * np.exp(-((times - 0.15 + 0.05 * rng.normal(1)) ** 2) / 0.01) )
    baseline=np.zeros(true_onset-1)
    signal = np.concatenate( (baseline, erp), axis=0)
    signal = signal + rng.normal(loc=0.0, scale=scale, size=len(signal))
    return signal

##############################################################
# The filtfilt function in scipy.signal is a noncausal filter.
#
# Causal Filter: Only processes data in one direction (forward).
# Output depends only on current and past inputs.
#
# Noncausal Filter: Processes data in both directions (forward and backward).
# Output depends on both past and future inputs. filtfilt achieves this by
# applying the filter twice: once forward and once backward, resulting in
# zero phase distortion.
#
def bandpass_filter_noncausal(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter_causal(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#
# demo
#
# parameters
def demo():
    my_rand_seed=12345

    mu, sigma = 0, 1 # mean and standard deviation for simulated data generation
    mu2, sigma2 = mu + 1, sigma + 0.5

    true_onset=160
    n_samples=451

    erp_len = 451-(true_onset-1)
    time_points = np.arange(erp_len)/500

    ### 1
    dataset_num=1

    signal_1, bkps = make_data_1(n_samples)

    plt.figure(figsize=(8,6))
    plt.plot(signal_1)
    plt.title("Example from synthetic dataset %d" % dataset_num, fontsize=22)
    plt.ylabel("amplitude",    fontsize=18)
    plt.xlabel("time points",  fontsize=18)
    plt.xticks( fontsize=18)
    plt.yticks( fontsize=18)
    plt.axvline(bkps[0], c="crimson", label="true onset");
    plt.legend(fontsize=18)
    plt.tight_layout()

    fn = "dataset_%d_example.pdf" % (dataset_num)

    plt.savefig(fn, format="pdf", bbox_inches="tight")

    ### 2
    dataset_num=2
    rng=np.random.default_rng(my_rand_seed)
    signal_2 = make_data_2(n_samples, true_onset, mu, sigma, mu2, sigma2, rng)

    plt.figure(figsize=(8,6))
    plt.plot(signal_2)
    plt.title("Example from synthetic dataset %d" % dataset_num, fontsize=22)
    plt.ylabel("amplitude",    fontsize=18)
    plt.xlabel("time points",  fontsize=18)
    plt.xticks( fontsize=18)
    plt.yticks( fontsize=18)
    plt.axvline(true_onset, c="crimson", label="true onset");
    plt.legend(fontsize=18)
    plt.tight_layout()

    fn = "dataset_%d_example.pdf" % (dataset_num)

    plt.savefig(fn, format="pdf", bbox_inches="tight")

    ### 3 shape varies
    dataset_num=3
    rng = np.random.RandomState(my_rand_seed)
    scale = 0.05

    signal_3=make_data_3(time_points, true_onset, rng, scale)
    #signal_3=abs(signal_3)

    plt.figure(figsize=(8,6))
    plt.plot(signal_3);
    plt.title("Example from synthetic dataset %d:\nflat baseline, signal + gaussian" % dataset_num, fontsize=22)
    plt.ylabel("amplitude",  fontsize=18)
    plt.xlabel("time points", fontsize=18)
    plt.xticks( fontsize=18)
    plt.yticks( fontsize=18)
    plt.axvline(true_onset, c="crimson", label="true onset");
    plt.legend(fontsize=18)
    plt.tight_layout()

    fn = "dataset_%d_example.pdf" % (dataset_num)

    plt.savefig(fn, format="pdf", bbox_inches="tight")

    ### 4
    dataset_num=4

    # Sample parameters
    lowcut = 1.0  # Low frequency cut-off
    highcut = 30.0  # High frequency cut-off
    fs = 500.0  # Sampling frequency
    order = 5  # Filter order


    rng = np.random.RandomState(my_rand_seed)
    scale = 0.1

    signal=make_data_4(time_points, true_onset, rng, scale)
    signal_4 = bandpass_filter_causal(signal, lowcut, highcut, fs, order)
    #signal_4=abs(signal_4)

    plt.figure(figsize=(8,6))
    plt.plot(signal_4);
    plt.title("Example from synthetic dataset %d:\n (flat baseline, signal) + gaussian (causal filter)" % dataset_num, fontsize=22)
    plt.ylabel("amplitude",   fontsize=18)
    plt.xlabel("time points",  fontsize=18)
    plt.xticks( fontsize=18)
    plt.yticks( fontsize=18)
    plt.axvline(true_onset, c="crimson", label="true onset");
    plt.legend(fontsize=18)
    plt.tight_layout()

    fn = "dataset_%d_example.pdf" % (dataset_num)

    plt.savefig(fn, format="pdf", bbox_inches="tight")

demo()
