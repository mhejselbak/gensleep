import argparse
import glob
import re
import math
import ntpath
import os
import shutil
import urllib
import urllib2

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

#import dhedfreader
# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}

EPOCH_SEC_SIZE = 30

def print_performance(cm,kappa):
    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    print "Sample: {}".format(np.sum(cm))
    print "W: {}".format(tpfn[W])
    print "N1: {}".format(tpfn[N1])
    print "N2: {}".format(tpfn[N2])
    print "N3: {}".format(tpfn[N3])
    print "REM: {}".format(tpfn[REM])
    print "Confusion matrix:"
    print cm
    print "Precision: {}".format(precision)
    print "Recall: {}".format(recall)
    print "F1: {}".format(f1)
    print "Kappa coefficient: {}".format(kappa)
    print "Overall accuracy: {}".format(acc)
    print "Macro-F1 accuracy: {}".format(mf1)

# load hypnogram data
def loadHypno(filename):
    with open(filename, 'r') as f:
        data = np.fromfile(f, dtype=np.int8)
    data[np.where(data==0)]=-4
    data[np.where(data==1)]=-0
    return (data)*-1

def perf_overall(data_dir):
    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir+"/250-ens/")
    outputfiles = []
    outputfiles2 = []
    for idx, f in enumerate(allfiles):
        #if re.match("^output_.+\d+\.npz", f):
        if re.match("^output_fold_0_.+\d+\.npz", f):
            outputfiles.append(os.path.join(data_dir+"/250-ens/", f))
            outputfiles2.append(os.path.join(data_dir+"2/250-ens/", f))

    outputfiles.sort()
    outputfiles2.sort()

    y_true = []
    y_pred = []
    for i in range(len(outputfiles)):
        f_y_true = np.load(outputfiles[i])['y_true']
        f_y_pred = np.load(outputfiles2[i])['y_true']
        #if len(f["y_true"].shape) == 1:
        #    if len(f["y_true"]) < 10:
        #        f_y_true = np.hstack(f["y_true"])
        #        f_y_pred = np.hstack(f["y_pred"])
        #    else:
        #        f_y_true = f["y_true"]
        #        f_y_pred = f["y_pred"]
        #else:
        #    f_y_true = f["y_true"].flatten()
        #    f_y_pred = f["y_pred"].flatten()

        y_true.extend(f_y_true)
        y_pred.extend(f_y_pred)

        print "File: {}".format(outputfiles[i])
        cm = confusion_matrix(f_y_true, f_y_pred, labels=[0, 1, 2, 3, 4])
        kappa_ = cohen_kappa_score(f_y_true, f_y_pred)
        print_performance(cm,kappa_)
        print " "

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    total = np.sum(cm, axis=1)

    print_performance(cm,kappa)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/sune/Data/Glostrup/data/",
                        help="File path to the CSV or NPY file that contains walking data.")
    args = parser.parse_args()
    perf_overall(args.data_dir)
    # Read annotation EDF files
    #data_folders = os.listdir(args.data_dir)
    #sampling_rate = 256
    #epoch_size = sampling_rate*EPOCH_SEC_SIZE
    ## Currently limited to 20 for testing purposes, change the two lines below to do all
    ##for i in range(len(data_folders)):
    #y1_all=[]
    #y2_all=[]
    #for i in range(28):
    #    print "subject " + str(i) + " of " + str(28)
    #    raw_ann = loadHypno(args.data_dir+data_folders[i]+"/AASM1/HYPNOGRAM.int8")
    #    raw_ann2 = loadHypno(args.data_dir+data_folders[i]+"/AASM2/HYPNOGRAM.int8")
    #    y = raw_ann.astype(np.int32)
    #    y2 = raw_ann2.astype(np.int32)
    #    if len(y)==0:
    #        continue
    #    w_edge_mins = 30
    #    nw_idx = np.where(y != stage_dict["W"])[0]
    #    #print nw_idx[0]
    #    if len(nw_idx)>0:
    #        start_idx = nw_idx[0] - (w_edge_mins * 2)
    #        end_idx = nw_idx[-1] + (w_edge_mins * 2)
    #        if start_idx < 0: start_idx = 0
    #        if end_idx >= len(y): end_idx = len(y) - 1
    #        select_idx = np.arange(start_idx, end_idx+1)
    #        #print("Data before selection: {}, {}".format(x.shape, y.shape))
    #        y = y[select_idx]
    #        y2 = y2[select_idx]
    #        #print("Data after selection: {}, {}".format(x.shape, y.shape))
    #    y1_all.extend(y)
    #    y2_all.extend(y2)

    #cm = confusion_matrix(y1_all, y2_all, labels=[0, 1, 2, 3, 4])
    #kappa_ = cohen_kappa_score(y1_all, y2_all)
    #print_performance(cm,kappa_)

if __name__ == "__main__":
    main()
