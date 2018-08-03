import argparse
import glob
import math
import ntpath
import os
import shutil
import urllib
import urllib2

from datetime import datetime

import numpy as np
import pandas as pd

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

# find sampling rate for data in header file, useless always 256
def findSR(filename,ch):
    with open(filename, 'r') as f:
        for line in f:
            if ch in line:
                data = line.split(': ',1)
    return int(data[1][:3])

# load hypnogram data
def loadHypno(filename):
    with open(filename, 'r') as f:
        data = np.fromfile(f, dtype=np.int8)
    data[np.where(data==0)]=-4
    data[np.where(data==1)]=-0
    return (data)*-1

# load EEG data
def loadData(filename):
    with open(filename+".float32", 'r') as f:
        data = np.fromfile(f, dtype=np.float32)
    return (data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/sune/Data/Glostrup/data/",
                        help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--output_dir", type=str, default="/media/sune/Data/Glostrup/E1-M2-250-data/",
                        help="Directory where to save outputs.")
    #C3-M2 is standard chan
    #parser.add_argument("--select_ch", type=str, default="F3-M2",
    parser.add_argument("--select_ch", type=str, default="EOG",
                        help="File path to the trained model used to estimate walking speeds.")
    parser.add_argument("--n_subjs", type=int, default=100,help="Number of subjects to take from data")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir+"/1/")
        os.makedirs(args.output_dir+"/2/")
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir+"/1/")
        os.makedirs(args.output_dir+"/2/")

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation EDF files
    data_folders = os.listdir(args.data_dir)
    sampling_rate = 256
    epoch_size = sampling_rate*EPOCH_SEC_SIZE
    # Currently limited to 20 for testing purposes, change the two lines below to do all
    #for i in range(len(data_folders)):
    long_subs=0
    y1_tot = 0
    y2_tot = 0
    for i in range(args.n_subjs):
        print "subject " + str(i) + " of " + str(args.n_subjs-1)
        raw_ch = loadData(args.data_dir+data_folders[i]+"/AASM1/"+select_ch)
        raw_ann = loadHypno(args.data_dir+data_folders[i]+"/AASM1/HYPNOGRAM.int8")
        raw_ann2 = loadHypno(args.data_dir+data_folders[i]+"/AASM2/HYPNOGRAM.int8")
        n_epochs = (raw_ch.shape[0]/epoch_size)
        if (n_epochs< raw_ann.shape[0]):
            raw_ann = raw_ann[:n_epochs]
        x = np.asarray(np.split(raw_ch,n_epochs)).astype(np.float32)
        y = raw_ann.astype(np.int32)
        y2 = raw_ann2.astype(np.int32)
        #Select on sleep periods
        w_edge_mins = 30
        nw_idx1 = np.where(y !=0 )[0]
        nw_idx2 = np.where(y2 !=0 )[0]
        if len(nw_idx1)>0:
            #for y1
            start_idx = nw_idx1[0] - (w_edge_mins * 2)
            end_idx = nw_idx1[-1] + (w_edge_mins * 2)
            if start_idx < 0: start_idx = 0
            if end_idx >= len(y): end_idx = len(y) - 1
            select_idx = np.arange(start_idx, end_idx+1)
            x = x[select_idx]
            y = y[select_idx]
            # for y2
            y2 = y2[select_idx]
            print("Data selection y1, y2: {}, {}".format(y2.shape, y.shape))

        filename = data_folders[i]+".npz"
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": None,
            "header_annotation": None,
        }
        np.savez(os.path.join(args.output_dir+"/1", filename), **save_dict)

        save_dict = {
            "x": x,
            "y": y2,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": None,
            "header_annotation": None,
        }
        np.savez(os.path.join(args.output_dir+"/2", filename), **save_dict)

if __name__ == "__main__":
    main()
