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
    parser.add_argument("--output_dir", type=str, default="/media/sune/Data/Glostrup/100-C3-M2-data-LO-100/",
    #parser.add_argument("--output_dir", type=str, default="/media/sune/Data/Glostrup/E1-M2-250-data/",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="C3-M2",
    #parser.add_argument("--select_ch", type=str, default="E1-M2",
                        help="File path to the trained model used to estimate walking speeds.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation EDF files
    data_folders = os.listdir(args.data_dir)
    sampling_rate = 256
    epoch_size = sampling_rate*EPOCH_SEC_SIZE
    # Currently limited to 20 for testing purposes, change the two lines below to do all
    #for i in range(len(data_folders)):
    for i in range(100):
        print "subject " + str(i) + " of 99"
        raw_ch = loadData(args.data_dir+data_folders[i]+"/"+select_ch)
        raw_ann = loadHypno(args.data_dir+data_folders[i]+"/HYPNOGRAM.int8")
        n_epochs = (raw_ch.shape[0]/epoch_size)
        if (n_epochs< raw_ann.shape[0]):
            raw_ann = raw_ann[:n_epochs]

        LO = np.fromfile(args.data_dir+data_folders[i]+"/LIGHTS-OFF.int8", dtype=np.int8)

        split = (np.argmax(LO)/30)-1
        x = np.asarray(np.split(raw_ch,n_epochs)).astype(np.float32)
        y = raw_ann.astype(np.int32)
        if len(y)==0:
            continue
        if len(x)==0:
            print "RIP"
            continue
        x = x[(len(x)-1700):]
        y = y[(len(y)-1700):]
        w_edge_mins = 15
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        print("Data after LO split: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        print("Data after selection: {}, {}".format(x.shape, y.shape))

        filename = data_folders[i]+".npz"
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": None,
            "header_annotation": None,
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)


if __name__ == "__main__":
    main()
