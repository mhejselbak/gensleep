import argparse
import math
import os
import shutil
from scipy import signal
from datetime import datetime

import numpy as np
import pandas as pd

def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        chan = f["ch_label"]
    return data, labels, chan

def resample_npz(npz_files,new_hz,output_dir,data_dir):
    """Load data and labels from list of npz files."""
    for npz_f in npz_files:
        #print "Loading {} ...".format(npz_f)
        new_f = os.path.join(data_dir,npz_f)
        tmp_data, y, fs, = load_npz_file(new_f)
        stack_data = tmp_data[0]
        ## resampling
        stack_data = tmp_data.reshape(tmp_data.shape[0]*tmp_data.shape[1])
        samp_data=signal.resample(stack_data,tmp_data.shape[0]*(30*new_hz))
        #-# format back
        samp_data = samp_data.reshape(samp_data.shape[0]/(30*new_hz),30*new_hz)
        save_dict = {
            "x": samp_data,
            "y": y,
            "fs": new_hz,
            "ch_label": fs,
            "header_raw": None,
            "header_annotation": None,
        }
        np.savez(os.path.join(output_dir, npz_f), **save_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/sune/Data/Glostrup/100-C3-M2-data-LO-100/",
                        help="File path to the NPZ file that contains walking data.")
    parser.add_argument("--output_dir", type=str, default="/media/sune/Data/Glostrup/100-downsampled-glo",
                        help="Directory where to save outputs as NPZ.")
    parser.add_argument("--hz",type=int,default=100,
                        help="Target resampling Hz")
    parser.add_argument("--subjects",type=int,default=50,
                        help="Target resampling Hz")
    args = parser.parse_args()
    #if not os.path.exists(args.output_dir):
    #    os.makedirs(args.output_dir)
    #else:
    #    shutil.rmtree(args.output_dir)
    #    os.makedirs(args.output_dir)

    filenames = os.listdir(args.data_dir)

    filenames = filenames[:args.subjects]
    for i in filenames:
        shutil.copy(args.data_dir+"/"+i,args.output_dir+"/"+i)


if __name__ == "__main__":
    main()
