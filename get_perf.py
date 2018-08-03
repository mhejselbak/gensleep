import argparse
import os
import re

import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from data_loaders.sleep_stage import W, N1, N2, N3, REM

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
    return f1,acc

def perf_overall(data_dir):
    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir)
    outputfiles = []
    for idx, f in enumerate(allfiles):
        if re.match("^output_.+\d+\.npz", f):
        #if re.match("^output_fold_0_.+\d+\.npz", f):
            outputfiles.append(os.path.join(data_dir, f))

    outputfiles.sort()

    y_true = []
    y_pred = []
    i = 0
    for fpath in outputfiles:
        print "index in files ", str(i)
        i+=1
        with np.load(fpath) as f:
            if len(f["y_true"].shape) == 1:
                if len(f["y_true"]) < 10:
                    f_y_true = np.hstack(f["y_true"])
                    f_y_pred = np.hstack(f["y_pred"])
                else:
                    f_y_true = f["y_true"]
                    f_y_pred = f["y_pred"]
            else:
                f_y_true = f["y_true"].flatten()
                f_y_pred = f["y_pred"].flatten()

            y_true.extend(f_y_true)
            y_pred.extend(f_y_pred)

            print "File: {}".format(fpath)
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

    print "DeepSleepNet (current)"
    print_performance(cm,kappa)

def fold_avg_perf(data_dir):
    # Remove non-output files, and perform ascending sort
    all_f1 = np.zeros((5,5))
    all_mf1 = np.zeros(5)
    all_kap = np.zeros(5)
    all_acc = np.zeros(5)
    for i in range(5):
        allfiles = os.listdir(data_dir)
        outputfiles = []
        for idx, f in enumerate(allfiles):
            #if re.match("^output_.+\d+\.npz", f):
            if re.match("^output_fold_"+str(i)+"_.+\d+\.npz", f):
                outputfiles.append(os.path.join(data_dir, f))
        outputfiles.sort()

        y_true = []
        y_pred = []
        for fpath in outputfiles:
            with np.load(fpath) as f:
                    f_y_true = f["y_true"]
                    f_y_pred = f["y_pred"]

            y_true.extend(f_y_true)
            y_pred.extend(f_y_pred)

                #print "File: {}".format(fpath)
                #cm = confusion_matrix(f_y_true, f_y_pred, labels=[0, 1, 2, 3, 4])
                #kappa_ = cohen_kappa_score(f_y_true, f_y_pred)
                #print_performance(cm,kappa_)
        #print " "
        cm = confusion_matrix(y_true, y_pred)
        #acc = np.mean(y_true == y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro")
        kappa = cohen_kappa_score(y_true, y_pred)
        #all_acc[i] = acc
        all_kap[i] = kappa
        all_mf1[i] = mf1
        total = np.sum(cm, axis=1)
        print "Fold : ",i
        #print "DeepSleepNet (current)"
        all_f1[i],all_acc[i] = print_performance(cm,kappa)
        print " "

    print "Average performance"
    print "acc mean : ",np.mean(all_acc)*100
    print "acc std : ",np.std(all_acc)*100
    print "mf1 mean : ", np.mean(all_mf1,axis=0)*100
    print "mf1 std : ", np.std(all_mf1,axis=0)*100
    print "kappa mean : ", np.mean(all_kap)
    print "kappa std : ", np.std(all_kap)
    print "f1 mean : ", np.mean(all_f1,axis=0)*100
    print "f1 std : ", np.std(all_f1,axis=0)*100

    best=np.argmax(all_acc)
    print "Best performance is ", best
    print "acc best : ",all_acc[best]*100
    print "mf1 best : ", all_mf1[best]*100
    print "kappa best : ", all_kap[best]
    print "f1 best : ", all_f1[best]*100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/akara/Workspace/deepsleep_output/results/outputs",
                        help="Directory where to load prediction outputs")
    args = parser.parse_args()

    perf_overall(data_dir=args.data_dir)
    #fold_avg_perf(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
