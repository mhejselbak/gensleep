import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
# skal laves fra
# w 0 , n1 1, n2 2, n3 3, rem 4
# til
#W 1, R 0, N1 -1, N2 -2, N3 -3

def reshape_hyp(y):
    y = y*-1
    y[np.where(y==0)]=1
    y[np.where(y==-4)]=0
    y=y.astype(np.float32)
    y_r=np.zeros(y.shape).astype(np.float32)
    y_r[y!=0]=np.nan
    #y[y==0]=np.nan
    return y,y_r

def plot_hypnos(args,data,rem_data,fname,d2,dr2,dras,drasr,true,true_rem):
    plt.figure(figsize=(20,10), dpi=100)
    plt.subplot(411)
    plt.title(args.mod_name1)
    plt.plot(np.arange(len(data)),data,color='black')
    plt.plot(np.arange(len(data)),rem_data,linewidth=1.8,color='black')
    hours = (len(data)/2)/60
    #plt.yticks(np.arange(5), ['W', 'N1', 'N2', 'N3', 'REM'] )
    plt.ylim(-3.2,1.2)
    plt.yticks([-3,-2,-1,-0,1], ['N3', 'N2', 'N1', 'REM','W'] )
    #plt.xticks([i *120 for i in range(hours+1)], np.arange(hours+1).astype(str) )
    plt.xticks([], [])

    plt.subplot(412)
    plt.title(args.mod_name2)
    plt.plot(np.arange(len(d2)),d2,color='black')
    plt.plot(np.arange(len(d2)),dr2,linewidth=1.8,color='black')
    hours = (len(true)/2)/60
    #plt.yticks(np.arange(5), ['W', 'N1', 'N2', 'N3', 'REM'] )
    plt.ylim(-3.2,1.2)
    plt.yticks([-3,-2,-1,-0,1], ['N3', 'N2', 'N1', 'REM','W'] )
    #plt.xticks([i *120 for i in range(hours+1)], np.arange(hours+1).astype(str) )
    plt.xticks([], [])

    plt.subplot(413)
    plt.title("A. Sors et al.")
    plt.plot(np.arange(len(dras)),dras,color='black')
    plt.plot(np.arange(len(dras)),drasr,linewidth=1.8,color='black')
    hours = (len(true)/2)/60
    #plt.yticks(np.arange(5), ['W', 'N1', 'N2', 'N3', 'REM'] )
    plt.ylim(-3.2,1.2)
    plt.yticks([-3,-2,-1,-0,1], ['N3', 'N2', 'N1', 'REM','W'] )
    #plt.xticks([i *120 for i in range(hours+1)], np.arange(hours+1).astype(str) )
    plt.xticks([], [])

    plt.subplot(414)
    plt.title("Sleep expert")
    plt.plot(np.arange(len(true)),true,color='black')
    plt.plot(np.arange(len(true)),true_rem,linewidth=1.8,color='black')
    hours = (len(true)/2)/60
    #plt.yticks(np.arange(5), ['W', 'N1', 'N2', 'N3', 'REM'] )
    plt.ylim(-3.2,1.2)
    plt.yticks([-3,-2,-1,-0,1], ['N3', 'N2', 'N1', 'REM','W'] )
    plt.xticks([i *120 for i in range(hours+1)], np.arange(hours+1).astype(str) )
    plt.xlabel("Hours")
    plt.savefig(fname+".png",dpi=100)
    plt.close()
   # plt.show()

def doit(args):
    files1 = glob.glob(os.path.join(args.data_dir1,'*fold_4*.npz'))
    files2 = glob.glob(os.path.join(args.data_dir2,'*fold_4*.npz'))
    files1.sort()
    files2.sort()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    for i in range(28):
        print "ploting subject " + str(i) + " out of " + str(len(files1)-1)
        y_t = np.load(files1[i])['y_true']
        y_t2 = np.load(files2[i])['y_true']

        y_p = np.load(files1[i])['y_pred']
        y_p,ypr = reshape_hyp(y_p)#[0])#[:])

        y_2 = np.load(files2[i])['y_pred']
        y_2,y2r = reshape_hyp(y_2)#[0])#[:])

        y_d = np.load('/home/mhejsel/drasros-shhs/glo-multi-hyp/pred_test.npy')
        #y_d = np.load('/home/mhejsel/drasros-shhs/edf-test-hyp/pred_test.npy')
        if len(y_d)<len(y_p):
            y_d = np.hstack((y_d,np.zeros(len(y_p)-len(y_d))))
        y_d,ydr = reshape_hyp(y_d)#[0])#[:])

        y_t,ytr = reshape_hyp(y_t)#[0])#[:])
        plot_hypnos(args,y_p,ypr,args.out_dir+"/subj_"+str(i), y_2, y2r, y_d, ydr, y_t, ytr)

def main():
     parser = argparse.ArgumentParser(description='Create hypnogram of the output from the neural networks. Comparison of two models')
     parser.add_argument('-d1','--data_dir1',help='Path to the model',type=str)
     parser.add_argument('-d2','--data_dir2',help='Path to the model',type=str)
     parser.add_argument('-m1','--mod_name1',help='Model1 name for plotting',type=str)
     parser.add_argument('-m2','--mod_name2',help='Model2 name for plotting',type=str)
     parser.add_argument('-o','--out_dir',help='Path for saving the histograms',type=str)
     args = parser.parse_args()
     doit(args)

if __name__ == "__main__":
    main()
