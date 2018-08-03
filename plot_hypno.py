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

def plot_val(fname,true,true_rem):
    plt.figure(figsize=(20,5), dpi=100)
    plt.title('Hypnogram annotated by sleep expert')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel("Hours")
    plt.plot(np.arange(len(true)),true,color='black')
    plt.plot(np.arange(len(true)),true_rem,linewidth=1.8,color='black')
    hours = (len(true)/2)/60
    #plt.yticks(np.arange(5), ['W', 'N1', 'N2', 'N3', 'REM'] )
    plt.ylim(-3.2,1.2)
    plt.yticks([-3,-2,-1,-0,1], ['N3', 'N2', 'N1', 'REM','W'] )
    plt.xlabel("Hours")
    plt.xticks([i *120 for i in range(hours+1)], np.arange(hours+1).astype(str) )
    plt.savefig(fname+".png",dpi=100)
    plt.close()

def plot_hypnos(data,rem_data,fname,true,true_rem):
    plt.figure(figsize=(20,5), dpi=100)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.subplot(211)
    plt.title('Predicted labels')
    plt.plot(np.arange(len(data)),data,color='black')
    plt.plot(np.arange(len(data)),rem_data,linewidth=1.8,color='black')
    hours = (len(data)/2)/60
    #plt.yticks(np.arange(5), ['W', 'N1', 'N2', 'N3', 'REM'] )
    plt.ylim(-3.2,1.2)
    plt.yticks([-3,-2,-1,-0,1], ['N3', 'N2', 'N1', 'REM','W'] )
    #plt.xticks([i *120 for i in range(hours+1)], np.arange(hours+1).astype(str) )
    plt.xticks([], [])

    plt.subplot(212)
    plt.title('Sleep expert')
    plt.plot(np.arange(len(true)),true,color='black')
    plt.plot(np.arange(len(true)),true_rem,linewidth=1.8,color='black')
    hours = (len(true)/2)/60
    #plt.yticks(np.arange(5), ['W', 'N1', 'N2', 'N3', 'REM'] )
    plt.ylim(-3.2,1.2)
    plt.yticks([-3,-2,-1,-0,1], ['N3', 'N2', 'N1', 'REM','W'] )
    plt.xlabel("Hours")
    plt.xticks([i *120 for i in range(hours+1)], np.arange(hours+1).astype(str) )
    plt.savefig(fname+".png",dpi=100)
    plt.close()
   # plt.show()

def doit(args):
    files = glob.glob(os.path.join(args.data_dir,'*fold_4*.npz'))
    #files = glob.glob(os.path.join(args.data_dir,'*.npz'))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if args.comp:
        for i in range(len(files)):
            print "ploting subject " + str(i) + " out of " + str(len(files)-1)
            y_p = np.load(files[i])['y_pred']
            y_p,ypr = reshape_hyp(y_p)
            #y_p,ypr = reshape_hyp(y_p[0][:])
            y_t = np.load(files[i])['y_true']
            y_t,ytr = reshape_hyp(y_t)
            #y_t,ytr = reshape_hyp(y_t[0][:])
            plot_hypnos(y_p,ypr,args.out_dir+"/subj_"+str(i),y_t,ytr)
    if args.valid:
        for i in range(len(files)):
            print "ploting subject " + str(i) + " out of " + str(len(files)-1)
            y_t = np.load(files[i])['y_true']
            y_t,ytr = reshape_hyp(y_t)
            #y_t,ytr = reshape_hyp(y_t[0][:])
            plot_val(args.out_dir+"/subj_"+str(i),y_t,ytr)
    if args.valid:
        for i in range(len(files)):
            print "ploting subject " + str(i) + " out of " + str(len(files)-1)
            y_p = np.load(files[i])['y_pred']
            y_p,ypr = reshape_hyp(y_p)
            plot_val(args.out_dir+"/subj_"+str(i),y_p,ypr)

def main():
     parser = argparse.ArgumentParser(description='Create hypnogram of the output from theneural networks. Default will also plot the true hypnogram')
     parser.add_argument('-d','--data_dir',help='Path to the model',type=str)
     parser.add_argument('-o','--out_dir',help='Path for saving the histograms',type=str)
     parser.add_argument('-v','--valid',help='If only valid hypnogram',type=bool,default=False)
     parser.add_argument('-p','--pred',help='If only predicted hypnogram',type=bool,default=False)
     parser.add_argument('-c','--comp',help='If comparison of valid/predicted hypnogram',type=bool,default=True)

     args = parser.parse_args()

     if not os.path.exists(args.out_dir):
         os.makedirs(args.out_dir)
     doit(args)

if __name__ == "__main__":
    main()
