#!/usr/bin/env python

import os
import argparse
import numpy as np

from utils.data.fileio import _read_root
from utils.data.tools import  _get_variable_names
from utils.data.preprocess import _build_new_variables,_apply_selection
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

def roc_input(table,var,label_sig,label_bkg):
    scores_sig = np.zeros(table[var].shape[0])
    scores_bkg = np.zeros(table[var].shape[0])
    scores_sig = table[var][(table[label_sig] == 1)]
    scores_bkg = table[var][(table[label_bkg] == 1)]
    predict = np.concatenate((scores_sig,scores_bkg),axis=None)
    siglabels = np.ones(scores_sig.shape)
    bkglabels = np.zeros(scores_bkg.shape)
    truth = np.concatenate((siglabels,bkglabels),axis=None)
    return truth, predict

def plot_response(data, labels, name):
    plt.clf()
    bins=100
    for j in range(0,len(data)):
        plt.hist(data[j],bins,log=False,histtype='step',density=True,label=labels[j],fill=False,range=(-1.,1.))
    plt.legend(loc='best')
    plt.xlim(0,1)
    plt.xlabel('%s Response'%name)
    plt.ylabel('Number of events (normalized)')
    plt.title('NeuralNet applied to test samples')
    plt.savefig("%s_disc.pdf"%(name))

# get roc for  given table w consistent scores and label shapes
def get_roc(table, scores, label_sig, label_bkg):
    fprs = {}
    tprs = {}
    for score_name,score_label in scores.items():
        truth, predict =  roc_input(table,score_name,label_sig['label'],label_bkg['label'])
        fprs[score_label], tprs[score_label], threshold = roc_curve(truth, predict)
    return fprs, tprs

def plot_roc(label_sig, label_bkg, fprs, tprs):
    plt.clf()
    def get_round(x_effs,y_effs,to_get=[0.01,0.02,0.03]):
        effs = []
        for eff in to_get:
            for i,f in enumerate(x_effs):
                if round(f,2) == eff:
                    effs.append(y_effs[i])
                    print(round(f,2),y_effs[i])
                    break
        return effs     

    markers = ['v','^','o','s']
    ik = 0
    for k,it in fprs.items():
        plt.plot(fprs[k], tprs[k], lw=2.5, label=r"{}, AUC = {:.1f}%".format(k,auc(fprs[k],tprs[k])*100))
        #x_effs = [0.01,0.02,0.03]
        #y_effs = get_round(fprs[k],tprs[k])
        #print(x_effs,y_effs)
        #plt.scatter(x_effs,y_effs,marker=markers[ik],label=k)
        ik+=1

    plt.legend(loc='upper left')
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.ylabel(r'Tagging efficiency %s'%label_sig['legend']) 
    plt.xlabel(r'Mistagging rate %s'%label_bkg['legend'])
    plt.savefig("roc_%s.pdf"%label_sig['label'])
    plt.xscale('log')
    plt.savefig("roc_%s_xlog.pdf"%label_sig['label'])
    plt.xscale('linear')    

def main(args):

    label_bkg = {'qcd':{'legend': 'QCD',
                        'label':  'fj_isQCD'},
             }
    label_sig = {'4q':{'legend': 'H(WW)4q',
                       'label': 'label_4q',
                       #'label':  'fj_H_WW_4q',
                       'scores': 'H4q'
                   },
    }

    label_sig = label_sig[args.channel]
    label_bkg = label_bkg['qcd']
    
    funcs = {
        'score_H4q': 'score_label_4q/(score_label_4q+score_fj_isQCD)',
    }
    
    inputfile = args.input
    name  = args.name
    
    # make dict of branches to load
    lfeatures = ['fj_msoftdrop','fj_pt']

    # go to plot directory
    cwd=os.getcwd()
    odir = '%s/'%(args.tag)
    os.system('mkdir -p %s'%odir)
    os.chdir(odir)

    # now build tables
    scores = {'score_%s'%label_sig['scores']: '%s %s'%(name,label_sig['scores'])}
    
    loadbranches = set()
    for k,kk in scores.items():
        # load scores
        if k in funcs.keys(): loadbranches.update(_get_variable_names(funcs[k]))
        else: loadbranches.add(k)
        
    # load features
    loadbranches.add(label_bkg['label'])
    loadbranches.add(label_sig['label'])
    for k in lfeatures: loadbranches.add(k)
    loadbranches.add('fj_genH_mass')
    loadbranches.add('fj_genH_pt')

    mh_values = list(range(50,200,20))
    mh_values += [120,125]
    
    #pt_values = list(range(200,1200,100))
    pt_values = list(range(300,1200,100))
    pt_val_last = 2000

    newfprs = {}
    newtprs = {}
    data = []
    labels = []
    #for n,mh in enumerate(mh_values):
    for n,pti in enumerate(pt_values):
        table = _read_root(inputfile, loadbranches)
        #config_selection = '((fj_isQCD==1) | ((label_4q==1) & (fj_genH_mass==%i)))'%mh
        pt_low = pti
        if n < len(pt_values)-1:
            pt_high = pt_values[n+1]
        else:
            pt_high = pt_val_last
        #config_selection = '((fj_isQCD==1) | ((label_4q==1) & (fj_genH_pt>%i) & (fj_genH_pt<%i)))'%(pt_low,pt_high)
        config_selection = '((fj_isQCD==1) | ((label_4q==1) & (fj_pt>%i) & (fj_pt<%i)))'%(pt_low,pt_high)

        _apply_selection(table, config_selection)
        _build_new_variables(table, {k: v for k,v in funcs.items() if k in scores.keys()})

        fprs, tprs = get_roc(table, scores, label_sig, label_bkg)
        oldkeys = list(fprs.keys())
        for key in oldkeys:
            #newkey=key+'_mh%i'%mh
            newkey=key+'_pTh%i'%(pt_low)
            fprs[newkey] = fprs[key]
            tprs[newkey] = tprs[key]
            del fprs[key]
            del tprs[key]

        print(fprs)
        for score_name in scores.keys():
            var = table[score_name]
            data += [var[(table[label_sig['label']] == 1)]]
            #labels += [label_sig['legend']+' mH=%i'%mh]
            #labels += [label_sig['legend']+' pTH[%i,%i]'%(pt_low,pt_high)]
            labels += [label_sig['legend']+' JpT[%i,%i]'%(pt_low,pt_high)]

            if n==0:
                data+=[var[(table[label_bkg['label']] == 1)]]
                labels+=[label_bkg['legend']]

        if n==0: 
            newfprs = fprs
            newtprs = tprs
        else:
            for k in fprs:
                newfprs[k] = fprs[k]
                newtprs[k] = tprs[k]

        
    #plot_response(data, labels, name+args.channel+'_allmh')
    plot_response(data, labels, name+args.channel+'_allJpT')
    plot_roc(label_sig, label_bkg, newfprs, newtprs)
    os.chdir(cwd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file(s)')
    parser.add_argument('--name', help='name ROC(s)')
    parser.add_argument('--tag', help='folder tag')
    parser.add_argument('--idir', help='idir')
    parser.add_argument('--odir', help='odir')
    parser.add_argument('--channel', help='channel')
    parser.add_argument('--jet', default="AK8", help='jet type')
    parser.add_argument('--selection', default=None, type=str, help='selection')
    args = parser.parse_args()

    main(args)
