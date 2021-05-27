#!/usr/bin/env python

import os
import argparse
import numpy as np

from utils.data.fileio import _read_root
from utils.data.tools import  _get_variable_names
from utils.data.preprocess import _build_new_variables
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

def plot_loss(indir,outdir,name):
    loss_vals_training = np.load('%s/loss_vals_training.npy'%indir)
    loss_vals_validation = np.load('%s/loss_vals_validation.npy'%indir)
    loss_std_training = np.load('%s/loss_std_training.npy'%indir)
    loss_std_validation = np.load('%s/loss_std_validation.npy'%indir)
    epochs = np.array(range(len(loss_vals_training)))
    f, ax = plt.subplots(figsize=(10, 10))
    ax.plot(epochs, loss_vals_training, label='Training')
    ax.plot(epochs, loss_vals_validation, label='Validation', color = 'green')
    leg = ax.legend(loc='upper right', title=name, borderpad=1, frameon=False, fontsize=16)
    leg._legend_box.align = "right" 
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch') 
    ax.set_xlim(0,np.max(epochs))
    #ax.set_xlim(5,np.max(epochs))
    f.savefig('%s/Loss_%s.png'%(outdir,indir.replace('/','')))
    f.savefig('%s/Loss_%s.pdf'%(outdir,indir.replace('/','')))
    plt.clf()

def plot_accuracy(indir,outdir,name):
    acc_vals_validation = np.load('%s/acc_vals_validation.npy'%indir)
    epochs = np.array(range(len(acc_vals_validation)))
    f, ax = plt.subplots(figsize=(10, 10))
    ax.plot(epochs, acc_vals_validation, label='Validation', color = 'green')
    leg = ax.legend(loc='upper right', title=name, borderpad=1, frameon=False, fontsize=16)
    leg._legend_box.align = "right"
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_xlim(0,np.max(epochs))
    ax.set_ylim(0.8,0.95)
    f.savefig('%s/Acc_%s.png'%(outdir,indir.replace('/','')))
    plt.clf()

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

def plot_features_qcd(table, scores, label_sig, label_bkg, name, features=['fj_sdmass','fj_pt']):
    labels = {'fj_msoftdrop': r'$m_{SD}$',
              'fj_pt': r'$p_{T}$',
          }
    feature_range = {'fj_msoftdrop': [[30,260],[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260]],
                     #'fj_pt': [[300,2500],[300, 380, 481, 608, 770, 974, 1233, 1561, 1975, 2500]],
                     'fj_pt': [[200,2500],[200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2500]],
                 }

    def computePercentiles(data,percentiles):
        mincut = 0.
        tmp = np.quantile(data,np.array(percentiles))
        tmpl = [mincut]
        for x in tmp: tmpl.append(x)
        perc = [0.]
        for x in percentiles: perc.append(x)
        return perc,tmpl

    for score_name,score_label in scores.items():
        bkg = (table[label_bkg['label']] == 1)
        var = table[score_name][bkg]
        percentiles = [0.1,0.2,0.8,0.9,0.95,0.97]
        per,cuts = computePercentiles(table[score_name][bkg],percentiles)
        for k in features:
            fig, ax = plt.subplots(figsize=(10,10))
            #bins = 40
            bins = feature_range[k][1]
            for i,cut in enumerate(cuts):
                c = (1-per[i])*100
                lab = '%i%% mistag-rate'%c
                ax.hist(table[k][bkg][var>cut], bins=bins, lw=2, density=True, range=feature_range[k][0],
                        histtype='step',label=lab)
            ax.legend(loc='best')
            ax.set_xlabel(labels[k]+' (GeV)')
            ax.set_ylabel('Number of events (normalized)')
            ax.set_title('%s dependence'%labels[k])
            plt.savefig("%s_%s.pdf"%(k,score_label))
            
def plot_response(table, scores, label_sig, label_bkg, name):
    for score_name,score_label in scores.items():
        plt.clf()
        bins=100
        var = table[score_name]
        data = [var[(table[label_sig['label']] == 1)], 
                var[(table[label_bkg['label']] == 1)]]
        labels = [label_sig['legend'],
                  label_bkg['legend']]
        for j in range(0,len(data)):
            plt.hist(data[j],bins,log=False,histtype='step',density=True,label=labels[j],fill=False,range=(-1.,1.))
        plt.legend(loc='best')
        plt.xlim(0,1)
        plt.xlabel('%s Response'%score_label)
        plt.ylabel('Number of events (normalized)')
        plt.title('NeuralNet applied to test samples')
        plt.savefig("%s_%s_disc.pdf"%(score_label,label_sig['label']))

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
    for k,it in fprs.items():
        plt.plot(fprs[k], tprs[k], lw=2.5, label=r"{}, AUC = {:.1f}%".format(k,auc(fprs[k],tprs[k])*100))
    plt.legend(loc='upper left')
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
                       'label':  'fj_H_WW_4q',
                       'scores': 'H4q'
                   },
                 'elenuqq':{'legend': 'H(WW) ele',
                            'label':  'fj_H_WW_elenuqq',
                            'scores': 'Helenuqq',
                      },
                 'munuqq':{'legend': 'H(WW) mu',
                           'label':  'fj_H_WW_munuqq',
                            'scores': 'Hmunuqq'
                       },
             }

    label_sig = label_sig[args.channel]
    label_bkg = label_bkg['qcd']
    
    funcs = {
        'PN_H4qvsQCD': 'fj_PN_H4qvsQCD',
        'score_H4q': 'score_fj_H_WW_4q/(1-score_fj_H_WW_elenuqq-score_fj_H_WW_munuqq)',
        'score_Helenuqq': 'score_fj_H_WW_elenuqq/(1-score_fj_H_WW_4q-score_fj_H_WW_munuqq)',
        'score_Hmunuqq': 'score_fj_H_WW_munuqq/(1-score_fj_H_WW_4q-score_fj_H_WW_elenuqq)',
    }
    
    if args.jet == "AK8":
        funcs['deepAK8MD_H4q'] = 'fj_deepTagMD_H4qvsQCD'
        funcs['deepAK8_H'] = 'fj_deepTag_HvsQCD'

    # inputfiles and names should have same shape
    inputfiles = args.input.split(',')
    names = args.name.split(',')
    
    # make dict of branches to load
    lfeatures = ['fj_msoftdrop','fj_pt']
    sameshape = True
    sh = []
    # check if the input files have the same shape
    for n,name in enumerate(names):
        table = _read_root(inputfiles[n], lfeatures)
        for k in table.keys():
            sh.append(table[k].shape)
            if n>0 and table[k].shape != sh[0]:
                sameshape = False

    # go to plot directory
    cwd=os.getcwd()
    odir = 'plots/%s/'%(args.tag)
    os.system('mkdir -p %s'%odir)
    os.chdir(odir)

    # now build tables
    for n,name in enumerate(names):
        scores = {'score_%s'%label_sig['scores']: '%s %s'%(name,label_sig['scores'])}
        if args.channel=='4q':
            scores['PN_H4qvsQCD'] = 'PN H4qvsQCD'
            if args.jet == "AK8":
                scores['deepAK8MD_H4q'] = 'DeepAK8 MD H4q' 
                scores['deepAK8_H'] = 'DeepAK8 H'  
        else:
            if args.jet == "AK8":
                scores['fj_lsf3'] = 'LSF'

        loadbranches = set()
        for k,kk in scores.items():
            # load scores
            if k in funcs.keys(): loadbranches.update(_get_variable_names(funcs[k]))
            else: loadbranches.add(k)

            # load features
            loadbranches.add(label_bkg['label'])
            loadbranches.add(label_sig['label'])
            for k in lfeatures: loadbranches.add(k)

        table = _read_root(inputfiles[n], loadbranches)
        if(n==0 or (not sameshape)):
            _build_new_variables(table, {k: v for k,v in funcs.items() if k in scores.keys()})
            newtable = table
            newscores = scores
        else:
            for k in table: 
                newtable[k] = table[k]
            for k in scores:
                newscores[k] = scores[k]

        if not sameshape:
            fprs, tprs = get_roc(table, scores, label_sig, label_bkg)
            if n==0: 
                newfprs = fprs
                newtprs = tprs
            else:
                for k in fprs:
                    newfprs[k] = fprs[k]
                    newtprs[k] = tprs[k]
            plot_response(table, scores, label_sig, label_bkg, name+args.channel)
            plot_features_qcd(table, scores, label_sig, label_bkg, name+args.channel, lfeatures)
            plot_features_signal(able, scores, label_sig, label_bkg, name+args.channel, lfeatures_signal)

    if sameshape:
        newfprs, newtprs = get_roc(newtable, newscores, label_sig, label_bkg)
        plot_response(newtable,  newscores, label_sig, label_bkg, args.channel)
        plot_features_qcd(newtable, newscores, label_sig, label_bkg, args.channel, lfeatures)

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
    parser.add_argument('--loss', action='store_true', default=False, help='plot loss and acc')
    parser.add_argument('--roc', action='store_true', default=False, help='plot roc and nn output')
    args = parser.parse_args()

    if args.roc:
        main(args)
    #if args.loss:
    #    plot_loss(args.idir,args.odir,args.name)
    #    plot_accuracy(args.idir,args.odir,args.name)
    
