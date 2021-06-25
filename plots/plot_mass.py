import uproot
import awkward as ak
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep

from coffea import hist
hist_mass_hww = hist.Hist("Gen H mass", 
                          hist.Cat("process", "Process"), 
                          hist.Bin("msd", r"fj msoftdrop [GeV]", 50, 20, 380),
                          #hist.Bin("pt", r"fj pT [GeV]", 50, 300, 1000),
                          hist.Bin("pt_over_mass", "gen H pT/mass", 30, 0, 30),
                          hist.Bin("mass", r"gen H mass [GeV]", 15, 50, 200),
                      )

signal_names = "/data/shared/cmantill/training/ak8_v03hww_genHmpt_Jun8/train/GravitonToHHToWWWW/*"
#signal_names = "/data/shared/cmantill/training/ak8_v03hww_genHmpt_Jun8/train/BulkGravitonToHHTo4Q_MX*/*"
#signal_names = "/data/shared/cmantill/training/ak8_v03hww_genHmpt_Jun8/test/Glu*/*"

branches = ["fj_genH_mass","fj_H_WW_4q","fj_H_WW_elenuqq","fj_H_WW_munuqq","fj_H_bb","fj_H_cc","fj_H_qq",'fj_msoftdrop','fj_pt','fj_genH_pt','fj_mass']

events_dict = uproot.iterate(signal_names+":Events", branches)

signals = ["fj_H_WW_4q","fj_H_WW_elenuqq","fj_H_WW_munuqq"]
#signals = ["fj_H_bb","fj_H_cc","fj_H_qq"]

for events_chunk in events_dict:
    for key in signals:
        h_mass = events_chunk.fj_genH_mass[events_chunk[key]==1]
        h_msd = events_chunk.fj_msoftdrop[events_chunk[key]==1]
        #h_msd = events_chunk.fj_mass[events_chunk[key]==1]
        h_pt_over_mass = events_chunk.fj_genH_pt[events_chunk[key]==1]/events_chunk.fj_genH_mass[events_chunk[key]==1]
        #h_pt = events_chunk.fj_pt[events_chunk[key]==1]
        h_pt = events_chunk.fj_genH_pt[events_chunk[key]==1]
        hist_mass_hww.fill(process=key,
                           mass = h_mass,
                           pt_over_mass = h_pt_over_mass,
                           msd = h_msd,
                           #pt = h_pt,
                       )
                        
'''
for key in signals:
    fig, ax = plt.subplots(1,1)
    legs = []
    i=0
    #for m in reversed(range(50,200,20)):
    for m in range(50,200,20):
        # print(hist_mass_hww.integrate('process',key).identifiers("mass", overflow='all'))
        if i==0:
            hist.plot1d(hist_mass_hww.integrate('process',key).integrate('mass',slice(m,m+10)).integrate('pt_over_mass',slice(5,30)),ax=ax)
        else:
            hist.plot1d(hist_mass_hww.integrate('process',key).integrate('mass',slice(m,m+10)).integrate('pt_over_mass',slice(5,30)),ax=ax,clear=False)
        i+=1
        legs.append('mH=%i GeV'%m)
    leg = ax.legend(legs)
    #ax.set_xlabel(r'fj gen H $p_T/mass$')
    ax.set_xlabel(r'fj $m_{SD}$ (GeV)')
    #ax.set_xlabel(r'fj $p_T$ (GeV)')
    #ax.set_xlabel(r'fj gen H $p_T$ (GeV)')
    #ax.set_xlabel(r'fj mass (GeV)')
    ax.set_ylabel('Jets')
    #fig.savefig("fj_genHptovermass_bymh_%s.png"%key) 
    fig.savefig("fj_msoftdrop_bymh_%s_ptovermass_more5.png"%key)
    #fig.savefig("fj_pt_bymh_%s_ptovermass_more5.png"%key)
    #fig.savefig("fj_genHpt_bymh_%s.png"%key)
    #fig.savefig("fj_mass_bymh_%s.png"%key)  

for key in signals:
    fig, ax = plt.subplots(1,1)
    legs = []
    for i in range(5):
        if i==0:
            hist.plot1d(hist_mass_hww.sum('mass').integrate('pt_over_mass',slice(i,30)).integrate('process',key),ax=ax)
        else:
            hist.plot1d(hist_mass_hww.sum('mass').integrate('pt_over_mass',slice(i,30)).integrate('process',key),ax=ax,clear=False)
        legs.append('pt/m>%i'%i)
    leg = ax.legend(legs)
    ax.set_xlabel(r'fj $m_{SD}$ (GeV)')
    ax.set_ylabel('Jets') 
    fig.savefig("fj_msoftdrop_hww_ptovermass_%s.png"%key)

'''
fig, ax = plt.subplots(1,1)
#hist.plot1d(hist_mass_hww.sum('mass').integrate('pt_over_mass',slice(5,30)),ax=ax)                
hist.plot1d(hist_mass_hww.sum('mass').integrate('pt_over_mass',slice(0,5)),ax=ax)
leg = ax.legend([r'$4q$',r'$e\nu qq$',r'$\mu\nu qq$'])
#leg = ax.legend([r'$bb$',r'$cc$',r'$qq$'])
ax.set_xlabel(r'fj $m_{SD}$ (GeV)')
#ax.set_xlabel(r'gen H mass (GeV)')
ax.set_ylabel('Jets')
fig.savefig("fj_msoftdrop_hww_ptovermass_moverptcut_inverted.png")

