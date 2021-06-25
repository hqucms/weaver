import uproot
import awkward as ak
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep

from coffea import hist
hists = {'fj_prop': hist.Hist("fj",
                              hist.Bin("msd", r"fj msoftdrop [GeV]", 50, 20, 380),
                              hist.Bin("pt", r"fj pT [GeV]", 50, 300, 1000),
                              hist.Bin("pn", r"fj PN non-MD", 30, 0, 1),
                              hist.Bin("mass", r"fj mass [GeV]", 30, 50, 200),
                              hist.Bin("gmass", r"gen H mass [GeV]", 30, 50, 200),
                          ),
         #'h_gen': hist.Hist("h",
         #                   hist.Bin("gmass", r"gen H mass [GeV]", 30, 50, 200),
         #                   hist.Bin("gpt_over_mass", "gen H pT/mass", 30, 0, 25),
         #                   hist.Bin("gpt", r" gen H pt [GeV]", 30, 200, 1600),
         #               ),
         'fj_dr': hist.Hist("dr",
                            hist.Bin("gmass", r"gen H mass [GeV]", 30, 50, 200),
                            hist.Bin("dr_W", r"dR(fj,W)", 25, 0, 0.6),
                            hist.Bin("dr_Wstar", r"dR(fj,W*)", 25, 0, 1),
                            hist.Bin("min_dr_Wdau", r"min dR(fj,4qs)", 30, 0, 0.6),
                            hist.Bin("max_dr_Wdau", r"max dR(fj,4qs)", 30, 0, 1.2)
                            ),
         'w_gen': hist.Hist("w",
                            hist.Bin("wmass", r"gen W mass [GeV]", 30, 10, 110),
                            hist.Bin("wsmass", r"gen W* mass [GeV]", 30, 10, 100),
                            hist.Bin("wpt", r"gen W pT [GeV]", 30, 50, 1000),
                            hist.Bin("wspt", r"gen W* pT [GeV]", 30, 50, 1000),
                            hist.Bin("gmass", r"gen H mass [GeV]", 30, 50, 200),
                        ),
     }

hist_prop = {'fj_prop': ['msd','pt','pn','mass'],
             'fj_dr': ['dr_W','dr_Wstar','min_dr_Wdau','max_dr_Wdau'],
             #'h_gen': ['gpt_over_mass','gpt'],
             'w_gen': ['wmass','wsmass','wpt','wspt'],
}

# outfile = "output/v03_ak8h4q_ep20_Jun9_ptmcut_onlyh4qandflatsamples.root"
#train_names = "/data/shared/cmantill/training/ak8_v03hww_genHmpt_Jun8/train/GravitonToHHToWWWW/*.root"
#test_names = "/data/shared/cmantill/training/ak8_v03hww_genHmpt_Jun8/test/HHToBBVVToBBQQQQ_cHHH1/*.root"

train_names = "/data/shared/cmantill/training/ak15_v03hww_May26/train/GravitonToHHToWWWW/*.root"
#test_names = "/data/shared/cmantill/training/ak15_v03hww_May26/test/HHToBBVVToBBQQQQ_cHHH1/*.root"
test_names = "/data/shared/cmantill/training/ak15_v03hww_May26/test/GluGluToHHTo4V_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/*.root"

branches = ["fj_genH_mass",#"fj_genH_pt",
            #"label_4q",
            "fj_H_WW_4q",
            "fj_msoftdrop","fj_pt","fj_mass","fj_PN_H4qvsQCD",
            "fj_dR_W","fj_dR_Wstar","fj_mindR_HWW_daus","fj_maxdR_HWW_daus",
            "fj_genW_pt","fj_genWstar_pt","fj_genW_mass","fj_genWstar_mass",
]

# events_dict = uproot.open(outfile+":Events").arrays(branches)

events = uproot.iterate([{train_names+":Events"},{test_names+":Events"}], branches)

for events_dict in events:
    # signal_mask = events_dict["label_4q"]==1
    signal_mask = events_dict["fj_H_WW_4q"]==1

    h_mass = events_dict.fj_genH_mass[signal_mask]
    
    hists['fj_prop'].fill(
        msd = events_dict.fj_msoftdrop[signal_mask],
        pt = events_dict.fj_pt[signal_mask],
        pn = events_dict.fj_PN_H4qvsQCD[signal_mask],
        mass = events_dict.fj_mass[signal_mask],
        gmass = h_mass
    )
    '''
    hists['h_gen'].fill(
        gpt_over_mass = events_dict.fj_genH_pt[signal_mask]/events_dict.fj_genH_mass[signal_mask],
        gpt = events_dict.fj_genH_pt[signal_mask],
        gmass = h_mass
    )
    '''
    hists['w_gen'].fill(
        wmass = events_dict.fj_genW_mass[signal_mask],
        wsmass = events_dict.fj_genWstar_mass[signal_mask],
        wpt = events_dict.fj_genW_pt[signal_mask],
        wspt = events_dict.fj_genWstar_pt[signal_mask],
        gmass = h_mass,
    )
    
    hists['fj_dr'].fill(
        gmass = h_mass,
        dr_W = events_dict.fj_dR_W[signal_mask],
        dr_Wstar = events_dict.fj_dR_Wstar[signal_mask],
        min_dr_Wdau = events_dict.fj_mindR_HWW_daus[signal_mask],
        max_dr_Wdau= events_dict.fj_maxdR_HWW_daus[signal_mask],
    )

masses = [125,120]
masses += list(range(50,200,20))
masses.remove(70)
masses.remove(110)
masses.remove(150)

invmasses = list(reversed(range(50,200,20)))
invmasses += [120,125]
invmasses.remove(70)
invmasses.remove(110)
invmasses.remove(150)

for hname,variables in hist_prop.items():
    for var in variables:
        for i in range(2):
            fig, ax = plt.subplots(1,1)
            legs = []
            
            lmasses = masses
            if var in ['wsmass']:
                lmasses = invmasses
            if var in ['dr_W','dr_Wstar','min_dr_Wdau','max_dr_Wdau','msd','mass']:
                lmasses = list(range(50,200,20)) + [120,125]
                lmasses.remove(70)
                lmasses.remove(110)
                lmasses.remove(150)
                
            for j,m in enumerate(lmasses):
                if m==120 or m==125:
                    h = hists[hname].sum(*[ax for ax in hists[hname].axes() if ax.name not in {'gmass',var}]).integrate('gmass',m)
                else:
                    h = hists[hname].sum(*[ax for ax in hists[hname].axes() if ax.name not in {'gmass',var}]).integrate('gmass',slice(m,m+10))
                if j==0:
                    if i==0:
                        hist.plot1d(h,ax=ax, density=True)
                    else:
                        hist.plot1d(h,ax=ax)
                else:
                    if i==0:
                        hist.plot1d(h,ax=ax, clear=False, density=True) 
                    else:
                        hist.plot1d(h,ax=ax, clear=False)
                legs.append('mH=%i GeV'%m)
            leg = ax.legend(legs)
            ax.set_ylabel('Jets')
            if i==0:
                fig.savefig("features_ak15/%s_by_mh_density.png"%var)
            else:
                fig.savefig("features_ak15/%s_by_mh.png"%var)
