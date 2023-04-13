import uproot
import numpy as np
import matplotlib.pyplot as plt
from array import array

fileptr = uproot.open("TT_Dilept_13.root")
e_pt = fileptr["Delphes_Ntuples"]['elec_pt'].array()
e_eta = fileptr["Delphes_Ntuples"]['elec_eta'].array()
e_phi = fileptr["Delphes_Ntuples"]['elec_phi'].array()

mu_pt = fileptr["Delphes_Ntuples"]['muon_pt'].array()
mu_eta = fileptr["Delphes_Ntuples"]['muon_eta'].array()
mu_phi = fileptr["Delphes_Ntuples"]['muon_phi'].array()

jet_pt = fileptr["Delphes_Ntuples"]['jet_pt'].array()
jet_eta = fileptr["Delphes_Ntuples"]['jet_eta'].array()
jet_phi = fileptr["Delphes_Ntuples"]['jet_phi'].array()


e_pt_plotting=[]
e_eta_plotting=[]
e_phi_plotting=[]

mu_pt_plotting=[]
mu_eta_plotting=[]
mu_phi_plotting=[]

jet_pt_plotting=[]
jet_eta_plotting=[]
jet_phi_plotting=[]

for eventidx in range(len(e_pt)):
    for event in range(len(e_pt[eventidx])):
        e_pt_plotting.append(e_pt[eventidx][event])
        e_eta_plotting.append(e_eta[eventidx][event])
        e_phi_plotting.append(e_phi[eventidx][event])

    for event in range(len(mu_pt[eventidx])):
        mu_pt_plotting.append(mu_pt[eventidx][event])
        mu_eta_plotting.append(mu_eta[eventidx][event])
        mu_phi_plotting.append(mu_phi[eventidx][event])

    for event in range(len(jet_pt[eventidx])):
        jet_pt_plotting.append(jet_pt[eventidx][event])
        jet_eta_plotting.append(jet_eta[eventidx][event])
        jet_phi_plotting.append(jet_phi[eventidx][event])

fig, ax= plt.subplots(1,3)
ax[0].set_yscale('log')

## electrons
"""ax[0].hist(e_pt_plotting, bins= 5000)
ax[0].set_title('Electron Momentum')

ax[1].hist(e_eta_plotting, bins= 5000)
ax[1].set_xlim([-10, 10])
ax[1].set_title('Electron Eta')

ax[2].hist(e_phi_plotting, bins= 5000)
ax[2].set_title('Electron Phi')

plt.show()
"""
## muons

"""ax[0].hist(mu_pt_plotting, bins= 5000)
ax[0].set_title('Muon Momentum')

ax[1].hist(mu_eta_plotting, bins= 5000)
#ax[1].set_xlim([-10, 10])
ax[1].set_title('Muon Eta')

ax[2].hist(mu_phi_plotting, bins= 5000)
ax[2].set_title('Muon Phi')

plt.show()"""

## jets

ax[0].hist(jet_pt_plotting, bins= 5000)
ax[0].set_title('Jet Momentum')

ax[1].hist(jet_eta_plotting, bins= 5000)
#ax[1].set_xlim([-10, 10])
ax[1].set_title('Jet Eta')

ax[2].hist(jet_phi_plotting, bins= 5000)
ax[2].set_title('Jet Phi')

plt.show()





