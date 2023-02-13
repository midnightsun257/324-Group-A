
import uproot
import numpy as np
from   array import array

fileptr = uproot.open("TT_Dilept_13.root")

elec_pt = fileptr['Delphes_Ntuples']['elec_pt'].array()
elec_eta = fileptr['Delphes_Ntuples']['elec_eta'].array()
elec_phi = fileptr['Delphes_Ntuples']['elec_phi'].array()
elec_mass = fileptr['Delphes_Ntuples']['elec_mass'].array()
elec_charge = fileptr['Delphes_Ntuples']['elec_charge'].array()
elec_reliso = fileptr['Delphes_Ntuples']['elec_reliso'].array()

muon_pt = fileptr['Delphes_Ntuples']['muon_pt'].array()
muon_eta = fileptr['Delphes_Ntuples']['muon_eta'].array()
muon_phi = fileptr['Delphes_Ntuples']['muon_phi'].array()
muon_mass = fileptr['Delphes_Ntuples']['muon_mass'].array()
muon_charge = fileptr['Delphes_Ntuples']['muon_charge'].array()
muon_reliso = fileptr['Delphes_Ntuples']['muon_reliso'].array()

jet_pt = fileptr['Delphes_Ntuples']['jet_pt'].array()
jet_eta = fileptr['Delphes_Ntuples']['jet_eta'].array()
jet_btag = fileptr['Delphes_Ntuples']['jet_btag'].array()

e_pt = []
e_eta = []
e_phi = []
e_charge = []

mu_pt = []
mu_eta = []
mu_phi = []
mu_charge = []

j_pt = []
j_eta = []
j_btag= []

for event_idx in range(len(elec_pt)):
    e_idx = []
    mu_idx = []
    j_idx= []

    ef_idx = []
    muf_idx = []
    jf_idx=[]
## electrons
    for i in range(len(elec_pt[event_idx])):
        if elec_pt[event_idx][i] >= 20:
            continue
        if abs(elec_eta[event_idx][i])<=2.4:
            continue
        e_idx.append(i)
## muons
    for i in range(len(muon_pt[event_idx])):
        if muon_pt[event_idx][i] >= 20:
            continue
        if abs(muon_eta[event_idx][i])<=2.4:
            continue
        if muon_reliso[event_idx][i] <= 0.15:
            continue
        mu_idx.append(i)

## jets
    for i in range(len(jet_pt[event_idx])):
        if jet_pt[event_idx][i] >= 30:
            continue
        if abs(jet_eta[event_idx][i])<=2.4:
            continue
        j_idx.append(i)

    if (len(e_idx) == 0 or len(mu_idx) == 0):
        continue

## events with atleast 1 jet that has btag
    counter = 0
    for i in range(len(jet_btag[event_idx])):
        if jet_btag[event_idx][i] > 0:
            counter += 1
        if jet_btag[event_idx][i] == 0:
            continue
    print(counter)

    for i in range(len(e_idx)):
        for j in range(len(mu_idx)):

            tmp_e_idx = e_idx[i]
            tmp_mu_idx = mu_idx[j]

            if (elec_charge[event_idx][tmp_e_idx] * muon_charge[event_idx][tmp_mu_idx] == -1):
                ef_idx.append(tmp_e_idx)
                muf_idx.append(tmp_mu_idx)

        # Ensure such a pairing exists
    if (len(ef_idx) == 0 or len(muf_idx) == 0):
        continue

    e_index = ef_idx[0]
    mu_index = muf_idx[0]

    e_pt.append(elec_pt[event_idx][e_index])
    e_eta.append(elec_eta[event_idx][e_index])
    e_phi.append(elec_phi[event_idx][e_index])
    e_charge.append(elec_charge[event_idx][e_index])

    mu_pt.append(muon_pt[event_idx][mu_index])
    mu_eta.append(muon_eta[event_idx][mu_index])
    mu_phi.append(muon_phi[event_idx][mu_index])
    mu_charge.append(muon_charge[event_idx][mu_index])

#print(e_charge[0])
#print(mu_charge[0])



'''
outputfile = ROOT.TFile("NewRoot.root", 'recreate')
tree = ROOT.TTree("CutTree", "CutTree")

elec_pt_arr = array('f', [0.])

tree.Branch("elec_pt", elec_pt_arr, "elec_pt/F")




outputfile.Write()
outputfile.Close()
'''