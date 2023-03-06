import uproot
import numpy as np
from array import array
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input')
parser.add_argument('-o', '--output', help='Output')
args = parser.parse_args()


def deltaphi(e_phi, m_phi):
    d_phi = e_phi - m_phi
    if (d_phi > np.pi):
        d_phi -= 2 * np.pi
    if (d_phi < -np.pi):
        d_phi += 2 * np.pi
    return d_phi


def dR(e_phi, e_eta, m_phi, m_eta):
    d_eta = abs(e_eta - m_eta)
    d_phi = deltaphi(e_phi, m_phi)
    return np.sqrt(d_phi ** 2 + d_eta ** 2)


fileptr = uproot.open(args.input)

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
jet_phi = fileptr['Delphes_Ntuples']['jet_phi'].array()
jet_btag = fileptr['Delphes_Ntuples']['jet_btag'].array()

met_pt = fileptr['Delphes_Ntuples']['met_pt'].array()
met_phi = fileptr['Delphes_Ntuples']['met_phi'].array()

weight = fileptr['Delphes_Ntuples']['weight'].array()
scalar_ht = fileptr['Delphes_Ntuples']['scalar_ht'].array()

genjet_pt = fileptr['Delphes_Ntuples']['genjet_pt'].array()
genjet_eta = fileptr['Delphes_Ntuples']['genjet_eta'].array()
genjet_phi = fileptr['Delphes_Ntuples']['genjet_phi'].array()
genjet_mass = fileptr['Delphes_Ntuples']['genjet_mass'].array()

genpart_pt = fileptr['Delphes_Ntuples']['genpart_pt'].array()
genpart_eta = fileptr['Delphes_Ntuples']['genpart_eta'].array()
genpart_phi = fileptr['Delphes_Ntuples']['genpart_phi'].array()
genpart_mass = fileptr['Delphes_Ntuples']['genpart_mass'].array()
genpart_pid = fileptr['Delphes_Ntuples']['genpart_pid'].array()
genpart_status = fileptr['Delphes_Ntuples']['genpart_status'].array()
genpart_charge = fileptr['Delphes_Ntuples']['genpart_charge'].array()

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
j_btag = []

## ljet_pt, sl_jetpt --- done
ljet_pt = []
ljet_eta = []
ljet_phi = []
ljet_mass = []

sljet_pt = []
sljet_eta = []
sljet_phi = []
sljet_mass = []

l_pt = []
l_eta = []
l_phi = []
l_mass = []

sl_pt = []
sl_eta = []
sl_phi = []
sl_mass = []

e4vector = ROOT.TLorentzVector()
m4vector = ROOT.TLorentzVector()
final_array = [0] * len(elec_pt)
#print(len(final_array))

for event_idx in range(len(elec_pt)):
    e_idx = []
    mu_idx = []
    j_idx = []

    ef_idx = []
    muf_idx = []
    jf_idx = []

    e4vector = ROOT.TLorentzVector()
    m4vector = ROOT.TLorentzVector()

    ## electrons

    for i in range(len(elec_pt[event_idx])):
        if elec_pt[event_idx][i] < 20:
            continue
        if abs(elec_eta[event_idx][i]) > 2.4 or (1.4442 < abs(elec_eta[event_idx][i]) < 1.5660):
            continue
        e_idx.append(i)

    ## muons
    for i in range(len(muon_pt[event_idx])):
        if muon_pt[event_idx][i] < 20:
            continue
        if abs(muon_eta[event_idx][i]) > 2.4:
            continue
        if muon_reliso[event_idx][i] > 0.15:
            continue
        mu_idx.append(i)

    for i in range(len(e_idx)):
        for j in range(len(mu_idx)):

            tmp_e_idx = e_idx[i]
            tmp_mu_idx = mu_idx[j]

            if (elec_charge[event_idx][tmp_e_idx] * muon_charge[event_idx][tmp_mu_idx] == -1):
                ef_idx.append(tmp_e_idx)
                muf_idx.append(tmp_mu_idx)
    ##print(j_idx) --- ok till here

    # Ensure such a pairing exists
    if (len(ef_idx) == 0 or len(muf_idx) == 0):
        continue

    ## jets
    counter = 0
    for i in range(len(jet_pt[event_idx])):
        if jet_pt[event_idx][i] < 30:
            continue
        if abs(jet_eta[event_idx][i]) > 2.4 or (1.4442 < abs(jet_eta[event_idx][i]) < 1.5660):
            continue
        j_idx.append(i)

        if jet_btag[event_idx][i] > 0:
            counter += 1
    ##print(j_idx) --- ok till here

    if (dR(elec_phi[i][e_index], elec_eta[i][e_index], jet_phi[i][j], jet_eta[i][j]) < 0.4):
        continue
    if dR(elec_phi[event_idx][e_index], elec_eta[event_idx][e_index], jet_phi[event_idx][i],
          jet_eta[event_idx][i]) < 0.4 \
            or dR(muon_phi[event_idx][mu_index], muon_eta[event_idx][mu_index], jet_phi[event_idx][i],
                  jet_eta[event_idx][i]) < 0.4:
        continue
    e_index = ef_idx[0]
    mu_index = muf_idx[0]

    e4vector.SetPtEtaPhiM(elec_pt[event_idx][e_index], elec_eta[event_idx][e_index], elec_phi[event_idx][e_index],
                          elec_mass[event_idx][e_index])
    m4vector.SetPtEtaPhiM(elec_pt[event_idx][e_index], elec_eta[event_idx][e_index], elec_phi[event_idx][e_index],
                          elec_mass[event_idx][e_index])
    if (e4vector + m4vector).M() < 20:
        continue
    if counter == 0:
        continue

        # look for event with great pt and greater than 25 and append to corresponding array after cuts
    ## add corresponding eta, phi, mass value in l and sl

    if elec_pt[event_idx][e_index] > muon_pt[event_idx][mu_index] and elec_pt[event_idx][e_index] > 25:
        l_pt.append(elec_pt[event_idx][e_index])
        sl_pt.append(muon_pt[event_idx][mu_index])
        l_eta.append(elec_eta[event_idx][e_index])
        sl_eta.append(muon_eta[event_idx][mu_index])
        l_phi.append(elec_phi[event_idx][e_index])
        sl_phi.append(muon_phi[event_idx][mu_index])
        l_mass.append(elec_mass[event_idx][e_index])
        sl_mass.append(muon_mass[event_idx][mu_index])
        # add charge

    if (mu_pt[event_idx][mu_index] > elec_pt[event_idx][e_index]) and (muon_pt[event_idx][mu_index]) > 25:
        l_pt.append(muon_pt[event_idx][mu_index])
        sl_pt.append(elec_pt[event_idx][e_index])
        l_eta.append(muon_eta[event_idx][mu_index])
        sl_eta.append(elec_eta[event_idx][e_index])
        l_phi.append(muon_phi[event_idx][mu_index])
        sl_phi.append(elec_phi[event_idx][e_index])
        l_mass.append(muon_mass[event_idx][mu_index])
        sl_mass.append(elec_mass[event_idx][e_index])
    else:
        continue

    e_pt.append(elec_pt[event_idx][e_index])
    e_eta.append(elec_eta[event_idx][e_index])
    e_phi.append(elec_phi[event_idx][e_index])
    e_mass.append(elec_mass[event_idx][e_index])

    mu_pt.append(muon_pt[event_idx][mu_index])
    mu_eta.append(muon_eta[event_idx][mu_index])
    mu_phi.append(muon_phi[event_idx][mu_index])
    mu_mass.append(muon_mass[event_idx][mu_index])

    ljet_pt.append(jet_pt[event_idx][ljet_idx])
    ljet_phi.append(jet_phi[event_idx][ljet_idx])
    ljet_eta.append(jet_eta[event_idx][ljet_idx])
    ljet_mass.append(jet_mass[event_idx][ljet_idx])

    sljet_pt.append(jet_pt[event_idx][sljet_idx])
    sljet_phi.append(jet_phi[event_idx][sljet_idx])
    sljet_eta.append(jet_eta[event_idx][sljet_idx])
    sljet_mass.append(jet_mass[event_idx][sljet_idx])

    ## output file to save post cut arrays

# print(counter)
# plt.hist(mu_eta, bins=100)
# plt.show()

# print(e_charge[0])
# print(mu_charge[0])

## make histograms
outputfile = ROOT.TFile("args.Output", 'recreate')
tree = ROOT.TTree("CutTree", "CutTree")
# make arrays for each
elec_pt_arr = array('f', [0.])
elec_eta_arr = array('f', [0.])
elec_phi_arr = array('f', [0.])
elec_charge_arr = array('f', [0.])

mu_pt_arr = array('f', [0.])
mu_eta_arr = array('f', [0.])
mu_phi_arr = array('f', [0.])
mu_charge_arr = array('f', [0.])

ljet_pt_arr = array('f', [0.])
ljet_eta_arr = array('f', [0.])
ljet_phi_arr = array('f', [0.])
ljet_mass_arr = array('f', [0.])

sljet_pt_arr = array('f', [0.])
sljet_eta_arr = array('f', [0.])
sljet_phi_arr = array('f', [0.])
sljet_mass_arr = array('f', [0.])

l_pt_arr = array('f', [0.])
l_eta_arr = array('f', [0.])
l_phi_arr = array('f', [0.])
l_charge_arr = array('f', [0.])

sl_pt_arr = array('f', [0.])
sl_eta_arr = array('f', [0.])
sl_phi_arr = array('f', [0.])
sl_charge_arr = array('f', [0.])

# create tree.Branch
tree.Branch("elec_pt", elec_pt_arr, "elec_pt/F")
tree.Branch("elec_eta", elec_eta_arr, "elec_eta/F")
tree.Branch("elec_phi", elec_phi_arr, "elec_phi/F")
tree.Branch("elec_charge", elec_charge_arr, "elec_charge/F")

tree.Branch("mu_pt", mu_pt_arr, "mu_pt/F")
tree.Branch("mu_eta", mu_eta_arr, "mu_eta/F")
tree.Branch("mu_phi", mu_phi_arr, "mu_phi/F")
tree.Branch("mu_charge", mu_charge_arr, "mu_charge/F")

tree.Branch("ljet_pt", ljet_pt_arr, "ljet_pt/F")
tree.Branch("ljet_eta", ljet_eta_arr, "ljet_eta/F")
tree.Branch("ljet_phi", ljet_phi_arr, "ljet_phi/F")
tree.Branch("ljet_charge", ljet_mass_arr, "ljet_charge/F")

tree.Branch("sljet_pt", sljet_pt_arr, "sljet_pt/F")
tree.Branch("sljet_eta", sljet_eta_arr, "ljet_eta/F")
tree.Branch("sljet_phi", sljet_phi_arr, "ljet_phi/F")
tree.Branch("sljet_mass", sljet_mass_arr, "ljet_mass/F")

tree.Branch("l_pt", l_pt_arr, "l_pt/F")
tree.Branch("l_eta", l_eta_arr, "l_eta/F")
tree.Branch("l_phi", l_phi_arr, "l_phi/F")
tree.Branch("l_charge", l_charge_arr, "l_charge/F")

tree.Branch("sl_pt", sl_pt_arr, "sl_pt/F")
tree.Branch("sl_eta", sl_eta_arr, "sl_eta/F")
tree.Branch("sl_phi", sl_phi_arr, "sl_phi/F")
tree.Branch("sl_charge", sl_charge_arr, "sl_charge/F")

## tree fill for all the arrays

for i in range(len(e_pt)):
    e_pt_arr[0] = e_pt[i]
    e_phi_arr[0] = e_phi[i]
    e_eta_arr[0] = e_eta[i]
    e_charge_arr[0] = e_charge[i]

    mu_pt_arr[0] = mu_pt[i]
    mu_phi_arr[0] = mu_phi[i]
    mu_eta_arr[0] = mu_eta[i]
    mu_charge_arr[0] = mu_charge[i]

    ljet_pt_arr[0] = ljet_pt[i]
    ljet_phi_arr[0] = ljet_phi[i]
    ljet_eta_arr[0] = ljet_eta[i]
    ljet_mass_arr[0] = ljet_mass[i]

    sljet_pt_arr[0] = sljet_pt[i]
    sljet_phi_arr[0] = sljet_phi[i]
    sljet_eta_arr[0] = sljet_eta[i]
    sljet_mass_arr[0] = sljet_mass[i]

    l_pt_arr[0] = l_pt[i]
    l_phi_arr[0] = l_phi[i]
    l_eta_arr[0] = l_eta[i]
    l_charge_arr[0] = l_charge[i]

    sl_pt_arr[0] = sl_pt[i]
    sl_phi_arr[0] = sl_phi[i]
    sl_eta_arr[0] = sl_eta[i]
    sl_charge_arr[0] = sl_charge[i]

    tree.Fill()

outputfile.Write()
outputfile.Close()

## top quark reconstruction
## loook at the entanglements once we have quarks reconstructed
