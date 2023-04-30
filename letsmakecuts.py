import ROOT
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

## create output file
outputfile = ROOT.TFile(args.output, 'recreate')
tree = ROOT.TTree("CutTree", "CutTree")

elec_pt = fileptr['Delphes_Ntuples']['elec_pt'].array()
elec_eta = fileptr['Delphes_Ntuples']['elec_eta'].array()
elec_phi = fileptr['Delphes_Ntuples']['elec_phi'].array()
elec_mass = fileptr['Delphes_Ntuples']['elec_mass'].array()
elec_charge = fileptr['Delphes_Ntuples']['elec_charge'].array()
#elec_reliso = fileptr['Delphes_Ntuples']['elec_reliso'].array()

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
jet_mass = fileptr['Delphes_Ntuples']['jet_mass'].array()

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
e_mass = []

mu_pt = []
mu_eta = []
mu_phi = []
mu_charge = []
mu_mass = []

j_pt = []
j_eta = []
j_btag = []
j_mass = []

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
l_charge = []

sl_pt = []
sl_eta = []
sl_phi = []
sl_mass = []
sl_charge = []

met_pt_arr = []
met_phi_arr = []

total_pt = []
total_jet_pt = []
scalar_ht_arr = []

llbar_deta = []
llbar_dphi = []
bbbar_deta = []
bbbar_dphi = []


final_array = np.zeros(len(elec_pt))
# print(len(final_array))

for event_idx in range(len(elec_pt)):
    e_idx = []
    ef_idx = []

    mu_idx = []
    muf_idx = []

    j_idx = []
    jf_idx = []

    # check if its running
    # if event_idx%1000 == 0:
    #   print((event_idx/len(elec_pt)))

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

    if (len(e_idx) == 0 or len(mu_idx) == 0):
        continue

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

    ## jets
    counter = 0
    for i in range(len(jet_pt[event_idx])):
        if jet_pt[event_idx][i] < 30:
            continue
        if abs(jet_eta[event_idx][i]) > 2.4 or (1.4442 < abs(jet_eta[event_idx][i]) < 1.5660):
            continue
        if dR(elec_phi[event_idx][e_index], elec_eta[event_idx][e_index], jet_phi[event_idx][i],
              jet_eta[event_idx][i]) < 0.4 or dR(muon_phi[event_idx][mu_index], muon_eta[event_idx][mu_index],
                                                 jet_phi[event_idx][i], jet_eta[event_idx][i]) < 0.4:
            continue

        j_idx.append(i)

        if jet_btag[event_idx][i] > 0:
            counter += 1
    if len(j_idx) < 2:
        continue
    if counter == 0:
        continue


    ljet_idx = j_idx[0]
    sljet_idx = j_idx[1]

    e4vector.SetPtEtaPhiM(elec_pt[event_idx][e_index], elec_eta[event_idx][e_index], elec_phi[event_idx][e_index],
                          elec_mass[event_idx][e_index])
    m4vector.SetPtEtaPhiM(muon_pt[event_idx][mu_index], muon_eta[event_idx][mu_index], muon_phi[event_idx][mu_index],
                          muon_mass[event_idx][mu_index])
    if (e4vector + m4vector).M() < 20:
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
        l_charge.append(elec_charge[event_idx][e_index])
        sl_charge.append(muon_charge[event_idx][mu_index])

    elif (muon_pt[event_idx][mu_index] > elec_pt[event_idx][e_index]) and (muon_pt[event_idx][mu_index]) > 25:
        l_pt.append(muon_pt[event_idx][mu_index])
        sl_pt.append(elec_pt[event_idx][e_index])
        l_eta.append(muon_eta[event_idx][mu_index])
        sl_eta.append(elec_eta[event_idx][e_index])
        l_phi.append(muon_phi[event_idx][mu_index])
        sl_phi.append(elec_phi[event_idx][e_index])
        l_mass.append(muon_mass[event_idx][mu_index])
        sl_mass.append(elec_mass[event_idx][e_index])
        l_charge.append(elec_charge[event_idx][e_index])
        sl_charge.append(muon_charge[event_idx][mu_index])
    else:
        continue

    e_pt.append(elec_pt[event_idx][e_index])
    e_eta.append(elec_eta[event_idx][e_index])
    e_phi.append(elec_phi[event_idx][e_index])
    e_mass.append(elec_mass[event_idx][e_index])
    e_charge.append(elec_charge[event_idx][e_index])

    mu_pt.append(muon_pt[event_idx][mu_index])
    mu_eta.append(muon_eta[event_idx][mu_index])
    mu_phi.append(muon_phi[event_idx][mu_index])
    mu_mass.append(muon_mass[event_idx][mu_index])
    mu_charge.append(muon_charge[event_idx][mu_index])

    ljet_pt.append(jet_pt[event_idx][ljet_idx])
    ljet_phi.append(jet_phi[event_idx][ljet_idx])
    ljet_eta.append(jet_eta[event_idx][ljet_idx])
    ljet_mass.append(jet_mass[event_idx][ljet_idx])

    sljet_pt.append(jet_pt[event_idx][sljet_idx])
    sljet_phi.append(jet_phi[event_idx][sljet_idx])
    sljet_eta.append(jet_eta[event_idx][sljet_idx])
    sljet_mass.append(jet_mass[event_idx][sljet_idx])

    met_pt_arr.append(met_pt[event_idx][0])
    met_phi_arr.append(met_pt[event_idx][0])
    scalar_ht_arr.append(scalar_ht[event_idx][0])
    temp_total_pt = 0
    temp_total_jet_pt = 0

    final_array[event_idx] = 1
    for i in range(len(elec_pt[event_idx])):
        temp_total_pt += elec_pt[event_idx][i]

    for i in range(len(muon_pt[event_idx])):
        temp_total_pt += muon_pt[event_idx][i]

    for i in range(len(jet_pt[event_idx])):
        temp_total_jet_pt += jet_pt[event_idx][i]
    temp_total_pt += temp_total_jet_pt

    temp_total_pt += (met_pt[event_idx][0])

    total_jet_pt.append(temp_total_jet_pt)
    total_pt.append(temp_total_pt)

# print(len(e_pt))

## output file to save post cut arrays

# print(counter)
# plt.hist(mu_eta, bins=100)
# plt.show()

# print(e_charge[0])
# print(mu_charge[0])

final_array = np.array(final_array)

selected_weight = weight[final_array == 1]
selected_jet_pt = jet_pt[final_array == 1]
selected_jet_eta = jet_eta[final_array == 1]
selected_jet_phi = jet_phi[final_array == 1]
selected_jet_mass = jet_mass[final_array == 1]
selected_jet_btag = jet_btag[final_array == 1]
selected_genjet_eta = genjet_eta[final_array == 1]
selected_genjet_mass = genjet_mass[final_array == 1]
selected_genjet_pt = genjet_pt[final_array == 1]
selected_genjet_phi = genjet_phi[final_array == 1]
selected_genpart_pt = genpart_pt[final_array == 1]
selected_genpart_eta = genpart_eta[final_array == 1]
selected_genpart_phi = genpart_phi[final_array == 1]
selected_genpart_mass = genpart_mass[final_array == 1]
selected_genpart_pid = genpart_pid[final_array == 1]
selected_genpart_status = genpart_status[final_array == 1]
selected_genpart_charge = genpart_charge[final_array == 1]

ljet_eta = np.array(ljet_eta)
sljet_eta = np.array(sljet_eta)

ljet_phi = np.array(ljet_phi)
sljet_phi = np.array(sljet_phi)

l_eta = np.array(l_eta)
sl_eta = np.array(sl_eta)

l_phi = np.array(l_phi)
sl_phi = np.array(sl_phi)

bbbar_dphi = abs(abs(abs(ljet_phi - sljet_phi) - np.pi) - np.pi)
bbbar_deta = abs(ljet_eta - sljet_eta)

llbar_dphi = abs(abs(abs(l_phi - sl_phi) - np.pi) - np.pi)
llbar_deta = abs(l_eta - sl_eta)

# make arrays for each
elec_pt_arr = array('f', [0.])
elec_eta_arr = array('f', [0.])
elec_phi_arr = array('f', [0.])
elec_charge_arr = array('f', [0.])
elec_mass_arr = arrary('f', [0.])

mu_pt_arr = array('f', [0.])
mu_eta_arr = array('f', [0.])
mu_phi_arr = array('f', [0.])
mu_charge_arr = array('f', [0.])
mu_mass_arr = arrary('f', [0.])

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
l_mass_arr = array('f', [.0])

sl_pt_arr = array('f', [0.])
sl_eta_arr = array('f', [0.])
sl_phi_arr = array('f', [0.])
sl_charge_arr = array('f', [0.])
sl_mass_arr = array('f', [.0])

MET_pt_arr = array('f', [0.])
MET_phi_arr = array('f', [0.])

total_pt_arr = array('f', [0.])
total_jet_pt_arr = array('f', [0.])
#scalar_ht_pt_arr = array('f', [0.])

llbar_deta_arr = array('f', [0.])
llbar_dphi_arr = array('f', [0.])
bbbar_deta_arr = array('f', [0.])
bbbar_dphi_arr = array('f', [0.])

weight_size_arr = array('i', [0])
selected_weight_arr = array('f',10000 * [0.])

jet_pt_arr = array('f', 10000 * [0.])
jet_eta_arr = array('f', 10000 * [0.])
jet_phi_arr = array('f', 10000 * [0.])
jet_btag_arr = array('f', 10000 * [0.])
jet_mass_arr= array('f', 10000* [0.])
jet_size_arr = array('i', [0])

genjet_eta_arr = array('f', 10000 * [0.])
genjet_mass_arr = array('f', 10000 * [0.])
genjet_pt_arr = array('f', 10000 * [0.])
genjet_phi_arr = array('f', 10000 * [0.])
genjet_size_arr = array('i', [0])

genpart_pt_arr = array('f', 10000 * [0.])
genpart_eta_arr = array('f', 10000 * [0.])
genpart_phi_arr = array('f', 10000 * [0.])
genpart_mass_arr = array('f', 10000 * [0.])
genpart_pid_arr = array('f', 10000 * [0.])
genpart_status_arr = array('f', 10000 * [0.])
genpart_charge_arr = array('f', 10000 * [0.])
genpart_size_arr = array('i', [0])

# create tree.Branch
tree.Branch("weight_size", weight_size_arr, "weight_size/I")
tree.Branch("selected_weight", selected_weight_arr, "selected_weight[weight_size]/F")

tree.Branch("selected_elec_pt", elec_pt_arr, "selected_elec_pt/F")
tree.Branch("selected_elec_eta", elec_eta_arr, "selected_elec_eta/F")
tree.Branch("selected_elec_phi", elec_phi_arr, "selected_elec_phi/F")
tree.Branch("selected_elec_charge", elec_charge_arr, "selected_elec_charge/F")
tree.Branch('selected_elec_mass', elec_mass_arr, 'selected_elec_mass/F')

tree.Branch("selected_mu_pt", mu_pt_arr, "selected_mu_pt/F")
tree.Branch("selected_mu_eta", mu_eta_arr, "selected_mu_eta/F")
tree.Branch("selected_mu_phi", mu_phi_arr, "selected_mu_phi/F")
tree.Branch("selected_mu_charge", mu_charge_arr, "selected_mu_charge/F")
tree.Branch('selected_mu_mass', mu_mass_arr, 'selected_mu_mass/F')

tree.Branch("selected_ljet_pt", ljet_pt_arr, "selected_ljet_pt/F")
tree.Branch("selected_ljet_eta", ljet_eta_arr, "selected_ljet_eta/F")
tree.Branch("selected_ljet_phi", ljet_phi_arr, "selected_ljet_phi/F")
tree.Branch("selected_ljet_charge", ljet_mass_arr, "selected_ljet_charge/F")

tree.Branch("selected_sljet_pt", sljet_pt_arr, "selected_sljet_pt/F")
tree.Branch("selected_sljet_eta", sljet_eta_arr, "selected_ljet_eta/F")
tree.Branch("selected_sljet_phi", sljet_phi_arr, "selected_ljet_phi/F")
tree.Branch("selected_sljet_mass", sljet_mass_arr, "selected_ljet_mass/F")

tree.Branch("selected_l_pt", l_pt_arr, "selected_l_pt/F")
tree.Branch("selected_l_eta", l_eta_arr, "selected_l_eta/F")
tree.Branch("selected_l_phi", l_phi_arr, "selected_l_phi/F")
tree.Branch("selected_l_charge", l_charge_arr, "selected_l_charge/F")

tree.Branch("selected_sl_pt", sl_pt_arr, "selected_sl_pt/F")
tree.Branch("selected_sl_eta", sl_eta_arr, "selected_sl_eta/F")
tree.Branch("selected_sl_phi", sl_phi_arr, "selected_sl_phi/F")
tree.Branch("selected_sl_charge", sl_charge_arr, "selected_sl_charge/F")

tree.Branch("jet_size", jet_size_arr, "jet_size/I")
tree.Branch("selected_jet_pt", jet_pt_arr, "selected_jet_pt[jet_size]/F")
tree.Branch("selected_jet_eta", jet_eta_arr, "selected_jet_eta[jet_size]/F")
tree.Branch("selected_jet_phi", jet_phi_arr, "selected_jet_phi[jet_size]/F")
tree.Branch("selected_jet_mass", jet_mass_arr, "selected_jet_mass[jet_size]/F")
tree.Branch("selected_jet_btag", jet_btag_arr, "selected_jet_btag[jet_size]/F")

tree.Branch("genjet_size", genjet_size_arr, "genjet_size/I")
tree.Branch("selected_genjet_eta", genjet_eta_arr, "selected_genjet_eta[genjet_size]/F")
tree.Branch("selected_genjet_mass", genjet_mass_arr, "selected_genjet_mass[genjet_size]/F")
tree.Branch("selected_genjet_pt", genjet_pt_arr, "selected_genjet_pt[genjet_size]/F")
tree.Branch("selected_genjet_phi", genjet_phi_arr, "selected_genjet_phi[genjet_size]/F")

tree.Branch("genpart_size", genpart_size_arr, "genpart_size/I")
tree.Branch("selected_genpart_pt", genpart_pt_arr, "selected_genpart_pt[genpart_size]/F")
tree.Branch("selected_genpart_eta", genpart_eta_arr, "selected_genpart_eta[genpart_size]/F")
tree.Branch("selected_genpart_phi", genpart_phi_arr, "selected_genpart_phi[genpart_size]/F")
tree.Branch("selected_genpart_mass", genpart_mass_arr, "selected_genpart_mass[genpart_size]/F")
tree.Branch("selected_genpart_pid", genpart_pid_arr, "selected_genpart_pid[genpart_size]/F")
tree.Branch("selected_genpart_status", genpart_status_arr, "selected_genpart_status[genpart_size]/F")
tree.Branch("selected_genpart_charge", genpart_charge_arr, "selected_genpart_charge[genpart_size]/F")

tree.Branch("llbar_deta", llbar_deta_arr, "llbar_deta/F")
tree.Branch("llphi_deta", llbar_dphi_arr, "llbar_dphi/F")
tree.Branch("bbbar_deta", bbbar_deta_arr, "bbbar_deta/F")
tree.Branch("bbbar_dphi", bbbar_dphi_arr, "bbbar_dphi/F")

tree.Branch("met_pt_arr", MET_pt_arr, "met_pt_arr/F")
tree.Branch("met_phi_arr", MET_phi_arr, "met_phi_arr/F")
## tree fill for all the arrays

for i in range(len(e_pt)):
    elec_pt_arr[0] = e_pt[i]
    elec_phi_arr[0] = e_phi[i]
    elec_eta_arr[0] = e_eta[i]
    elec_charge_arr[0] = e_charge[i]
    elec_mass_arr[0] = e_mass[i]

    mu_pt_arr[0] = mu_pt[i]
    mu_phi_arr[0] = mu_phi[i]
    mu_eta_arr[0] = mu_eta[i]
    mu_charge_arr[0] = mu_charge[i]
    mu_mass_arr[0] = mu_mass[i]

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
    l_mass_arr[0] = l_mass[i]

    sl_pt_arr[0] = sl_pt[i]
    sl_phi_arr[0] = sl_phi[i]
    sl_eta_arr[0] = sl_eta[i]
    sl_charge_arr[0] = sl_charge[i]
    sl_mass_arr[0] = sl_mass[i]

    llbar_deta_arr[0] = llbar_deta[i]
    llbar_dphi_arr[0] = llbar_dphi[i]
    bbbar_deta_arr[0] = bbbar_deta[i]
    bbbar_dphi_arr[0] = bbbar_dphi[i]

    weight_size_arr[0] = len(selected_weight[i])
    jet_size_arr[0] = len(jet_pt[i])
    genjet_size_arr[0] = len(selected_genjet_pt[i])
    genpart_size_arr[0] = len(selected_genpart_pt[i])
    
    MET_pt_arr[0]= met_pt_arr[i]
    MET_phi_arr[0]= met_phi_arr[i]

    for j in range(weight_size_arr[0]):
        selected_weight_arr[j] = selected_weight[i][j]

    for j in range(jet_size_arr[0]):
        jet_pt_arr[j] = jet_pt[i][j]
        jet_eta_arr[j] = jet_eta[i][j]
        jet_phi_arr[j] = jet_phi[i][j]
        jet_btag_arr[j] = jet_btag[i][j]
        jet_mass_arr[j] = jet_mass[i][j]

    for j in range(genpart_size_arr[0]):
        genpart_pt_arr[j] = selected_genpart_pt[i][j]
        genpart_eta_arr[j] = selected_genpart_eta[i][j]
        genpart_phi_arr[j] = selected_genpart_phi[i][j]
        genpart_mass_arr[j] = selected_genpart_mass[i][j]
        genpart_pid_arr[j] =selected_genpart_pid[i][j]
        genpart_status_arr[j] = selected_genpart_status[i][j]
        genpart_charge_arr[j] = selected_genpart_charge[i][j]

    for j in range(genjet_size_arr[0]):
        genjet_eta_arr[j] = selected_genjet_eta[i][j]
        genjet_mass_arr[j] = selected_genjet_mass[i][j]
        genjet_phi_arr[j] = selected_genjet_phi[i][j]
        genjet_pt_arr[j] = selected_genjet_pt[i][j]

    tree.Fill()

outputfile.Write()
outputfile.Close()

## top quark reconstruction
## loook at the entanglements once we have quarks reconstructed
