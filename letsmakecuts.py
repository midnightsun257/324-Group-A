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
mu_mass= []

j_pt = []
j_eta = []
j_btag = []
j_mass = []

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
l_charge = []

sl_pt = []
sl_eta = []
sl_phi = []
sl_mass = []
sl_charge = []

MET_pt= []
MET_phi=[]

total_pt = []
total_jet_pt= []
SCALAR_ht = []

llbar_deta = []
llbar_dphi = []
bbbar_deta = []
bbbar_dphi = []

total_jet_pt = []
total_pt = []

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

   #check if its running
    #if event_idx%1000 == 0:
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
    e_index = ef_idx[0]
    mu_index = muf_idx[0]

    #check e_index
    #print(e_index)

    e4vector.SetPtEtaPhiM(elec_pt[event_idx][e_index], elec_eta[event_idx][e_index], elec_phi[event_idx][e_index],
                          elec_mass[event_idx][e_index])
    m4vector.SetPtEtaPhiM(muon_pt[event_idx][mu_index], muon_eta[event_idx][mu_index], muon_phi[event_idx][mu_index],
                          muon_mass[event_idx][mu_index])
    if (e4vector + m4vector).M() < 20:
        continue

    ## jets
     # where is ljet_idx and sljet_idx ??
    counter = 0
    for i in range(len(jet_pt[event_idx])):
        if dR(elec_phi[event_idx][e_index], elec_eta[event_idx][e_index], jet_phi[event_idx][i],
              jet_eta[event_idx][i]) < 0.4 or dR(muon_phi[event_idx][mu_index], muon_eta[event_idx][mu_index],
                                                 jet_phi[event_idx][i], jet_eta[event_idx][i]) < 0.4:
            continue
        if jet_pt[event_idx][i] < 30:
            continue
        if abs(jet_eta[event_idx][i]) > 2.4 or (1.4442 < abs(jet_eta[event_idx][i]) < 1.5660):
            continue
        j_idx.append(i)


        if jet_btag[event_idx][i] > 0:
            counter += 1
    if len(j_idx) < 2:
        continue
    if counter == 0:
        continue

   # print(len(j_idx))
    ljet_idx = j_idx[0]
    sljet_idx = j_idx[1]
   # print(j_idx) #--- ok till here
    '''    
    llbar_deta= np.abs(np.asarray(l_eta)- np.asarray(sl_eta))
    llbar_dphi= np.abs(np.abs(np.abs(np.asarray(l_phi)- np.asarray(sl_phi)) - np.pi) - np.pi)
    bbbar_deta= np.abs(np.asarray(ljet_eta)- np.asarray(sljet_eta))
    bbbar_dphi= np.abs(np.abs(np.abs(np.asarray(ljet_phi)- np.asarray(sljet_phi)) - np.pi) - np.pi)
'''

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
        # add charge

    if (muon_pt[event_idx][mu_index] > elec_pt[event_idx][e_index]) and (muon_pt[event_idx][mu_index]) > 25:
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

    #print (e_pt)

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

    temp_total_pt = 0
    temp_total_jet_pt =0
    
    for i in range(len(elec_pt[event_idx])):
        temp_total_pt += elec_pt[event_idx][i]

    for i in range(len(muon_pt[event_idx])):
        temp_total_pt += muon_pt[event_idx][i]
        
    for i in range(len(jet_pt[event_idx])):
        temp_total_pt += jet_pt[event_idx][i]
        temp_total_jet_pt += jet_pt[event_idx][i]
    
    temp_total_pt += (met_pt[event_idx][0])
    SCALAR_ht.append(scalar_ht[event_idx][0])
    MET_pt.append(met_pt[event_idx][0])
    MET_phi.append(met_phi[event_idx][0])
    
    total_jet_pt.append(temp_total_jet_pt)
    total_pt.append(temp_total_pt)
    

#print(len(e_pt))
        
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



## make histograms
outputfile = ROOT.TFile(args.output, 'recreate')
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

MET_pt_arr = array('f', [0.])
MET_phi_arr = array('f', [0.])

Total_pt_arr = array('f', [0.])
Total_jet_pt_arr = array('f', [0.])
scalar_ht_pt_arr = array('f', [0.])

llbar_deta_arr = array('f', [0.])
llbar_dphi_arr = array('f', [0.])
bbbar_deta_arr = array('f', [0.])
bbbar_dphi_arr = array('f', [0.])

weight_arr = array('f', [0.])

jet_pt_arr = array('f', 10000*[0.])
jet_eta_arr = array('f', 10000*[0.])
jet_phi_arr = array('f', 10000*[0.])
jet_btag_arr = array('f', 10000*[0.])

genjet_eta_arr = array('f', 10000*[0.])
genjet_mass_arr = array('f', 10000*[0.])
genjet_pt_arr = array('f', 10000*[0.])
genjet_phi_arr = array('f', 10000*[0.])

genpart_pt_arr = array('f', 10000*[0.])
genpart_eta_arr = array('f', 10000*[0.])
genpart_phi_arr = array('f', 10000*[0.])
genpart_mass_arr = array('f', 10000*[0.])
genpart_pid_arr = array('f', 10000*[0.])
genpart_status_arr = array('f', 10000*[0.])
genpart_charge_arr = array('f', 10000*[0.])

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

tree.Branch("jet_pt", jet_pt_arr, "jet_pt/F")
tree.Branch("jet_eta", jet_eta_arr, "jet_eta/F")
tree.Branch("jet_phi", jet_phi_arr, "jet_phi/F")
tree.Branch("jet_btag", jet_btag_arr, "jet_btag/F")

tree.Branch("genjet_eta", genjet_eta_arr, "genjet_eta/F")
tree.Branch("genjet_mass", genjet_mass_arr, "genjet_mass/F")
tree.Branch("genjet_pt", genjet_pt_arr, "genjet_pt/F")
tree.Branch("genjet_phi", genjet_phi_arr, "genjet_phi/F")

tree.Branch("genpart_pt", genpart_pt_arr, "genpart_pt/F")
tree.Branch("genpart_eta", genpart_eta_arr, "genpart_eta/F")
tree.Branch("genpart_phi", genpart_phi_arr, "genpart_phi/F")
tree.Branch("genpart_mass", genpart_mass_arr, "genpart_mass/F")
tree.Branch("genpart_pid", genpart_pid_arr, "genpart_pid/F")
tree.Branch("genpart_status", genpart_status_arr, "genpart_status/F")
tree.Branch("genpart_charge", genpart_charge_arr, "genpart_charge/F")

tree.Branch("genjet_eta", genjet_eta_arr, "genjet_eta/F")
tree.Branch("genjet_mass", genjet_mass_arr, "genjet_mass/F")
tree.Branch("genjet_pt", genjet_pt_arr, "genjet_pt/F")
tree.Branch("genjet_phi", genjet_phi_arr, "genjet_phi/F")

tree.Branch("genpart_pt", genpart_pt_arr, "genpart_pt/F")
tree.Branch("genpart_eta", genpart_eta_arr, "genpart_eta/F")
tree.Branch("genpart_phi", genpart_phi_arr, "genpart_phi/F")
tree.Branch("genpart_mass", genpart_mass_arr, "genpart_mass/F")
tree.Branch("genpart_pid", genpart_pid_arr, "genpart_pid/F")
tree.Branch("genpart_status", genpart_status_arr, "genpart_status/F")
tree.Branch("genpart_charge", genpart_charge_arr, "genpart_charge/F")

tree.Branch("llbar_deta", llbar_deta_arr, "llbar_deta/F")
tree.Branch("llphi_deta", llbar_dphi_arr, "llbar_dphi/F")
tree.Branch("bbbar_deta", bbbar_deta_arr, "bbbar_deta/F")
tree.Branch("bbbar_dphi", bbbar_dphi_arr, "bbbar_dphi/F")

## tree fill for all the arrays

for i in range(len(e_pt)):
    elec_pt_arr[0] = e_pt[i]
    elec_phi_arr[0] = e_phi[i]
    elec_eta_arr[0] = e_eta[i]
    elec_charge_arr[0] = e_charge[i]

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
    
    for j in range(len(selected_jet_pt[i])):
        jet_pt_arr[j] = selected_jet_pt[i][j]
    for j in range(len(selected_jet_eta[i])):
        jet_eta_arr[j] = selected_jet_eta[i][j]
        ## do same for others
        jet_phi_arr[0] = jet_phi[i]
        jet_btag_arr[0] = jet_btag[i]
        
    for j in range(len(selected_genpart_pt[i])):
        genpart_pt_arr[j] = selected_genpart_pt[i][j]
    for j in range(len(selected_genpart_eta[i])):
        genpart_eta_arr[j] = selected_genpart_eta[i][j]
    for j in range(len(selected_genpart_phi[i])):
        genpart_phi_arr[j] = selected_genpart_phi[i][j]
    for j in range(len(selected_genpart_mass[i])):
        genpart_mass_arr[j] = selected_genpart_mass[i][j]
    for j in range(len(selected_genpart_pid[i])):
        genpart_pid_arr[j] = selected_genpart_pid[i][j]
    for j in range(len(selected_genpart_status[i])):
        genpart_status_arr[j] = selected_genpart_status[i][j]
    for j in range(len(selected_genpart_pt[i])):
        genpart_charge_arr[j] = selected_genpart_charge[i][j]

    genjet_eta_arr[0] = genjet_eta[i]
    genjet_mass_arr[0] = genjet_mass[i]
    genjet_phi_arr[0] = genjet_phi[i]
    genjet_pt_arr[0] = genjet_pt[i]

    llbar_deta_arr[0] = llbar_deta[i]
    llbar_dphi_arr[0] = llbar_dphi[i]
    bbbar_deta_arr[0] = bbbar_deta[i]
    bbbar_dphi_arr[0] = bbbar_dphi[i]

    tree.Fill()

outputfile.Write()
outputfile.Close()

## top quark reconstruction
## loook at the entanglements once we have quarks reconstructed

