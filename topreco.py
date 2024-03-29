import ROOT
import uproot
import numpy as np
from array import array
import matplotlib.pyplot as plt
import argparse
import math
from Top_reco_helpers import try_smear

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input')
parser.add_argument('-o', '--output', help='Output')
args = parser.parse_args()

#read in files
fileptr = uproot.open(args.input)
outputfile = ROOT.TFile(args.output, 'recreate')

genpart_pt = fileptr['CutTree']['selected_genpart_pt'].array()
genpart_eta = fileptr['CutTree']['selected_genpart_eta'].array()
genpart_phi = fileptr['CutTree']['selected_genpart_phi'].array()
genpart_mass = fileptr['CutTree']['selected_genpart_mass'].array()
genpart_pid = fileptr['CutTree']['selected_genpart_pid'].array()
genpart_status = fileptr['CutTree']['selected_genpart_status'].array()
genpart_charge = fileptr['CutTree']['selected_genpart_charge'].array()

met_pt = fileptr['CutTree']['met_pt_arr'].array()
met_phi = fileptr['CutTree']['met_phi_arr'].array()

elec_pt = fileptr['CutTree']['selected_elec_pt'].array()
elec_eta = fileptr['CutTree']['selected_elec_eta'].array()
elec_phi = fileptr['CutTree']['selected_elec_phi'].array()
elec_mass = fileptr['CutTree']['selected_elec_mass'].array()
elec_charge = fileptr['CutTree']['selected_elec_charge'].array()

muon_pt = fileptr['CutTree']['selected_mu_pt'].array()
muon_eta = fileptr['CutTree']['selected_mu_eta'].array()
muon_phi = fileptr['CutTree']['selected_mu_phi'].array()
muon_mass = fileptr['CutTree']['selected_mu_mass'].array()
muon_charge = fileptr['CutTree']['selected_mu_charge'].array()

jet_pt = fileptr['CutTree']['selected_jet_pt'].array()
jet_eta = fileptr['CutTree']['selected_jet_eta'].array()
jet_phi = fileptr['CutTree']['selected_jet_phi'].array()
jet_btag = fileptr['CutTree']['selected_jet_btag'].array()
jet_mass = fileptr['CutTree']['selected_jet_mass'].array()

weight = fileptr['CutTree']['selected_weight'].array()

#empty arrys
lep_pt = []
lep_eta = []
lep_phi = []
lep_mass = []

alep_pt = []
alep_eta = []
alep_phi = []
alep_mass = []

b_pt = []
b_eta = []
b_phi = []
b_mass = []

ab_pt = []
ab_eta = []
ab_phi = []
ab_mass = []

nu_pt = []
nu_eta = []
nu_phi = []

anu_pt = []
anu_eta = []
anu_phi = []

top_pt = []
top_eta = []
top_phi = []
top_rap = []

atop_pt = []
atop_eta = []
atop_phi = []
atop_rap = []

tt_mass = []
gen_tt_mass = []

gen_top_pt = []
gen_top_eta = []
gen_top_phi = []
gen_top_rap = []

gen_atop_pt = []
gen_atop_eta = []
gen_atop_phi = []
gen_atop_rap = []

final_array = [0] * len(elec_pt)

for eventidx in range(len(elec_pt)):

    # assigning data to 4vector
    # if e charge (of eventidx) is -1 and mu == 1 then e is lep and m is alep
    # if charges are flipped then the opposite is true

    lep4vector = ROOT.TLorentzVector()
    alep4vector = ROOT.TLorentzVector()

    if elec_charge[eventidx] == -1 and muon_charge[eventidx] == 1:

        lep4vector.SetPtEtaPhiM(elec_pt[eventidx], elec_eta[eventidx], elec_phi[eventidx],
                              elec_mass[eventidx])
        alep4vector.SetPtEtaPhiM(muon_pt[eventidx], muon_eta[eventidx],
                              muon_phi[eventidx],
                              muon_mass[eventidx])

    elif elec_charge[eventidx] == 1 and muon_charge[eventidx] == -1:

        alep4vector.SetPtEtaPhiM(elec_pt[eventidx], elec_eta[eventidx], elec_phi[eventidx],
                              elec_mass[eventidx])
        lep4vector.SetPtEtaPhiM(muon_pt[eventidx], muon_eta[eventidx],
                              muon_phi[eventidx],
                              muon_mass[eventidx])

    metx = met_pt[eventidx] * math.cos(met_phi[eventidx])
    mety = met_pt[eventidx] * math.sin(met_phi[eventidx])

    tt_mass_final = 0
    btag_counter = 0
    high_w = 0

    for i in range(len(jet_pt[eventidx])):
        for j in range(len(jet_pt[eventidx])):
            if i >= j:
                continue
            if jet_pt[eventidx][i] < 30:
                continue
            if jet_pt[eventidx][j] < 30:
                continue
            if abs(jet_eta[eventidx][i]) > 2.4:
                continue
            if abs(jet_eta[eventidx][j]) > 2.4:
                continue
            if jet_btag[eventidx][i] == 0 and jet_btag[eventidx][j] == 0:
                continue

            jet14vector = ROOT.TLorentzVector()
            jet24vector = ROOT.TLorentzVector()

            jet14vector.SetPtEtaPhiM(jet_pt[eventidx][i], jet_eta[eventidx][i], jet_phi[eventidx][i],
                                    jet_mass[eventidx][i])
            jet24vector.SetPtEtaPhiM(jet_pt[eventidx][j], jet_eta[eventidx][j],
                                     jet_phi[eventidx][j],
                                     jet_mass[eventidx][j])

            if jet_btag[eventidx][i] != 0 and jet_btag[eventidx][j] != 0:
                tt_mass1, topp41, atopp41, nup41, anup41, sw1 = try_smear(jet14vector, jet24vector, alep4vector, lep4vector, metx, mety, eventidx)
                tt_mass2, topp42, atopp42, nup42, anup42, sw2 = try_smear(jet24vector, jet14vector, alep4vector, lep4vector, metx, mety, eventidx)

                if tt_mass1 == -999 and tt_mass2 == -999:
                    continue

                btag_counter = 2

                if (tt_mass2 == -999) or ((tt_mass1 != -999 and tt_mass2 != -999) and ( sw2 <= sw1)):
                    tt_mass_final = tt_mass1
                    topp4_final = topp41
                    atopp4_final = atopp41
                    nup4_final = nup41
                    anup4_final = anup41
                    bp4_final = jet14vector
                    abp4_final = jet24vector
                if (tt_mass1 == -999) or ((tt_mass1 != -999 and tt_mass2 != -999) and ( sw1 < sw2)):
                    tt_mass_final = tt_mass2
                    topp4_final = topp42
                    atopp4_final = atopp42
                    nup4_final = nup42
                    anup4_final = anup42
                    bp4_final = jet24vector
                    abp4_final = jet14vector

                continue

            if jet_btag[eventidx][i] + jet_btag[eventidx][j] == 1:
                tt_mass1, topp41, atopp41, nup41, anup41, sw1 = try_smear(jet14vector, jet24vector, alep4vector, lep4vector, metx, mety, eventidx)
                tt_mass2, topp42, atopp42, nup42, anup42, sw2 = try_smear(jet24vector, jet14vector, alep4vector, lep4vector, metx, mety, eventidx)

                if tt_mass1 == -999 and tt_mass2 == -999:
                    continue

                if (tt_mass2 == -999 and high_w <= sw1) or ((tt_mass1!= -999 and tt_mass2 != -999) and sw2 <= sw1 and high_w <= sw1):
                    tt_mass_final = tt_mass1
                    topp4_final = topp41
                    atopp4_final = atopp41
                    nup4_final = nup41
                    anup4_final = anup41
                    bp4_final = jet14vector
                    abp4_final = jet24vector
                if (tt_mass1 == -999 and high_w < sw2) or ((tt_mass1!= -999 and tt_mass2 != -999) and sw1 <= sw2 and high_w <= sw2):
                    tt_mass_final = tt_mass2
                    topp4_final = topp42
                    atopp4_final = atopp42
                    nup4_final = nup42
                    anup4_final = anup42
                    bp4_final = jet24vector
                    abp4_final = jet14vector

        if tt_mass_final == 0 :
            continue

        for k in range(len(genpart_pt[eventidx])):
            if genpart_pid[eventidx][k] == 6 and genpart_status[eventidx][k] == 62:
                gentop4vector = ROOT.TLorentzVector()
                gentop4vector.SetPtEtaPhiM(genpart_pt[eventidx][k], genpart_eta[eventidx][k], genpart_phi[eventidx][k], genpart_mass[eventidx][k])
            if genpart_pid[eventidx][k] == -6 and genpart_status[eventidx][k] == 62:
                genatop4vector = ROOT.TLorentzVector()
                genatop4vector.SetPtEtaPhiM(genpart_pt[eventidx][k], genpart_eta[eventidx][k], genpart_phi[eventidx][k], genpart_mass[eventidx][k])

        com4vector = genatop4vector + gentop4vector
        tt_mass.append(tt_mass_final)
        top_pt.append(topp4_final.Pt())
        top_eta.append(topp4_final.Eta())
        top_phi.append(topp4_final.Phi())
        top_rap.append(topp4_final.Rapidity())

        atop_pt.append(atopp4_final.Pt())
        atop_eta.append(atopp4_final.Eta())
        atop_phi.append(atopp4_final.Phi())
        atop_rap.append(atopp4_final.Rapidity())

        nu_pt.append(nup4_final.Pt())
        nu_eta.append(nup4_final.Eta())
        nu_phi.append(nup4_final.Phi())

        anu_pt.append(anup4_final.Pt())
        anu_eta.append(anup4_final.Eta())
        anu_phi.append(anup4_final.Phi())

        lep_pt.append(lep4vector.Pt())
        lep_eta.append(lep4vector.Eta())
        lep_phi.append(lep4vector.Phi())
        lep_mass.append(lep4vector.M())

        alep_pt.append(alep4vector.Pt())
        alep_eta.append(alep4vector.Eta())
        alep_phi.append(alep4vector.Phi())
        alep_mass.append(alep4vector.M())

        b_pt.append(bp4_final.Pt())
        b_phi.append(bp4_final.Phi())
        b_eta.append(bp4_final.Eta())
        b_mass.append(bp4_final.M())

        ab_pt.append(abp4_final.Pt())
        ab_eta.append(abp4_final.Eta())
        ab_phi.append(abp4_final.Phi())
        ab_mass.append(abp4_final.M())

        gen_tt_mass.append(com4vector.M())
        gen_top_pt.append(gentop4vector.Pt())
        gen_top_eta.append(gentop4vector.Eta())
        gen_top_phi.append(gentop4vector.Phi())
        gen_top_rap.append(gentop4vector.Rapidity())

        gen_atop_pt.append(genatop4vector.Pt())
        gen_atop_eta.append(genatop4vector.Eta())
        gen_atop_phi.append(genatop4vector.Phi())
        gen_atop_rap.append(genatop4vector.Rapidity())

        final_array[eventidx] = 1


tree = ROOT.TTree("TopReco", "TopReco")

tt_mass_arr = array('f', [0.])
top_pt_arr = array('f', [0.])
top_eta_arr = array('f', [0.])
top_phi_arr = array('f', [0.])
top_rap_arr = array('f', [0.])

atop_pt_arr = array('f', [0.])
atop_eta_arr = array('f', [0.])
atop_phi_arr = array('f', [0.])
atop_rap_arr = array('f', [0.])

nu_pt_arr = array('f', [0.])
nu_eta_arr = array('f', [0.])
nu_phi_arr = array('f', [0.])

anu_pt_arr = array('f', [0.])
anu_eta_arr = array('f', [0.])
anu_phi_arr = array('f', [0.])

lep_pt_arr = array('f', [0.])
lep_eta_arr = array('f', [0.])
lep_phi_arr = array('f', [0.])
lep_mass_arr = array('f', [0.])

alep_pt_arr = array('f', [0.])
alep_eta_arr = array('f', [0.])
alep_phi_arr = array('f', [0.])
alep_mass_arr = array('f', [0.])

b_pt_arr = array('f', [0.])
b_phi_arr = array('f', [0.])
b_eta_arr = array('f', [0.])
b_mass_arr = array('f', [0.])

ab_pt_arr = array('f', [0.])
ab_eta_arr = array('f', [0.])
ab_phi_arr = array('f', [0.])
ab_mass_arr = array('f', [0.])

gen_tt_mass_arr = array('f', [0.])
gen_top_pt_arr = array('f', [0.])
gen_top_eta_arr = array('f', [0.])
gen_top_phi_arr = array('f', [0.])
gen_top_rap_arr = array('f', [0.])

gen_atop_pt_arr = array('f', [0.])
gen_atop_eta_arr = array('f', [0.])
gen_atop_phi_arr = array('f', [0.])
gen_atop_rap_arr = array('f', [0.])

tree.Branch("top_pt", top_pt_arr, "top_pt/F")
tree.Branch("tt_mass", tt_mass_arr, "tt_mass/F")
tree.Branch("top_eta", top_eta_arr, "top_eta/F")
tree.Branch("top_phi", top_phi_arr, "top_phi/F")
tree.Branch("top_rap", top_rap_arr, "top_rap/F")

tree.Branch("atop_pt", atop_pt_arr, "atop_pt/F")
tree.Branch("atop_eta", atop_eta_arr, "atop_eta/F")
tree.Branch("atop_phi", atop_phi_arr, "atop_phi/F")
tree.Branch("atop_rap", atop_rap_arr, "atop_rap/F")

tree.Branch("nu_pt", nu_pt_arr, "nu_pt/F")
tree.Branch("nu_eta", nu_eta_arr, "nu_eta/F")
tree.Branch("nu_phi", nu_phi_arr, "nu_phi/F")

tree.Branch("anu_phi", anu_phi_arr, "anu_phi/F")
tree.Branch("anu_pt", anu_pt_arr, "anu_pt/F")
tree.Branch("anu_eta", anu_eta_arr, "anu_eta/F")

tree.Branch("lep_eta", lep_eta_arr, "lep_eta/F")
tree.Branch("lep_pt", lep_pt_arr, "lep_pt/F")
tree.Branch("lep_phi", lep_phi_arr, "lep_phi/F")
tree.Branch("lep_mass", lep_mass_arr, "lep_mass/F")

tree.Branch("alep_mass", alep_mass_arr, "alep_mass/F")
tree.Branch("alep_pt", alep_pt_arr, "alep_pt/F")
tree.Branch("alep_eta", alep_eta_arr, "alep_eta/F")
tree.Branch("alep_phi", alep_phi_arr, "alep_phi/F")

tree.Branch("b_pt", b_pt_arr, "b_pt/F")
tree.Branch("b_eta", b_eta_arr, "b_eta/F")
tree.Branch("b_phi", b_phi_arr, "b_phi/F")
tree.Branch("b_mass", b_mass_arr, "b_mass/F")

tree.Branch("ab_pt", ab_pt_arr, "ab_pt/F")
tree.Branch("ab_eta", ab_eta_arr, "ab_eta/F")
tree.Branch("ab_phi", ab_phi_arr, "ab_phi/F")
tree.Branch("ab_mass", ab_mass_arr, "ab_mass/F")

tree.Branch("gen_tt_mass", gen_tt_mass_arr, "gen_tt_mass/F")
tree.Branch("gen_top_pt", gen_top_pt_arr, "gen_top_pt/F")
tree.Branch("gen_top_eta", gen_top_eta_arr, "gen_top_eta/F")
tree.Branch("gen_top_phi", gen_top_phi_arr, "gen_top_phi/F")
tree.Branch("gen_top_rap", gen_top_rap_arr, "gen_top_rap/F")

tree.Branch("gen_atop_pt", gen_atop_pt_arr, "gen_atop_pt/F")
tree.Branch("gen_atop_eta", gen_atop_eta_arr, "gen_atop_eta/F")
tree.Branch("gen_atop_phi", gen_atop_phi_arr, "gen_atop_phi/F")
tree.Branch("gen_atop_rap", gen_atop_rap_arr, "gen_atop_rap/F")

for i in range(len(top_pt)):
    tt_mass_arr[0] = tt_mass[i]
    top_pt_arr[0] = top_pt[i]
    top_eta_arr[0] = top_eta[i]
    top_phi_arr[0] = top_phi[i]
    top_rap_arr[0] = top_rap[i]

    atop_pt_arr[0] = atop_pt[i]
    atop_eta_arr[0] = atop_eta[i]
    atop_phi_arr[0] = atop_phi[i]
    atop_rap_arr[0] = atop_rap[i]

    nu_pt_arr[0] = nu_pt[i]
    nu_eta_arr[0] = nu_eta[i]
    nu_phi_arr[0] = nu_phi[i]

    anu_pt_arr[0] = anu_pt[i]
    anu_eta_arr[0] = anu_eta[i]
    anu_phi_arr[0] = anu_phi[i]

    lep_pt_arr[0] = lep_pt[i]
    lep_eta_arr[0] = lep_eta[i]
    lep_phi_arr[0] = lep_phi[i]
    lep_mass_arr[0] = lep_mass[i]

    alep_pt_arr[0] = lep_pt[i]
    alep_eta_arr[0] = lep_eta[i]
    alep_phi_arr[0] = lep_phi[i]
    alep_mass_arr[0] = lep_mass[i]

    b_pt_arr[0] = b_pt[i]
    b_phi_arr[0] = b_phi[i]
    b_eta_arr[0] = b_eta[i]
    b_mass_arr[0] = b_mass[i]

    ab_pt_arr[0] = ab_pt[i]
    ab_eta_arr[0] = ab_eta[i]
    ab_phi_arr[0] = ab_phi[i]
    ab_mass_arr[0] = ab_mass[i]

    gen_tt_mass_arr[0] = gen_tt_mass[i]
    gen_top_pt_arr[0] = gen_top_pt[i]
    gen_top_eta_arr[0] = gen_top_eta[i]
    gen_top_phi_arr[0] = gen_top_phi[i]
    gen_top_rap_arr[0] = gen_top_rap[i]

    gen_atop_pt_arr[0] = gen_atop_pt[i]
    gen_atop_eta_arr[0] = gen_atop_eta[i]
    gen_atop_phi_arr[0] = gen_atop_phi[i]
    gen_atop_rap_arr[0] = gen_atop_rap[i]

    tree.Fill()

    outputfile.Write()
    outputfile.Close()














