import uproot
import numpy as np
from array import array
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input')
parser.add_argument('-o', '--output', help='Output')
args = parser.parse_args()

#read in files
fileptr = uproot.open(args.input)

genpart_pt = fileptr['CutTree']['genpart_pt'].array()
genpart_eta = fileptr['CutTree']['genpart_eta'].array()
genpart_phi = fileptr['CutTree']['genpart_phi'].array()
genpart_mass = fileptr['CutTree']['genpart_mass'].array()
genpart_pid = fileptr['CutTree']['genpart_pid'].array()
genpart_status = fileptr['CutTree']['genpart_status'].array()
genpart_charge = fileptr['CutTree']['genpart_charge'].array()

met_pt = fileptr['CutTree']['met_pt'].array()
met_phi = fileptr['CutTree']['met_phi'].array()

elec_pt = fileptr['CutTree']['elec_pt'].array()
elec_eta = fileptr['CutTree']['elec_eta'].array()
elec_phi = fileptr['CutTree']['elec_phi'].array()
elec_mass = fileptr['CutTree']['elec_mass'].array()
elec_charge = fileptr['CutTree']['elec_charge'].array()

muon_pt = fileptr['CutTree']['muon_pt'].array()
muon_eta = fileptr['CutTree']['muon_eta'].array()
muon_phi = fileptr['CutTree']['muon_phi'].array()
muon_mass = fileptr['CutTree']['muon_mass'].array()
muon_charge = fileptr['CutTree']['muon_charge'].array()

jet_pt = fileptr['CutTree']['jet_pt'].array()
jet_eta = fileptr['CutTree']['jet_eta'].array()
jet_phi = fileptr['CutTree']['jet_phi'].array()
jet_btag = fileptr['CutTree']['jet_btag'].array()
jet_mass = fileptr['CutTree']['jet_mass'].array()

weight = fileptr['CutTree']['weight'].array()

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

    lep4vector = ROOT.TLorentzVector()
    alep4vector = ROOT.TLorentzVector()

    # assigning data to 4vector
    # if e charge (of eventidx) is -1 and mu == 1 then e is lep and m is alep
    #if charges are flipped then the opposite is true
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

    metx = met_pt[eventidx] * cos(met_phi[eventidx])
    mety = met_pt[eventidx] * sin(met_phi[eventidx])

    tt_mass_final = 0
    btag_counter = 0
    high_w = 0

    for i in range(len(jet_pt[eventidx])):
        for j in range(len(jet_pt[eventidx])):
            if i >= j:
                continue
            if jet_pt[i] < 30:
                continue
            if jet_pt[j] < 30:
                continue
            if abs(jet_eta[i]) > 2.4:
                continue
            if abs(jet_eta[j]) > 2.4:
                continue
            if jet_btag[i] == 0 and jet_btag[j] == 0:
                continue

            jet14vector = ROOT.TLorentzVector()
            jet24vector = ROOT.TLorentzVector()

            jet14vector.SetPtEtaPhiM(jet_pt[eventidx][i], jet_eta[eventidx][i], jet_phi[eventidx][i],
                                    jet_mass[eventidx][i])
            jet24vector.SetPtEtaPhiM(jet_pt[eventidx][j], jet_eta[eventidx][j],
                                     jet_phi[eventidx][j],
                                     jet_mass[eventidx][j])

            if jet_btag[i] != 0 and jet_btag[j] != 0:
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
                genatop4vector.SetPtEtaPhiM(genpart_pt[eventidx][k], genpart_eta[eventidx][k], genpart_phi[eventidx][k], [eventidx][k])

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



