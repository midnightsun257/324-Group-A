
import ROOT
import uproot
import argparse
import numpy as np
from array import array
import awkward as ak
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input')
parser.add_argument('-o', '--output', help='Output')
args = parser.parse_args()

fileptr = uproot.open(args.input)
genpart_pt = fileptr['CutTree']['selected_genpart_pt'].array()
genpart_eta = fileptr['CutTree']['selected_genpart_eta'].array()
genpart_phi = fileptr['CutTree']['selected_genpart_phi'].array()
genpart_mass = fileptr['CutTree']['selected_genpart_mass'].array()
genpart_pid = fileptr['CutTree']['selected_genpart_pid'].array()
genpart_status = fileptr['CutTree']['selected_genpart_status'].array()
genpart_charge = fileptr['CutTree']['selected_genpart_charge'].array()

gen_lep_pt=[]
gen_lep_eta=[]
gen_lep_phi=[]
gen_lep_mass=[]

gen_alep_pt=[]
gen_alep_eta=[]
gen_alep_phi=[]
gen_alep_mass=[]

gen_top_pt=[]
gen_top_eta=[]
gen_top_phi=[]
gen_top_mass=[]

gen_atop_pt=[]
gen_atop_eta=[]
gen_atop_phi=[]
gen_atop_mass=[]

for i in range(len(genpart_pt)):
    tempnum_lep=0
    tempnum_alep=0
    tempnum_b=0
    tempnum_ab=0
    lep_idx=0
    alep_idx=0
    for j in range(len(genpart_pt[i])-1):
        if (genpart_pid[i][j]==11 or genpart_pid[i][j]==13 or genpart_pid[i][j]==15) and (genpart_pid[i][j+1]==-12 or genpart_pid[i][j+1]==-14 or genpart_pid[i][j+1]==-16) and tempnum_lep==0:
            lep_idx=j
            tempnum_lep+=1
        if (genpart_pid[i][j]==-11 or genpart_pid[i][j]==-13 or genpart_pid[i][j]==-15) and (genpart_pid[i][j+1]==12 or genpart_pid[i][j+1]==14 or genpart_pid[i][j+1]==16) and tempnum_alep==0:
            alep_idx=j
            tempnum_alep+=1
        #if tempnum_lep==1 and tempnum_alep==1:
         #   break

    if (tempnum_alep ==0 or tempnum_lep==0) and (lep_idx==0 or alep_idx==0):
        continue

    gen_lep_pt.append(genpart_pt[i][lep_idx])
    gen_lep_eta.append(genpart_eta[i][lep_idx])
    gen_lep_phi.append(genpart_phi[i][lep_idx])
    gen_lep_mass.append(genpart_mass[i][lep_idx])
    gen_alep_pt.append(genpart_pt[i][alep_idx])
    gen_alep_eta.append(genpart_eta[i][alep_idx])
    gen_alep_phi.append(genpart_phi[i][alep_idx])
    gen_alep_mass.append(genpart_mass[i][alep_idx])

    gen_top_pt.append(genpart_pt[i][2])
    gen_top_eta.append(genpart_eta[i][2])
    gen_top_phi.append(genpart_phi[i][2])
    gen_top_mass.append(genpart_mass[i][2])

    gen_atop_pt.append(genpart_pt[i][3])
    gen_atop_eta.append(genpart_eta[i][3])
    gen_atop_phi.append(genpart_phi[i][3])
    gen_atop_mass.append(genpart_mass[i][3])


gen_lep_pt_arr = array('f', [0.])
gen_lep_eta_arr = array('f', [0.])
gen_lep_phi_arr = array('f', [0.])
gen_lep_mass_arr = array('f', [0.])
gen_alep_pt_arr = array('f', [0.])
gen_alep_eta_arr = array('f', [0.])
gen_alep_phi_arr = array('f', [0.])
gen_alep_mass_arr = array('f', [0.])

gen_top_pt_arr = array('f', [0.])
gen_top_eta_arr = array('f', [0.])
gen_top_phi_arr = array('f', [0.])
gen_top_mass_arr = array('f', [0.])
gen_atop_pt_arr = array('f', [0.])
gen_atop_eta_arr = array('f', [0.])
gen_atop_phi_arr = array('f', [0.])
gen_atop_mass_arr = array('f', [0.])

genpart_outputfile = ROOT.TFile(args.output, 'recreate')
tree = ROOT.TTree("GenPartTree", "GenPartTree")

tree.Branch("gen_lep_pt", gen_lep_pt_arr, "gen_lep_pt/F")
tree.Branch("gen_lep_eta", gen_lep_eta_arr, "gen_lep_eta/F")
tree.Branch("gen_lep_phi", gen_lep_phi_arr, "gen_lep_phi/F")
tree.Branch("gen_lep_mass", gen_lep_mass_arr, "gen_lep_mass/F")
tree.Branch("gen_alep_pt", gen_alep_pt_arr, "gen_alep_pt/F")
tree.Branch("gen_alep_eta", gen_alep_eta_arr, "gen_alep_eta/F")
tree.Branch("gen_alep_phi", gen_alep_phi_arr, "gen_alep_phi/F")
tree.Branch("gen_alep_mass", gen_alep_mass_arr, "gen_alep_mass/F")

tree.Branch("gen_top_pt", gen_top_pt_arr, "gen_top_pt/F")
tree.Branch("gen_top_eta", gen_top_eta_arr, "gen_top_eta/F")
tree.Branch("gen_top_phi", gen_top_phi_arr, "gen_top_phi/F")
tree.Branch("gen_top_mass", gen_top_mass_arr, "gen_top_mass/F")
tree.Branch("gen_atop_pt", gen_atop_pt_arr, "gen_atop_pt/F")
tree.Branch("gen_atop_eta", gen_atop_eta_arr, "gen_atop_eta/F")
tree.Branch("gen_atop_phi", gen_atop_phi_arr, "gen_atop_phi/F")
tree.Branch("gen_atop_mass", gen_atop_mass_arr, "gen_atop_mass/F")

# tree fill
for i in range(len(gen_top_pt)):
    gen_top_pt_arr[0] = gen_top_pt[i]
    gen_top_phi_arr[0] = gen_top_phi[i]
    gen_top_eta_arr[0] = gen_top_eta[i]
    gen_top_mass_arr[0] = gen_top_mass[i]
    gen_atop_pt_arr[0] = gen_atop_pt[i]
    gen_atop_phi_arr[0] = gen_atop_phi[i]
    gen_atop_eta_arr[0] = gen_atop_eta[i]
    gen_atop_mass_arr[0] = gen_atop_mass[i]

    gen_lep_pt_arr[0] = gen_lep_pt[i]
    gen_lep_phi_arr[0] = gen_lep_phi[i]
    gen_lep_eta_arr[0] = gen_lep_eta[i]
    gen_lep_mass_arr[0] = gen_lep_mass[i]
    gen_alep_pt_arr[0] = gen_alep_pt[i]
    gen_alep_phi_arr[0] = gen_alep_phi[i]
    gen_alep_eta_arr[0] = gen_alep_eta[i]
    gen_alep_mass_arr[0] = gen_alep_mass[i]

    tree.Fill()

genpart_outputfile.Write()
genpart_outputfile.Close()
