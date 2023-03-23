
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
genpart_pt = fileptr['CutTree']['genpart_pt_arr'].array()
genpart_eta = fileptr['CutTree']['genpart_eta_arr'].array()
genpart_phi = fileptr['CutTree']['genpart_phi_arr'].array()
genpart_mass = fileptr['CutTree']['genpart_mass_arr'].array()
genpart_pid = fileptr['CutTree']['genpart_pid_arr'].array()
genpart_status = fileptr['CutTree']['genpart_status_arr'].array()
genpart_charge = fileptr['CutTree']['genpart_charge_arr'].array()

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
        if (genpart_pid[j]==11 or genpart_pid[j]==13 or genpart_pid[j]==15) and (genpart_pid[j+1]==-12 or genpart_pid[j+1]==-14 or genpart_pid[j+1]==-16) and tempnum_lep==0:
            lep_idx=j
            tempnum_lep+=1
            break
