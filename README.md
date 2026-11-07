# Top Quark Reconstruction

## Overview

This project focuses on reconstructing top–antitop ($t\bar{t}$) events produced in proton–proton collisions at the Large Hadron Collider (LHC). The analysis focuses on the dileptonic decay channel, in which both top quarks decay into a bottom quark, a charged lepton, and a neutrino:

$t\bar{t} \rightarrow b\ell^+\nu_\ell + \bar{b}\ell^-\bar{\nu}_\ell$

Since neutrinos cannot be directly detected, their momenta must be inferred using missing transverse momentum (MET), conservation laws, and known particle-mass constraints.

The project implements event selection and kinematic reconstruction of the top and antitop quarks using detector-level information. The reconstructed events provide a basis for studying top-quark spin correlations and quantum entanglement through the angular distributions of their decay products.

## Repository Structure

| File | Description |
|---|---|
| `letsmakecuts.py` | Selects events based on lepton, jet, and b-tagging requirements. |
| `topreco.py` | Reconstructs top and antitop four-momenta. |
| `Top_reco_helpers.py` | Handles detector smearing and reconstruction calculations. |
| `nuSolutions.py` | Solves for possible neutrino momenta using kinematic constraints. |
| `genpart.py` | Extracts generator-level particle information. |
| `graphing.py` | Plots particle kinematic distributions. |
| `scaling.py` | Prepares batch processing of multiple ROOT files. |
| `KinReco_input.root` | Contains histograms used for detector smearing and reconstruction. |

## Workflow

1. **Event selection:** Filter collision events to identify dileptonic \(t\bar{t}\) candidates.
2. **Kinematic reconstruction:** Reconstruct top and antitop four-momenta from leptons, jets, and missing transverse momentum.
3. **Neutrino reconstruction:** Apply W-boson and top-quark mass constraints to estimate invisible neutrino momenta.
4. **Validation:** Extract generator-level information for comparison with reconstructed quantities.
5. **Visualization:** Plot particle kinematics and inspect distributions.

## Requirements

- Python 3
- ROOT / PyROOT
- Uproot
- NumPy
- Matplotlib
- Awkward Array

## Input Data

The analysis uses ROOT files containing a `Delphes_Ntuples` tree, along with `KinReco_input.root` for reconstruction and detector-resolution information.

## Future Analysis

The reconstructed events provide a basis for studying top–antitop spin correlations and quantum entanglement through the angular distributions of their decay products.

**Note:** This repository focuses on event selection and kinematic reconstruction. The full quantum-entanglement analysis is a downstream step.
