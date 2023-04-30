import math
import ROOT
import uproot
import numpy as np
from nuSolutions import doubleNeutrinoSolutions

dataFile = ROOT.TFile("KinReco_input.root", "OPEN")

h_jetAngleRes = dataFile.Get("KinReco_d_angle_jet_step7")
h_lepAngleRes = dataFile.Get("KinReco_d_angle_lep_step7")
h_jetEres = dataFile.Get("KinReco_fE_jet_step7")
h_lepEres = dataFile.Get("KinReco_fE_lep_step7")
h_mbl_w = dataFile.Get("KinReco_mbl_true_step0")
h_wmass = dataFile.Get("KinReco_W_mass_step0")


def Convertnuto4vec(nu, i, j):
    neutrino = ROOT.TLorentzVector()
    Px = nu[i][j][0]
    Py = nu[i][j][1]
    Pz = nu[i][j][2]
    E = math.sqrt(Px ** 2 + Py ** 2 + Pz ** 2)
    neutrino.SetPxPyPzE(Px, Py, Pz, E)
    return neutrino


# Function to return event top reco weight calculated from the gen level m_lb distribution
def get_mlb_weight(lv_lepton, lv_jet):
    reco_m_lb = (lv_lepton + lv_jet).M()
    evt_weight = h_mbl_w.GetBinContent(h_mbl_w.FindBin(reco_m_lb))
    return evt_weight / 100000000


# Gets a mean solution from the smeared tops
# Takes as input the vector of top lorentzvectors, vector of weights and assumed top mass
# Similar to https://gitlab.cern.ch/TopAnalysis/Configuration/analysis/common/src/KinematicReconstruction_MeanSol.cc

def get_mean_sol(vlv_top, v_weight, top_mass):
    px_sum = 0
    py_sum = 0
    pz_sum = 0

    # Weighted sum (weights according to the m_lb generator distribution)
    for i in range(len(vlv_top)):
        px_sum = px_sum + v_weight[i] * vlv_top[i].Px()
        py_sum = py_sum + v_weight[i] * vlv_top[i].Py()
        pz_sum = pz_sum + v_weight[i] * vlv_top[i].Pz()

    # Mean top momentum
    px_mean = px_sum / sum(v_weight)
    py_mean = py_sum / sum(v_weight)
    pz_mean = pz_sum / sum(v_weight)
    en_mean = np.sqrt((px_mean ** 2 + py_mean ** 2 + pz_mean ** 2) + top_mass ** 2)  # -,-,-,+ metric

    top_mean = ROOT.TLorentzVector()
    top_mean.SetPx(px_mean)
    top_mean.SetPy(py_mean)
    top_mean.SetPz(pz_mean)
    top_mean.SetE(en_mean)

    return top_mean


# Implemented from https://gitlab.cern.ch/TopAnalysis/Configuration/analysis/common/src/KinematicReconstruction.cc

def angle_rot(alpha, e, vect_sm):  # alpha should be passed, vect_sm is the smeared 4 vector to returned by reference

    # if abs(vect_sm.Px()) < e  : vect_sm.setPx(0)
    # if abs(vect_sm.Py()) < e  : vect_sm.setPy(0)
    # if abs(vect_sm.Pz()) < e  : vect_sm.setPz(0)

    # alpha = np.random.normal(0, 1)                      # should be related to vect_sm angular resolution
    phi = 2 * np.pi * np.random.uniform(-1, 1)
    pz_1 = vect_sm.Vect().Mag() * np.cos(alpha)
    px_1 = vect_sm.Vect().Mag() * np.sin(alpha) * np.sin(phi)
    py_1 = vect_sm.Vect().Mag() * np.sin(alpha) * np.cos(phi)

    if (vect_sm.Py() != 0 or vect_sm.Pz() != 0):
        d = np.sqrt(vect_sm.Pz() ** 2 + vect_sm.Py() ** 2)
        p = vect_sm.Vect().Mag()

        x1 = d / p
        y1 = 0
        z1 = vect_sm.Px() / p

        x2 = -(vect_sm.Px() * vect_sm.Py()) / (d * p)
        y2 = vect_sm.Pz() / d
        z2 = vect_sm.Py() / p

        x3 = -(vect_sm.Px() * vect_sm.Pz()) / (d * p)
        y3 = - vect_sm.Py() / d
        z3 = vect_sm.Pz() / p

        vect_sm.SetPx(x1 * px_1 + y1 * py_1 + z1 * pz_1)
        vect_sm.SetPy(x2 * px_1 + y2 * py_1 + z2 * pz_1)
        vect_sm.SetPz(x3 * px_1 + y3 * py_1 + z3 * pz_1)
        vect_sm.SetE(vect_sm.E())

    if (vect_sm.Px() == 0 and vect_sm.Py() == 0 and vect_sm.Pz() == 0):
        vect_sm.SetPx(vect_sm.Px())
        vect_sm.SetPy(vect_sm.Py())
        vect_sm.SetPz(vect_sm.Pz())
        vect_sm.SetE(vect_sm.E())

    if (vect_sm.Px() != 0 and vect_sm.Py() == 0 and vect_sm.Pz() == 0):
        vect_sm.SetPx(pz_1)
        vect_sm.SetPy(px_1)
        vect_sm.SetPz(py_1)
        vect_sm.SetE(vect_sm.E())


# Implemented from https://gitlab.cern.ch/TopAnalysis/Configuration/analysis/common/src/KinematicReconstruction.cc

def try_smear(jet1, jet2, lep1, lep2, metx, mety, evt_idx):
    v_weights = []
    vlv_tops = []
    vlv_antitops = []
    vlv_nus = []
    vlv_nubars = []

    if (((lep1 + jet1).M() > 180) or ((jet2 + lep2).M() > 180)):
        zero_lv = ROOT.TLorentzVector()
        return -999, zero_lv, zero_lv, zero_lv, zero_lv, sum(v_weights)

    # Define met, not sure what Vx is
    met = ROOT.TVector3(metx, mety, 0)
    vX_reco = - jet1.Vect() - jet2.Vect() - lep1.Vect() - lep2.Vect() - met

    for i in range(100):

        met_sm = ROOT.TLorentzVector()
        jet1_sm = ROOT.TLorentzVector(jet1)
        jet2_sm = ROOT.TLorentzVector(jet2)
        lep1_sm = ROOT.TLorentzVector(lep1)
        lep2_sm = ROOT.TLorentzVector(lep2)

        metV3_sm = ROOT.TVector3()

        # Jet energy based smearing
        fB1 = h_jetEres.GetRandom()  # From Jet Energy Resolution
        xB1 = np.sqrt((fB1 ** 2 * jet1_sm.E() ** 2 - jet1_sm.M2()) / (jet1_sm.P() ** 2))

        jet1_sm.SetXYZT(jet1_sm.Px() * xB1, jet1_sm.Py() * xB1, jet1_sm.Pz() * xB1, jet1_sm.E() * fB1)
        angle_rot(h_jetAngleRes.GetRandom(), 0.001, jet1_sm)

        fB2 = h_jetEres.GetRandom()  # From Jet Energy Resolution

        xB2 = np.sqrt((fB2 ** 2 * jet2_sm.E() ** 2 - jet2_sm.M2()) / (jet2_sm.P() ** 2))

        jet2_sm.SetXYZT(jet2_sm.Px() * xB2, jet2_sm.Py() * xB2, jet2_sm.Pz() * xB2, jet2_sm.E() * fB2)
        angle_rot(h_jetAngleRes.GetRandom(), 0.001, jet2_sm)

        # Lepton energy based smearing
        fL1 = h_lepEres.GetRandom()  # From Lep Energy Resolution
        xL1 = np.sqrt((fL1 ** 2 * lep1_sm.E() ** 2 - lep1_sm.M2()) / (lep1_sm.P() ** 2))

        lep1_sm.SetXYZT(lep1_sm.Px() * xB1, lep1_sm.Py() * xB1, lep1_sm.Pz() * xB1, lep1_sm.E() * fB1)
        angle_rot(h_lepAngleRes.GetRandom(), 0.001, lep1_sm)

        fL2 = h_lepEres.GetRandom()  # From Lep Energy Resolution,
        xL2 = np.sqrt((fL2 ** 2 * lep2_sm.E() ** 2 - lep2_sm.M2()) / (lep2_sm.P() ** 2))

        lep2_sm.SetXYZT(lep2_sm.Px() * xB2, lep2_sm.Py() * xB2, lep2_sm.Pz() * xB2, lep2_sm.E() * fB2)
        angle_rot(h_lepAngleRes.GetRandom(), 0.001, lep2_sm)

        # Adjust MET so that it is still balanced
        metV3_sm = -jet1_sm.Vect() - jet2_sm.Vect() - lep1_sm.Vect() - lep2_sm.Vect() - vX_reco
        met_sm.SetXYZM(metV3_sm.Px(), metV3_sm.Py(), 0, 0)

        # Maybe see if a solution exists or not
        try:
            d = doubleNeutrinoSolutions(jet1_sm, jet2_sm, lep1_sm, lep2_sm, met_sm.Px(), met_sm.Py())
            if (len(d.nunu_s) > 0):
                nu1 = Convertnuto4vec(d.nunu_s, 0, 0)
                nubar1 = Convertnuto4vec(d.nunu_s, 0, 1)

                top_p1 = jet1_sm + lep1_sm + nu1
                atop_p1 = jet2_sm + lep2_sm + nubar1

                if (len(d.nunu_s) >= 2):
                    nu2 = Convertnuto4vec(d.nunu_s, 1, 0)
                    nubar2 = Convertnuto4vec(d.nunu_s, 1, 1)

                    top_p2 = jet1_sm + lep1_sm + nu2
                    atop_p2 = jet2_sm + lep2_sm + nubar2

                    if ((top_p2 + atop_p2).M() <= (top_p1 + atop_p1).M()):
                        nu_final = nu2
                        nubar_final = nubar2
                        top_final = jet1_sm + lep1_sm + nu2
                        atop_final = jet2_sm + lep2_sm + nubar2

                    else:
                        nu_final = nu1
                        nubar_final = nubar1
                        top_final = jet1_sm + lep1_sm + nu1
                        atop_final = jet2_sm + lep2_sm + nubar1

                else:
                    nu_final = nu1
                    nubar_final = nubar1
                    top_final = jet1_sm + lep1_sm + nu1
                    atop_final = jet2_sm + lep2_sm + nubar1

                evt_w = get_mlb_weight(jet1_sm, lep1_sm) * get_mlb_weight(jet2_sm, lep2_sm)

                v_weights.append(evt_w)
                vlv_nus.append(nu_final)
                vlv_nubars.append(nubar_final)
                vlv_tops.append(top_final)
                vlv_antitops.append(atop_final)

        except np.linalg.LinAlgError:
            pass

    if len(v_weights) == 0:
        zero_lv = ROOT.TLorentzVector()
        return -999, zero_lv, zero_lv, zero_lv, zero_lv, sum(v_weights)

    else:
        mean_top_sol = get_mean_sol(vlv_tops, v_weights, 173)
        mean_antitop_sol = get_mean_sol(vlv_antitops, v_weights, 173)
        mean_nu_sol = get_mean_sol(vlv_nus, v_weights, 0)
        mean_nubar_sol = get_mean_sol(vlv_nubars, v_weights, 0)
        ttbar_mass = (mean_top_sol + mean_antitop_sol).M()

        return ttbar_mass, mean_top_sol, mean_antitop_sol, mean_nu_sol, mean_nubar_sol, (
                    sum(v_weights) / len(v_weights))