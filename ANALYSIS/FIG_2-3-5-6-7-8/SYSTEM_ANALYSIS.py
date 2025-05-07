import shutil
import sys

import numpy as np
import scipy.special
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math
from scipy.stats import linregress
import seaborn as sns
from scipy import integrate
import pylab
import os
from IPython.display import display
import fnmatch
from ACF import ACF
from VSC import VSC
from tqdm import tqdm
from RDP import RDP
from DIFF import DIFF

class ANALYSIS():
    def __init__(self):
        pass

    def calc_diffusion(self, path, sm, folder, dt, T, iterations, slope_iterations, tolerance, boot_r2, boot_tolerance, seg_length):
        def read_files(sm, folder):
            msd_arr = []
            time_arr = []
            rg_arr = []
            for file in tqdm(os.listdir("{}/{}/DIFFUSIVITY".format(path,folder)), desc="Collecting Diffusion Files", unit="files"):
                if fnmatch.fnmatch(file, 'ProteinG3BP1_Diffusivity_{}*.csv'.format(sm)):
                    df_diffusion = pd.read_csv("{}/{}/DIFFUSIVITY/{}".format(path,folder, file))
                    msd_arr.append(list(df_diffusion["MSD (um)"][1:]))
                    time_arr = list(df_diffusion["Time (s)"][1:])
                    rg_arr.append(df_diffusion["Rg"].mean())
                elif sm == "DSM":
                    if "G3BP1" in file:
                        df_diffusion = pd.read_csv("{}/{}/DIFFUSIVITY/{}".format(path,folder, file))
                        msd_arr.append(list(df_diffusion["MSD (um)"][1:]))
                        time_arr = list(df_diffusion["Time (s)"][1:])
                        rg_arr.append(df_diffusion["Rg"].mean())
                elif sm == "NDSM":
                    if "G3BP1" in file:
                        df_diffusion = pd.read_csv("{}/{}/DIFFUSIVITY/{}".format(path,folder, file))
                        msd_arr.append(list(df_diffusion["MSD (um)"][1:]))
                        time_arr = list(df_diffusion["Time (s)"][1:])
                        rg_arr.append(df_diffusion["Rg"].mean())
            msd_array = np.array(msd_arr)
            time_array = np.array(time_arr)
            rg_array = np.array(rg_arr)
            return msd_array, time_array, rg_array

        msd_array, time_array, rg_array = read_files(sm, folder)
        diff = DIFF(path, msd_array, time_array, rg_array)
        D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme = diff.boot_strap(
            sm=sm, iterations=iterations, T=T, dt=dt, slope_iterations=slope_iterations,
            tolerance=tolerance, boot_r2=boot_r2, boot_tolerance=boot_tolerance, seg=seg_length)
        return D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme

    def calc_visc(self, path, folder, sm, r, segments, iterations, n_boot, dt_unit, n_point, n_tau, tmin, tmax):
        def read_files(sm, folder):
            Pxyz, time = [], []
            stress_file = "{}/{}/Stress_Tensor_{}.csv".format(path,folder, sm)
            with open(stress_file, "r") as file:
                print('\nPreparing Pressure Tensor Array')
                for line in file.readlines()[1:]:
                    ln = line.split(",")
                    t = float(ln[0])*20E-6
                    if int(t) <= tmax and int(t) >= tmin:
                        time.append(float(ln[0]) * 20)
                        Pxy = -float(ln[4])
                        Pxz = -float(ln[5])
                        Pyz = -float(ln[6])
                        Pxyz.append([Pxy, Pxz, Pyz])
                    else:
                        pass

            return np.array(Pxyz) * 101325

        Pxyz = read_files(sm, folder)
        print(len(Pxyz))
        vol_sys = 4 / 3 * math.pi * (r*10**-10) ** 3
        vsc = VSC(path, Pxyz)
        eta_raw, eta_raw_sem, eta_fit, eta_fit_sem, eta_theo, eta_theo_sem, acf_fits_total, amp_opts, tau_opts, y_log0s, dt_log = vsc.run_vsc(vol_sys=vol_sys,
                                                                                                      name=sm,
                                                                                                      segments=segments,
                                                                                                      iterations=iterations,
                                                                                                      n_boot=n_boot,
                                                                                                      dt_unit=dt_unit,
                                                                                                      n_point=n_point,
                                                                                                      n_tau=n_tau)
        return eta_raw, eta_raw_sem, eta_fit, eta_fit_sem, eta_theo, eta_theo_sem

    def plot_rdps(self, fitterSG, fitterProtein, fitterRNA, fitterSM, sm, path):

        sm_dict = {"sg_X": "SG",
                   "ndsm_dmso": "ND1",
                   "ndsm_valeric": "ND2",
                   "ndsm_ethylenediamine": "ND3",
                   "ndsm_propanedithiol": "ND4",
                   "ndsm_hexanediol": "ND5",
                   "ndsm_diethylaminopentane": "ND6",
                   "ndsm_aminoacridine": "ND7",
                   "ndsm_anthraquinone": "ND8",
                   "ndsm_acetylenapthacene": "ND9",
                   "ndsm_anacardic": "ND10",

                   "dsm_hydroxyquinoline": "D1",
                   "dsm_lipoamide": "D2",
                   "dsm_lipoicacid": "D3",
                   "dsm_dihydrolipoicacid": "D4",
                   "dsm_anisomycin": "D5",
                   "dsm_pararosaniline": "D6",
                   "dsm_pyrivinium": "D7",
                   "dsm_quinicrine": "D8",
                   "dsm_mitoxantrone": "D9",
                   "dsm_daunorubicin": "D10",
                   "DSM": "DSM",
                   "NDSM": "NDSM"
                   }



        col_pall1 = sns.color_palette("rocket", n_colors=3)
        col_pall2 = sns.color_palette("Blues", n_colors=11)

        afont = {'fontname': 'Arial'}

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        fig, ax1 = plt.subplots(figsize=(3.1, 3.1))

        ax1.plot(fitterSG.fit_x, fitterSG.fit_rho, color=col_pall2[5], label='SG', linewidth=4, zorder=1)
        sns.scatterplot(ax=ax1, x=fitterSG.distances, y=fitterSG.densities, color=col_pall2[5], legend=False, s=40,
                        edgecolor="k", linewidth=1, zorder=3)
        ax1.errorbar(x=fitterSG.distances, y=fitterSG.densities, yerr=fitterSG.errors, fmt=".", color=col_pall2[5],
                     zorder=2)

        ax1.plot(fitterProtein.fit_x, fitterProtein.fit_rho, color=col_pall2[10], label='Protein', linewidth=4,
                 zorder=1)
        sns.scatterplot(ax=ax1, x=fitterProtein.distances, y=fitterProtein.densities, color=col_pall2[10],
                        legend=False,
                        s=40, edgecolor="k", linewidth=1, zorder=3)
        ax1.errorbar(fitterProtein.distances, fitterProtein.densities, yerr=fitterSG.errors, fmt=".",
                     color=col_pall2[10], zorder=2)

        ax1.plot(fitterRNA.fit_x, fitterRNA.fit_rho, color=col_pall2[2], label='RNA', linewidth=4, zorder=2)
        sns.scatterplot(ax=ax1, x=fitterRNA.distances, y=fitterRNA.densities, color=col_pall2[2], legend=False,
                        s=40,
                        edgecolor="k", linewidth=1, zorder=3)
        ax1.errorbar(fitterRNA.distances, fitterRNA.densities, yerr=fitterRNA.errors, fmt=".", color=col_pall2[2],
                     zorder=2)

        if fitterSM is not None:
            ax2 = ax1.twinx()
            if "ND" not in sm_dict[sm]:
                col = col_pall1[0]
            else:
                col = col_pall1[2]

            sns.scatterplot(ax=ax2, x=fitterSM.distances, y=fitterSM.densities, color=col, legend=False, s=40,
                            edgecolor="k", linewidth=1, zorder=3)
            ax2.errorbar(fitterSM.distances, fitterSM.densities, yerr=fitterSM.errors, fmt=".", color=col, zorder=2)
            sns.lineplot(ax=ax2, x=fitterSM.fit_x, y=fitterSM.fit_rho, color=col, label=sm_dict[sm], linewidth=4, zorder=1)
            ax2.get_legend().remove()
            ax2.tick_params(left=False, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax2.set_ylim(0.0, 0.2)
            #ax2.set_yticks(np.arange(0.0, 0.2, 0.02))
            ax1.tick_params(left=True, right=False, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)

        else:
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)

        # ax2.tick_params(left=True, right=True, top=True, bottom=False, labelbottom=True, direction='in', length=4, width=2)

        # ax1.set_xlabel('Radial Distance ($\AA$)')
        # ax1.set_ylabel("$C_{SG}$ $(mg/mL)$")
        ax1.set_xlim(200, 400)
        ax1.set_ylim(0, 50)
        #ax1.set_ylim(0, int(math.ceil(fitterSG.c_dense_fit / 100.0) * 100))

        leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
        leg.get_frame().set_alpha(0)
        plt.savefig("{}/IMAGES/{}_RDP.png".format(path, sm), format="png", dpi=400)
        return fig


    # Plot SG Residue HeatMap
    def sg_residue_heat_map(self, name, contact_array, file, path, folder):

        if "Difference" in file:
            cmap = "coolwarm"
        else:
            cmap = "Blues"

        #nBio = np.loadtxt("{}/{}/BioPolNum_{}.csv".format(path,folder,name), delimiter=",", dtype=float)
        #nBio = np.loadtxt("BioPolNum.csv", delimiter=",", dtype=float)
        #contact_array = np.divide(contact_array, (np.sum(contact_array)*nBio))

        contact_array[[4, 0]] = contact_array[[0, 4]]
        contact_array[[2, 1]] = contact_array[[1, 2]]
        contact_array[[1, 5]] = contact_array[[5, 1]]
        contact_array[[3, 4]] = contact_array[[4, 3]]
        contact_array[[4, 6]] = contact_array[[6, 4]]
        contact_array[[5, 6]] = contact_array[[6, 5]]

        contact_array[:, [4, 0]] = contact_array[:, [0, 4]]
        contact_array[:, [2, 1]] = contact_array[:, [1, 2]]
        contact_array[:, [1, 5]] = contact_array[:, [5, 1]]
        contact_array[:, [3, 4]] = contact_array[:, [4, 3]]
        contact_array[:, [4, 6]] = contact_array[:, [6, 4]]
        contact_array[:, [5, 6]] = contact_array[:, [6, 5]]

        afont = {'fontname': 'Arial'}

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
        plt.rc('legend', fontsize=10)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)
        file_name = ""

        x_res_list = ["TDP43", "FUS", "TIA1", "G3BP1", "RNA", "PABP1", "TTP"]
        y_res_list = x_res_list
        plt.figure(figsize=(3.8, 3.0), tight_layout=True)
        ax = sns.heatmap(contact_array, xticklabels=x_res_list, yticklabels=y_res_list, cmap=cmap, square=True)
        cbar = ax.collections[0].colorbar
        cbar.formatter.set_powerlimits((0, 0))

        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)
        file_name = "{}/IMAGES/{}_{}.png".format(path, file, name)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        plt.savefig(file_name, format="png", dpi=400)

        # save the dataframe as a csv file
        np.savetxt("{}/RESULTS/{}_{}.csv".format(path, file, name), contact_array, delimiter=",")




    # Plot SG Acid HeatMap
    def sg_acid_heat_map(self, name, contact_array, file, path):

        if "Difference" in file:
            cmap = "coolwarm"
        else:
            cmap = "Blues"

        #nBonds = np.loadtxt("BondNum.csv", delimiter=",", dtype=float)

        swap_contact_array = (contact_array)

        swap_contact_array[:, [4, 0]] = swap_contact_array[:, [0, 4]]
        swap_contact_array[:, [15, 1]] = swap_contact_array[:, [1, 15]]
        swap_contact_array[:, [2, 2]] = swap_contact_array[:, [2, 2]]
        swap_contact_array[:, [6, 3]] = swap_contact_array[:, [3, 6]]
        swap_contact_array[:, [16, 4]] = swap_contact_array[:, [4, 16]]
        swap_contact_array[:, [7, 4]] = swap_contact_array[:, [4, 7]]
        swap_contact_array[:, [14, 5]] = swap_contact_array[:, [5, 14]]
        swap_contact_array[:, [11, 8]] = swap_contact_array[:, [8, 11]]
        swap_contact_array[:, [18, 9]] = swap_contact_array[:, [9, 18]]
        swap_contact_array[:, [15, 10]] = swap_contact_array[:, [10, 15]]
        swap_contact_array[:, [17, 11]] = swap_contact_array[:, [11, 17]]
        swap_contact_array[:, [14, 12]] = swap_contact_array[:, [12, 14]]
        swap_contact_array[:, [18, 13]] = swap_contact_array[:, [13, 18]]
        swap_contact_array[:, [19, 14]] = swap_contact_array[:, [14, 19]]
        swap_contact_array[:, [17, 18]] = swap_contact_array[:, [18, 17]]

        swap_contact_array[[4, 0]] = swap_contact_array[[0, 4]]
        swap_contact_array[[15, 1]] = swap_contact_array[[1, 15]]
        swap_contact_array[[2, 2]] = swap_contact_array[[2, 2]]
        swap_contact_array[[6, 3]] = swap_contact_array[[3, 6]]
        swap_contact_array[[16, 4]] = swap_contact_array[[4, 16]]
        swap_contact_array[[7, 4]] = swap_contact_array[[4, 7]]
        swap_contact_array[[14, 5]] = swap_contact_array[[5, 14]]
        swap_contact_array[[11, 8]] = swap_contact_array[[8, 11]]
        swap_contact_array[[18, 9]] = swap_contact_array[[9, 18]]
        swap_contact_array[[15, 10]] = swap_contact_array[[10, 15]]
        swap_contact_array[[17, 11]] = swap_contact_array[[11, 17]]
        swap_contact_array[[14, 12]] = swap_contact_array[[12, 14]]
        swap_contact_array[[18, 13]] = swap_contact_array[[13, 18]]
        swap_contact_array[[19, 14]] = swap_contact_array[[14, 19]]
        swap_contact_array[[17, 18]] = swap_contact_array[[18, 17]]

        #nAcids = np.loadtxt("AcidPolNumJAY.csv", delimiter=",",  dtype=float)
        #contact_array = np.divide(swap_contact_array, (nAcids*np.sum(contact_array)))
        #contact_array = np.divide(swap_contact_array, 2)

        acid_list = [
            'ARG',
            'HIS',
            'LYS',
            'ASP',
            'GLU',
            'SER',
            'THR',
            'ASN',
            'GLN',
            'CYS',
            'GLY',
            'PRO',
            'ALA',
            'VAL',
            'ILE',
            'LEU',
            'MET',
            'PHE',
            'TYR',
            'TRP',
            'A',
            'U',
            'C',
            'G'
        ]

        afont = {'fontname': 'Arial'}
        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=6)  # fontsize of the tick labels
        plt.rc('legend', fontsize=10)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        plt.figure(figsize=(4.0, 3.2), tight_layout=True)
        ax = sns.heatmap(contact_array, xticklabels=acid_list, yticklabels=acid_list, cmap=cmap, square=True)
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=10)
        #cbar.formatter.set_powerlimits((0, 0))

        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)

        file_name = "{}/IMAGES/{}_{}.png".format(path, file, name)

        plt.savefig(file_name, format="png", dpi=400)
        df_acid = pd.DataFrame(contact_array)

        # save the dataframe as a csv file
        np.savetxt("{}/RESULTS/{}_{}.csv".format(path, file, name), contact_array, delimiter=",")

    def sm_residue_heat_map(self, name, contact_array, sm_list, path):

        contact_array = np.divide(contact_array, np.sum(contact_array))

        swap_contact_array = contact_array
        swap_contact_array[0] = contact_array[4]
        swap_contact_array[1] = contact_array[2]
        swap_contact_array[1] = contact_array[5]
        swap_contact_array[3] = contact_array[4]
        swap_contact_array[4] = contact_array[6]
        swap_contact_array[5] = contact_array[6]
        swap_contact_array[2] = contact_array[5]

        afont = {'fontname': 'Arial'}

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
        plt.rc('legend', fontsize=10)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)
        file_name = ""

        x_res_list = sm_list
        y_res_list = ["TDP43", "TTP", "TIA1", "PABP1", "G3BP1", "FUS", "RNA"]
        plt.figure(figsize=(3.8, 3.0), tight_layout=True)
        ax = sns.heatmap(contact_array, xticklabels=x_res_list, yticklabels=y_res_list, cmap="Blues", square=True)
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=10)
        cbar.formatter.set_powerlimits((0, 0))

        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)

        file_name = "{}/IMAGES/{}_SM_Residue_HeatMap.png".format(path, name)
        plt.savefig(file_name, format="png", dpi=400)
        # save the dataframe as a csv file
        np.savetxt("{}/RESULTS/SM_Residue_Contacts_{}.csv".format(path, name), contact_array, delimiter=",")


    def sm_acid_heat_map(self, name, contact_array, sm_list, path):

        swap_contact_array = contact_array
        swap_contact_array[[4, 0]] = swap_contact_array[[0, 4]]
        swap_contact_array[[15, 1]] = swap_contact_array[[1, 15]]
        swap_contact_array[[2, 2]] = swap_contact_array[[2, 2]]
        swap_contact_array[[6, 3]] = swap_contact_array[[3, 6]]
        swap_contact_array[[16, 4]] = swap_contact_array[[4, 16]]
        swap_contact_array[[7, 4]] = swap_contact_array[[4, 7]]
        swap_contact_array[[14, 5]] = swap_contact_array[[5, 14]]
        swap_contact_array[[11, 8]] = swap_contact_array[[8, 11]]
        swap_contact_array[[18, 9]] = swap_contact_array[[9, 18]]
        swap_contact_array[[15, 10]] = swap_contact_array[[10, 15]]
        swap_contact_array[[17, 11]] = swap_contact_array[[11, 17]]
        swap_contact_array[[14, 12]] = swap_contact_array[[12, 14]]
        swap_contact_array[[18, 13]] = swap_contact_array[[13, 18]]
        swap_contact_array[[19, 14]] = swap_contact_array[[14, 19]]
        swap_contact_array[[17, 18]] = swap_contact_array[[18, 17]]

        contact_array = swap_contact_array

        acid_list = [
            'ARG',
            'HIS',
            'LYS',
            'ASP',
            'GLU',
            'SER',
            'THR',
            'ASN',
            'GLN',
            'CYS',
            'GLY',
            'PRO',
            'ALA',
            'VAL',
            'ILE',
            'LEU',
            'MET',
            'PHE',
            'TYR',
            'TRP',
            'A',
            'U',
            'C',
            'G'
        ]



        afont = {'fontname': 'Arial'}
        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the tick labels
        plt.rc('legend', fontsize=10)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        plt.figure(figsize=(4.0, 3.2), tight_layout=True)
        ax = sns.heatmap(contact_array, xticklabels=sm_list, yticklabels=acid_list, cmap="Blues", square=True)
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=10)
        cbar.formatter.set_powerlimits((0, 0))

        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)

        file_name = "{}/IMAGES/{}_SM_Acid_HeatMap.png".format(path, name)

        plt.savefig(file_name, format="png", dpi=400)
        df_acid = pd.DataFrame(contact_array)

        # save the dataframe as a csv file
        np.savetxt("{}/RESULTS/SM_Acid_Contacts_{}.csv".format(path, name), contact_array, delimiter=",")


if __name__ == '__main__':

    # Calculate Cluster, Phase, Dynamic Properties for a single SM System
    def sg_sm_analysis_full(path, folder, name, df_sg, mol_name, c, df_res_contact, df_acid_contact, df_res_count, df_acid_count, T, dt, tmin, tmax):
        print("ANALYZING SYSTEM: {}".format(name))
        analysis = ANALYSIS()
        stress_file = "{}/{}/Stress_Tensor_{}.csv".format(path, folder, name)
        diffusion_file = "{}/{}/G3BP1_Diffusivity_{}.csv".format(path, folder, name)
        lab = ''.join(name.split("_STRESS")[0])

        sm_dict = {"sg_X": "SG",
                   "ndsm_dmso": "ND1",
                   "ndsm_valeric": "ND2",
                   "ndsm_ethylenediamine": "ND3",
                   "ndsm_propanedithiol": "ND4",
                   "ndsm_hexanediol": "ND5",
                   "ndsm_diethylaminopentane": "ND6",
                   "ndsm_aminoacridine": "ND7",
                   "ndsm_anthraquinone": "ND8",
                   "ndsm_acetylenapthacene": "ND9",
                   "ndsm_anacardic": "ND10",

                   "dsm_hydroxyquinoline": "D1",
                   "dsm_lipoamide": "D2",
                   "dsm_lipoicacid": "D3",
                   "dsm_dihydrolipoicacid": "D4",
                   "dsm_anisomycin": "D5",
                   "dsm_pararosaniline": "D6",
                   "dsm_pyrivinium": "D7",
                   "dsm_quinicrine": "D8",
                   "dsm_mitoxantrone": "D9",
                   "dsm_daunorubicin": "D10",
                   "DSM": "DSM",
                   "NDSM": "NDSM"
                   }

        name_dict = {"sg_X": "SG",
                   "ndsm_dmso": "DMSO",
                   "ndsm_valeric": "Valeric Acid",
                   "ndsm_ethylenediamine": "Ethylenediamine",
                   "ndsm_propanedithiol": "Propanedithiol",
                   "ndsm_hexanediol": "Hexanediol",
                   "ndsm_diethylaminopentane": "Diethylaminopentane",
                   "ndsm_aminoacridine": "Aminoacridine",
                   "ndsm_anthraquinone": "Anthraquinone",
                   "ndsm_acetylenapthacene": "Acetylenapthacene",
                   "ndsm_anacardic": "Anacardic Acid",

                   "dsm_hydroxyquinoline": "Hydroxyquinoline",
                   "dsm_lipoamide": "Lipoamide",
                   "dsm_lipoicacid": "Lipoic Acid",
                   "dsm_dihydrolipoicacid": "Dihydrolipoic Acid",
                   "dsm_anisomycin": "Anisomycin",
                   "dsm_pararosaniline": "Pararosaniline",
                   "dsm_pyrivinium": "Pyrivinium",
                   "dsm_quinicrine": "Quinicrine",
                   "dsm_mitoxantrone": "Mitoxantrone",
                   "dsm_daunorubicin": "Daunorubicin",
                   "DSM": "DSM",
                   "NDSM": "NDSM"
                   }

        sm_mass_dict = {"sg_X": 0,
                        "ndsm_dmso": 78.12922,
                        "ndsm_valeric": 102.13350,
                        "ndsm_ethylenediamine": 104.15244,
                        "ndsm_propanedithiol": 108.21676,
                        "ndsm_hexanediol": 118.17638,
                        "ndsm_diethylaminopentane": 158.28774,
                        "ndsm_aminoacridine": 194.23610,
                        "ndsm_anthraquinone": 240.21536,
                        "ndsm_acetylenapthacene": 336.34452,
                        "ndsm_anacardic": 348.52712,

                        "dsm_hydroxyquinoline": 145.16089,
                        "dsm_lipoamide": 205.33365,
                        "dsm_lipoicacid": 206.31838,
                        "dsm_dihydrolipoicacid": 208.33432,
                        "dsm_anisomycin": 265.30973,
                        "dsm_pararosaniline": 287.36459,
                        "dsm_pyrivinium": 382.52926,
                        "dsm_quinicrine": 399.96460,
                        "dsm_mitoxantrone": 444.48836,
                        "dsm_daunorubicin": 527.52883,
                        "DSM": (145.16089+205.33365+206.31838+208.33432+265.30973+287.36459+382.52926+399.96460+444.48836+527.52883)/10,
                        "NDSM": (78.12922+102.13350+104.15244+108.21676+118.17638+158.28774+194.23610+240.21536+336.34452+348.52712)/10
                        }

        if name.split("_")[0] == "ndsm" or name=="NDSM":
            effect = "Non-Dissolving"
            d_bin = 0
        elif name.split("_")[0] == "dsm" or name=="DSM":
            effect = "Dissolving"
            d_bin = 1
        else:
            effect = "X"
            d_bin = 0

        name_sm = sm_dict[name]

        # Phase Analysis
        print("Stress Granule ANALYSIS")
        label = "SG"
        density_file_SG = "{}/{}/Density_Profile_SG_{}.csv".format(path, folder, lab)
        pca_file_SG = "{}/{}/PCA_SG_{}.csv".format(path, folder, lab)
        cluster_file_SG = "{}/{}/Cluster_SG_{}.csv".format(path, folder, lab)
        init = [250, 250, 200, 80]
        fitterSG = RDP(density_file_SG, pca_file_SG, cluster_file_SG, T, init, label)

        print("Protein ANALYSIS")
        label = "Protein"
        density_file_Protein = "{}/{}/Density_Profile_Protein_{}.csv".format(path, folder, lab)
        pca_file_Protein = "{}/{}/PCA_Protein_{}.csv".format(path, folder, lab)
        cluster_file_Protein = "{}/{}/Cluster_Protein_{}.csv".format(path, folder, lab)
        init = [150, 150, 200, 80]
        fitterProtein = RDP(density_file_Protein, pca_file_SG, cluster_file_Protein, T, init, label)

        print("RNA ANALYSIS")
        label = "RNA"
        density_file_RNA = "{}/{}/Density_Profile_RNA_{}.csv".format(path, folder, lab)
        pca_file_RNA = "{}/{}/PCA_RNA_{}.csv".format(path, folder, lab)
        cluster_file_RNA = "{}/{}/Cluster_RNA_{}.csv".format(path, folder, lab)
        init = [80, 80, 200, 80]
        fitterRNA = RDP(density_file_RNA, pca_file_SG, cluster_file_RNA, T, init, label)

        if effect != 'X':
            print("{} ANALYSIS".format(lab))
            label = "SM"
            density_file_SM = "{}/{}/Density_Profile_SM_{}.csv".format(path, folder, lab)
            init = [0, 0.2, 200, 50]
            fitterSM = RDP(density_file_SM, pca_file_SG, cluster_file_SG, T, init, label)
            print(fitterSM)
            c_dilute_sm = fitterSM.fit_rho[-1]
            c_dense_sm = fitterSM.fit_rho[0]

            c_dilute_sm_se = (fitterSM.c_dilute_fit_se)
            c_dense_sm_se = (fitterSM.c_dense_fit_se)

            if fitterSM.c_dilute_fit != 0:
                pc = np.abs(fitterSM.fit_rho[0] / fitterSM.fit_rho[-1])
                pc_se = np.sqrt((1 / c_dilute_sm) ** 2 * c_dilute_sm_se ** 2 + (
                        -c_dense_sm / c_dilute_sm ** 2) ** 2 * c_dilute_sm_se ** 2)
            else:
                pc = 1
                pc_se = 0

        else:
            fitterSM = None
            c_dilute_sm = 0
            c_dense_sm = 0
            c_dilute_sm_se = 0
            c_dense_sm_se = 0
            pc = 0
            pc_se = 0

        analysis.plot_rdps(fitterSG, fitterProtein, fitterRNA, fitterSM, name, path=path)

        c_dense_sg_fit = np.abs(fitterSG.c_dense_fit)
        c_dense_sg_fit_se = fitterSG.c_dense_fit_se
        c_dilute_sg_fit = np.abs(fitterSG.c_dilute_fit)
        c_dilute_sg_fit_se = fitterSG.c_dilute_fit_se
        c_dense_sg_calc = np.abs(fitterSG.c_dense_calc)
        c_dense_sg_calc_se = fitterSG.c_dense_calc_se
        c_dilute_sg_calc = np.abs(fitterSG.c_dilute_calc)
        c_dilute_sg_calc_se = fitterSG.c_dense_fit_se

        p_SG = np.abs(c_dense_sg_fit) / np.abs(c_dilute_sg_fit)
        sig_p_SG = np.sqrt((1 / c_dilute_sg_fit) ** 2 * c_dense_sg_fit_se ** 2 + (
                -c_dense_sg_fit / c_dilute_sg_fit ** 2) ** 2 * c_dilute_sg_fit_se ** 2)

        R_droplet = np.abs(fitterSG.fit_R)
        sig_R = np.sqrt(np.abs(fitterSG.se_R))
        W = np.abs(fitterSG.fit_W)
        sig_W = np.sqrt(np.abs(fitterSG.se_W))

        st1 = fitterSG.st_1
        st2 = fitterSG.st_2
        st1_se = np.sqrt(fitterSG.st_1_se)
        st2_se = np.sqrt(fitterSG.st_2_se)
        st = fitterSG.st
        st_se = np.sqrt(fitterSG.st_se)
        dG = (fitterSG.dG / 1000)
        se_dG = np.sqrt(fitterSG.se_dG) / 1000

        # Clustering Analysis
        df_analysis_cluster = pd.read_csv("{}/{}/Cluster_SG_{}.csv".format(path, folder, lab))

        phi_d = float(df_analysis_cluster["Mass of Largest Droplet (mg)"].iloc[0]) / float(
            df_analysis_cluster["Total Mass (mg)"].iloc[0])
        num_clus = float(df_analysis_cluster["Number of Droplets"].iloc[0])
        rg = float(df_analysis_cluster["Largest Droplet Radius of Gyration"].iloc[0])

        phi_d_std = np.sqrt((1 / float(df_analysis_cluster["Total Mass (mg)"].iloc[0])) ** 2 * float(
            df_analysis_cluster["Mass Largest SEM"].iloc[0]) ** 2)
        num_clus_std = float(df_analysis_cluster["NE SEM"].iloc[0])
        rg_std = float(df_analysis_cluster["RG SEM"].iloc[0])

        print("PHI: " + str(phi_d) + " ± " + str(phi_d_std))
        print("CLUS: " + str(num_clus) + " ± " + str(num_clus_std))
        print("RG: " + str(rg) + " ± " + str(rg_std))

        phi_R = R_droplet / rg
        phi_R_std = np.sqrt((1 / rg) ** 2 * sig_R + (-R_droplet / rg ** 2) ** 2 * rg_std)

        # Diffusion Analysis
        directory = "{}/{}/DIFFUSIVITY".format(path, folder)
        dt_diff = int(dt/5)
        seg_length = int((tmax)/10)
        try:
            T = 300
            iterations = 10
            slope_iterations = 1000
            tolerance = 0.2
            boot_r2 = 0.9
            boot_tolerance = 0.2
            D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme = analysis.calc_diffusion(path, name, folder, dt_diff, T,
                                                                                            iterations, slope_iterations,
                                                                                            tolerance, boot_r2,
                                                                                            boot_tolerance, seg_length)
        except:
            try:
                T = 300
                iterations = 10
                slope_iterations = 1000
                tolerance = 0.25
                boot_r2 = 0.75
                boot_tolerance = 0.25
                D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme = analysis.calc_diffusion(path, name, folder, dt_diff,
                                                                                                         T,
                                                                                                         iterations,
                                                                                                         slope_iterations,
                                                                                                         tolerance, boot_r2,
                                                                                                         boot_tolerance, seg_length)
            except:
                T = 300
                iterations = 10
                slope_iterations = 1000
                tolerance = 0.25
                boot_r2 = 0.75
                boot_tolerance = 0.25
                D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme = analysis.calc_diffusion(path, name,
                                                                                                         folder,
                                                                                                         dt_diff,
                                                                                                         T,
                                                                                                         iterations,
                                                                                                         slope_iterations,
                                                                                                         tolerance,
                                                                                                         boot_r2,
                                                                                                         boot_tolerance,
                                                                                                         seg_length)

        # Viscosity Analysis
        r = fitterSG.radius
        segments = 20
        iterations = 10
        n_boot = 10
        dt_unit = 2000000E-15
        n_point = 1000
        n_tau = 10

        eta_E, eta_E_sme, eta_fit, eta_fit_sme, eta_theo, eta_theo_sme = analysis.calc_visc(path, folder, name, r, segments, iterations, n_boot, dt_unit, n_point, n_tau, tmin, tmax)

        print("Viscosity GK (Eta_Raw): " + str(eta_E) + " ± " + str(eta_E_sme))

        # Contact Analysis
        # SG Residue Contact Map
        res_contact_file = "{}/{}/Residue_Contacts_Mean_{}.csv".format(path, folder, name)
        arr = np.loadtxt(res_contact_file, delimiter=",", dtype=float)
        analysis.sg_residue_heat_map(name, arr, "Residue_Contacts_Standardized_Mean", path, folder)

        # SG Acid Contact Map
        res_contact_file = "{}/{}/Acid_Contacts_Mean_{}.csv".format(path, folder, name)
        arr = np.loadtxt(res_contact_file, delimiter=",", dtype=float)
        analysis.sg_acid_heat_map(name, arr, "Acid_Contacts_Standardized_Mean", path=path)

        res_list = ["G3BP1", "PABP1", "TTP", "TIA1", "TDP43", "FUS", "RNA"]

        acid_list = ["Met",
                     "Gly",
                     "Lys",
                     "Thr",
                     "Arg",
                     "Ala",
                     "Asp",
                     "Glu",
                     "Tyr",
                     "Val",
                     "Leu",
                     "Gln",
                     "Trp",
                     "Phe",
                     "Ser",
                     "His",
                     "Asn",
                     "Pro",
                     "Cys",
                     "Ile",
                     "A",
                     "C",
                     "G",
                     "U",
                     ]

        if "X" not in name:
            sg_res_contact_file = "{}/ANALYSIS_SG_AVE/Residue_Contacts_Mean_sg_X.csv".format(path)
            sg_res_arr = np.loadtxt(sg_res_contact_file, delimiter=",", dtype=float)
            sg_acid_contact_file = "{}/ANALYSIS_SG_AVE/ACID_Contacts_Mean_sg_X.csv".format(path)
            sg_acid_arr = np.loadtxt(sg_acid_contact_file, delimiter=",", dtype=float)

            # SG SM Difference Contact Map
            sm_res_contact_file = "{}/{}/Residue_Contacts_Mean_{}.csv".format(path, folder, name)
            sm_res_arr = np.loadtxt(sm_res_contact_file, delimiter=",", dtype=float)

            arr = (sm_res_arr - sg_res_arr)/sg_res_arr
            np.savetxt("{}/RESULTS/Residue_Contacts_Difference_Mean_{}.csv".format(path, name), arr, delimiter=",")
            analysis.sg_residue_heat_map(name, arr, "Residue_Contacts_Difference_Mean", path, folder)

            # SG SM Acid Difference Contact Map
            sm_acid_contact_file = "{}/{}/Acid_Contacts_Mean_{}.csv".format(path, folder, name)
            sm_acid_arr = np.loadtxt(sm_acid_contact_file, delimiter=",", dtype=float)
            arr = (sm_acid_arr - sg_acid_arr) / sg_acid_arr
            analysis.sg_acid_heat_map(name, arr, "Acid_Contacts_Difference_Mean", path=path)

            # SM Residue Contact Map
            res_contact_file = "{}/{}/Residue_SM_Contacts_Mean_{}.csv".format(path, folder, name)
            arr = np.loadtxt(res_contact_file, delimiter=",", dtype=float)
            df_res_contact[name_sm] = arr

            # SG SM Acid Contact Map
            acid_contact_file = "{}/{}/Acid_SM_Contacts_Mean_{}.csv".format(path, folder, name)
            arr = np.loadtxt(acid_contact_file, delimiter=",", dtype=float)
            df_acid_contact[name_sm] = arr

            # SM Residue Count
            res_count_file = "{}/{}/BioNum_{}.csv".format(path, folder, name)
            arr = np.loadtxt(res_count_file, delimiter=",", dtype=float)
            df_res_count[name_sm] = arr

            # SM Acid Count Map
            acid_count_file = "{}/{}/AcidNum_{}.csv".format(path, folder, name)
            arr = np.loadtxt(acid_count_file, delimiter=",", dtype=float)
            df_acid_count[name_sm] = arr

        # Create SM DataFrame Row
        tempDF = pd.DataFrame([{'Small Molecule ID': sm_dict[name],
                                "Small Molecule Name": name,
                                "Compound Name": name_dict[name],
                                "c_{SM}": c,
                                "Compound Class": effect,
                                "D_Binary": d_bin,
                                "$Mass$ $(Da)$": sm_mass_dict[name],
                                "$c_{dense,SG,fit}$ $(mg/ml)$": c_dense_sg_fit,
                                "SIG$c_{dense,SG,fit}$ $(mg/ml)$": c_dense_sg_fit_se,
                                "$c_{dilute,SG,fit}$ $(mg/ml)$": c_dilute_sg_fit,
                                "SIG$c_{dilute,SG,fit}$ $(mg/ml)$": c_dilute_sg_fit_se,
                                "$c_{dense,SG,calc}$ $(mg/ml)$": c_dense_sg_calc,
                                "SIG$c_{dense,SG,calc}$ $(mg/ml)$": c_dense_sg_calc_se,
                                "$c_{dilute,SG,calc}$ $(mg/ml)$": c_dilute_sg_calc,
                                "SIG$c_{dilute,SG,calc}$ $(mg/ml)$": c_dilute_sg_calc_se,
                                "$P_{SG}$": p_SG,
                                "SIG$P_{SG}$": sig_p_SG,
                                "$R_{cond}$ $(\AA)$": R_droplet,
                                "SIG$R_{cond}$ $(\AA)$": sig_R,
                                "$W_{interface}$ $(\AA)$": W,
                                "SIG$W_{interface}$ $(\AA)$": sig_W,
                                "$\gamma_{1}$ $(mN/m)$": st1 * 1000,
                                "SIG$\gamma_{1}$ $(mN/m)$": st1_se * 1000,
                                "$\gamma_{2}$ $(mN/m)$": st2 * 1000,
                                "SIG$\gamma_{2}$ $(mN/m)$": st2_se * 1000,
                                "$\gamma_ave$ $(mN/m)$": st * 1000,
                                "SIG$\gamma_ave$ $(mN/m)$": st_se * 1000,
                                "$\Delta G_{trans}$ $(kJ/mol)$": dG,
                                "SIG$\Delta G_{trans}$ $(kJ/mol)$": se_dG,
                                "$c_{dilute,SM}$ $(mg/ml)$": c_dilute_sm,
                                "$c_{dense,SM}$ $(mg/ml)$": c_dense_sm,
                                "SIG$c_{dilute,SM}$ $(mg/ml)$": c_dilute_sm_se,
                                "SIG$c_{dense,SM}$ $(mg/ml)$": c_dense_sm_se,
                                "$P_{SM}$": pc,
                                "SIG$P_{SM}$": pc_se,
                                "$\phi_{D}$": phi_d,
                                "SIG$\phi_{D}$": phi_d_std,
                                "$N_{D}$": num_clus,
                                "SIG$N_{D}$": num_clus_std,
                                "$R_{g}$": rg,
                                "SIG$R_{g}$": rg_std,
                                "$\phi_{R}$": phi_R,
                                "SIG$\phi_{R}$": phi_R_std,
                                "$D$ $\mu m^{2} / s$": D,
                                "SIG$D$ $\mu m^{2} / s$": D_sme,
                                "$tau$ $ns$": tau,
                                "SIG$tau$ $ns$": tau_sme,
                                "$\l_{Cond}$ A": l,
                                "SIG$\l_{Cond}$ A": l_sme,
                                "$\eta_{D}$ Pa s": eta_D,
                                "SIG$\eta_{D}$ Pa s": eta_D_sme,
                                "$\eta_{GK}$ Pa s": eta_E,
                                "SIG$\eta_{GK}$ Pa s": eta_E_sme,
                                }])

        df_sg = pd.concat([df_sg, tempDF], ignore_index=True)

        return df_sg, df_res_contact, df_acid_contact, df_res_count, df_acid_count, fitterSG, fitterProtein, fitterRNA, fitterSM















    # Run Analysis
    def define_pd():
        df_acid_contact = pd.DataFrame(columns=["Acid",
                                                "D1",
                                                "D2",
                                                "D3",
                                                "D4",
                                                "D5",
                                                "D6",
                                                "D7",
                                                "D8",
                                                "D9",
                                                "D10",
                                                "ND1",
                                                "ND2",
                                                "ND3",
                                                "ND4",
                                                "ND5",
                                                "ND6",
                                                "ND7",
                                                "ND8",
                                                "ND9",
                                                "ND10",
                                                ])

        df_acid_contact["Acid"] = ["Met",
                                   "Gly",
                                   "Lys",
                                   "Thr",
                                   "Arg",
                                   "Ala",
                                   "Asp",
                                   "Glu",
                                   "Tyr",
                                   "Val",
                                   "Leu",
                                   "Gln",
                                   "Trp",
                                   "Phe",
                                   "Ser",
                                   "His",
                                   "Asn",
                                   "Pro",
                                   "Cys",
                                   "Ile",
                                   "A",
                                   "C",
                                   "G",
                                   "U"]

        df_acid_count = df_acid_contact.copy()

        df_res_contact = pd.DataFrame(columns=["Residue",
                                               "D1",
                                               "D2",
                                               "D3",
                                               "D4",
                                               "D5",
                                               "D6",
                                               "D7",
                                               "D8",
                                               "D9",
                                               "D10",
                                               "ND1",
                                               "ND2",
                                               "ND3",
                                               "ND4",
                                               "ND5",
                                               "ND6",
                                               "ND7",
                                               "ND8",
                                               "ND9",
                                               "ND10",
                                               ])

        df_res_contact["Residue"] = ["G3BP1",
                                     "PABP1",
                                     "TTP",
                                     "TIA1",
                                     "TDP43",
                                     "FUS",
                                     "RNA"]

        df_res_count = df_res_contact.copy()

        df_sg = pd.DataFrame(columns=['Small Molecule ID',
                                   "Small Molecule Name",
                                    "Compound Name",
                                   "c_{SM}",
                                   "Compound Class",
                                   "D_Binary",
                                   "$Mass$ $(Da)$",
                                   "$c_{dense,SG,fit}$ $(mg/ml)$",
                                   "SIG$c_{dense,SG,fit}$ $(mg/ml)$",
                                   "$c_{dilute,SG,fit}$ $(mg/ml)$",
                                   "SIG$c_{dilute,SG,fit}$ $(mg/ml)$",
                                   "$c_{dense,SG,calc}$ $(mg/ml)$",
                                   "SIG$c_{dense,SG,calc}$ $(mg/ml)$",
                                   "$c_{dilute,SG,calc}$ $(mg/ml)$",
                                   "SIG$c_{dilute,SG,calc}$ $(mg/ml)$",
                                   "$P_{SG}$",
                                   "SIG$P_{SG}$",
                                   "$R_{cond}$ $(\AA)$",
                                   "SIG$R_{cond}$ $(\AA)$",
                                   "$W_{interface}$ $(\AA)$",
                                   "SIG$W_{interface}$ $(\AA)$",
                                   "$\gamma_{1}$ $(mN/m)$",
                                   "SIG$\gamma_{1}$ $(mN/m)$",
                                   "$\gamma_{2}$ $(mN/m)$",
                                   "SIG$\gamma_{2}$ $(mN/m)$",
                                   "$\gamma_ave$ $(mN/m)$",
                                   "SIG$\gamma_ave$ $(mN/m)$",
                                   "$\Delta G_{trans}$ $(kJ/mol)$",
                                   "SIG$\Delta G_{trans}$ $(kJ/mol)$",
                                   "$c_{dilute,SM}$ $(mg/ml)$",
                                   "$c_{dense,SM}$ $(mg/ml)$",
                                   "SIG$c_{dilute,SM}$ $(mg/ml)$",
                                   "SIG$c_{dense,SM}$ $(mg/ml)$",
                                   "$P_{SM}$",
                                   "SIG$P_{SM}$",
                                   "$\phi_{D}$",
                                   "SIG$\phi_{D}$",
                                   "$N_{D}$",
                                   "SIG$N_{D}$",
                                   "$R_{g}$",
                                   "SIG$R_{g}$",
                                   "$\phi_{R}$",
                                   "SIG$\phi_{R}$",
                                   "$D$ $\mu m^{2} / s$",
                                   "SIG$D$ $\mu m^{2} / s$",
                                   "$tau$ $ns$",
                                   "SIG$tau$ $ns$",
                                   "$\l_{Cond}$ A",
                                   "SIG$\l_{Cond}$ A",
                                   "$\eta_{D}$ Pa s",
                                   "SIG$\eta_{D}$ Pa s",
                                   "$\eta_{GK}$ Pa s",
                                   "SIG$\eta_{GK}$ Pa s"])

        return df_sg, df_res_contact, df_acid_contact, df_res_count, df_acid_count

    def gen_path(path, folder):
        full_path = "{}/{}".format(path,folder)
        if not os.path.exists(full_path):
            # Create the folder
            os.makedirs(full_path)
        else:
            shutil.rmtree(full_path)
            os.makedirs(full_path)

    path = sys.argv[1]
    dt = int(sys.argv[2])
    tmin = int(sys.argv[3])
    tmax = int(sys.argv[4])
    c = 1
    T = 300
    gen_path(path, "IMAGES")
    gen_path(path, "FIGURES")
    gen_path(path, "RESULTS")

    if "THEORETICAL" not in path:
        sg = True
        prot = True
        rna = True
        df_all, df_res_contact_all, df_acid_contact_all, df_res_count_all, df_acid_count_all = define_pd()
        mol_name = "X"

        folder = "ANALYSIS_SG_AVE"
        name = "sg_X"
        sm = False
        df_sg, df_res_contact_sg, df_acid_contact_sg, df_res_count_sg, df_acid_count_sg = define_pd()

        print(name)
        dfs = sg_sm_analysis_full(path, folder, name, df_sg, mol_name, c, df_res_contact_sg, df_acid_contact_sg, df_res_count_sg, df_acid_count_sg, T, dt, tmin, tmax)
        df_sg = dfs[0]
        df_res_contact_sg = dfs[1]
        df_acid_contact_sg = dfs[2]
        df_res_count_sg = dfs[3]
        df_acid_count_sg = dfs[4]

        df_sg.to_csv("{}/RESULTS/SG_Quant_Data.csv".format(path), index=False)








        sm = True
        df_dsm, df_res_contact, df_acid_contact, df_res_count, df_acid_count = define_pd()
        folder = "ANALYSIS_DSM_AVE"

        dsm_names = ["dsm_anisomycin", "dsm_daunorubicin", "dsm_dihydrolipoicacid", "dsm_hydroxyquinoline","dsm_lipoamide","dsm_lipoicacid", "dsm_mitoxantrone", "dsm_pararosaniline", "dsm_pyrivinium", "dsm_quinicrine"]

        for name in dsm_names:
            print(name)
            dfs = sg_sm_analysis_full(path, folder, name, df_dsm, mol_name, c, df_res_contact, df_acid_contact, df_res_count, df_acid_count, T, dt, tmin, tmax)
            df_dsm = dfs[0]
            df_res_contact = dfs[1]
            df_acid_contact = dfs[2]
            df_res_count = dfs[3]
            df_acid_count = dfs[4]
    
        df_dsm = df_dsm.sort_values(by='Small Molecule ID', ascending=False)

        df = pd.DataFrame
        df_mean = df_dsm
        mean_list = []
        for columnName in df_mean:
            avg = pd.to_numeric(df_mean[columnName], errors='coerce').mean()
            if math.isnan(avg):
                avg = "DSM_AVG"
            mean_list.append(avg)

        df_mean.loc[len(df_mean.index)] = mean_list
        df_dsm = pd.concat([df_dsm, df_mean], ignore_index=True)
    

        dfs = sg_sm_analysis_full(path, "ANALYSIS_DSM_AGG", "DSM", df_dsm, "DSM", c, df_res_contact, df_acid_contact, df_res_count, df_acid_count, T, dt, tmin, tmax)
        df_dsm = dfs[0]
        df_res_contact = dfs[1]
        df_acid_contact = dfs[2]
        df_res_count = dfs[3]
        df_acid_count = dfs[4]

        df_dsm_clean = df_dsm.drop_duplicates(subset=["Small Molecule ID"])

        df_ndsm, df_res_contact_ndsm, df_acid_contact_ndsm, df_acid_contact_ndsm, df_res_count_ndsm = define_pd()
        folder = "ANALYSIS_NDSM_AVE"


        ndsm_names = ["ndsm_dmso", "ndsm_valeric", "ndsm_ethylenediamine", "ndsm_propanedithiol","ndsm_hexanediol", "ndsm_diethylaminopentane", "ndsm_aminoacridine","ndsm_anthraquinone", "ndsm_acetylenapthacene", "ndsm_anacardic"]

        for name in ndsm_names:
            print(name)
            dfs = sg_sm_analysis_full(path, folder, name, df_ndsm, mol_name, c, df_res_contact, df_acid_contact, df_res_count, df_acid_count, T, dt, tmin, tmax)
            df_ndsm = dfs[0]
            df_res_contact = dfs[1]
            df_acid_contact = dfs[2]
            df_res_count = dfs[3]
            df_acid_count = dfs[4]

        df_ndsm = df_ndsm.sort_values(by='Small Molecule ID', ascending=False)

        df = pd.DataFrame
        df_mean = df_ndsm
        mean_list = []
        for columnName in df_mean:
            avg = pd.to_numeric(df_mean[columnName], errors='coerce').mean()
            if math.isnan(avg):
                avg = "NDSM_AVG"
            mean_list.append(avg)

        df_mean.loc[len(df_mean.index)] = mean_list
        df_ndsm = pd.concat([df_ndsm, df_mean], ignore_index=True)

        dfs = sg_sm_analysis_full(path, "ANALYSIS_NDSM_AGG", "NDSM", df_ndsm, "NDSM", c, df_res_contact, df_acid_contact, df_res_count, df_acid_count, T, dt, tmin, tmax)
        df_ndsm = dfs[0]
        df_res_contact = dfs[1]
        df_acid_contact = dfs[2]
        df_res_count = dfs[3]
        df_acid_count = dfs[4]

        df_ndsm_clean = df_ndsm.drop_duplicates(subset=["Small Molecule ID"])

        df_res_contact.to_csv("{}/RESULTS/SM_ResMap_Data.csv".format(path), index=False)
        df_acid_contact.to_csv("{}/RESULTS/SM_AcidMap_Data.csv".format(path), index=False)

        df_res_count.to_csv("{}/RESULTS/SM_ResCount_Data.csv".format(path), index=False)
        df_acid_count.to_csv("{}/RESULTS/SM_AcidCount_Data.csv".format(path), index=False)

        df_all = pd.concat([df_sg, df_dsm, df_ndsm], ignore_index=True)

        df_all_clean = df_all.drop_duplicates(subset=["Small Molecule ID"])

        df_dsm_clean.to_csv("{}/RESULTS/DSM_Quant_Data.csv".format(path), index=False)
        df_ndsm_clean.to_csv("{}/RESULTS/NDSM_Quant_Data.csv".format(path), index=False)
        df_all_clean.to_csv("{}/RESULTS/Quant_Data.csv".format(path), index=False)


    if "THEORETICAL" in path:
        dsm_names = []
        ndsm_names = []
        list_path = path.replace("_THEORETICAL", "")
        with open('{}/dsm_list.txt'.format(list_path), 'r') as f:
            for i in f.readlines():
                dsm_names.append(i.strip())

        with open('{}/ndsm_list.txt'.format(list_path), 'r') as f:
            for i in f.readlines():
                ndsm_names.append(i.strip())

        df_all = pd.read_csv("{}/RESULTS/Quant_Data.csv".format(list_path))
        df_res_contact = pd.read_csv("{}/RESULTS/SM_ResMap_Data.csv".format(list_path)).iloc[:,0:21]
        df_acid_contact = pd.read_csv("{}/RESULTS/SM_AcidMap_Data.csv".format(list_path)).iloc[:,0:21]
        df_res_count = pd.read_csv("{}/RESULTS/SM_ResCount_Data.csv".format(list_path)).iloc[:, 0:21]
        df_acid_count = pd.read_csv("{}/RESULTS/SM_AcidCount_Data.csv".format(list_path)).iloc[:, 0:21]
        shutil.copy2('{}/RESULTS/Residue_Contacts_Standardized_Mean_sg_X.csv'.format(list_path), '{}/RESULTS/Residue_Contacts_Standardized_Mean_sg_X.csv'.format(path))
        shutil.copy2('{}/RESULTS/Acid_Contacts_Standardized_Mean_sg_X.csv'.format(list_path), '{}/RESULTS/Acid_Contacts_Standardized_Mean_sg_X.csv'.format(path))

        row_sel = ["SG",
                   "D1",
                   "D2",
                   "D3",
                   "D4",
                   "D5",
                   "D6",
                   "D7",
                   "D8",
                   "D9",
                   "D10",
                   "ND1",
                   "ND2",
                   "ND3",
                   "ND4",
                   "ND5",
                   "ND6",
                   "ND7",
                   "ND8",
                   "ND9",
                   "ND10"]

        mask = df_all["Small Molecule ID"].isin(row_sel)

        df_sel = df_all[mask].drop_duplicates()
        df_all_clean = df_sel

        count_dsm = 10
        count_ndsm = 10

        for ind in df_all_clean.index:
            if df_all_clean["Small Molecule Name"][ind] in dsm_names:
                df_all_clean["Compound Class"][ind] = "Dissolving"
                df_all_clean["D_Binary"][ind] = 1
                if "N" in df_all_clean["Small Molecule ID"][ind]:
                    count_dsm+=1
                    df_all_clean["Small Molecule ID"][ind] = "D{}".format(count_dsm)
            elif df_all_clean["Small Molecule Name"][ind] in ndsm_names:
                df_all_clean["Compound Class"][ind] = "Non-Dissolving"
                df_all_clean["D_Binary"][ind] = 0
                if "N" not in df_all_clean["Small Molecule ID"][ind]:
                    count_ndsm+=1
                    df_all_clean["Small Molecule ID"][ind] = "ND{}".format(count_ndsm)

        df_mean = df_all_clean.loc[df_all_clean['D_Binary'] == 1]
        mean_list = []
        for columnName in df_mean:
            avg = pd.to_numeric(df_mean[columnName], errors='coerce').mean()
            if math.isnan(avg):
                avg = "DSM_AVG"
            mean_list.append(avg)

        df_mean.loc[len(df_mean.index)] = mean_list
        df_all_clean = pd.concat([df_all_clean, df_mean], ignore_index=True)

        dfs = sg_sm_analysis_full(path, "ANALYSIS_DSM_AGG", "DSM", df_all_clean, "DSM", c, df_res_contact, df_acid_contact, df_res_count, df_acid_count, T,
                                  dt, tmin, tmax)
        df_all_clean = dfs[0]
        df_res_contact = dfs[1]
        df_acid_contact = dfs[2]
        df_res_count = dfs[3]
        df_acid_count = dfs[4]

        print(df_res_contact)
        print(df_res_count)

        df_mean = df_all_clean.loc[df_all_clean['D_Binary'] == 0]
        mean_list = []
        for columnName in df_mean:
            avg = pd.to_numeric(df_mean[columnName], errors='coerce').mean()
            if math.isnan(avg):
                avg = "NDSM_AVG"
            mean_list.append(avg)

        df_mean.loc[len(df_mean.index)] = mean_list
        df_all_clean = pd.concat([df_all_clean, df_mean], ignore_index=True)

        dfs = sg_sm_analysis_full(path, "ANALYSIS_NDSM_AGG", "NDSM", df_all_clean, "NDSM", c, df_res_contact,
                                  df_acid_contact, df_res_count, df_acid_count, T,
                                  dt, tmin, tmax)
        df_all_clean = dfs[0]
        df_res_contact = dfs[1]
        df_acid_contact = dfs[2]
        df_res_count = dfs[3]
        df_acid_count = dfs[4]

        print(df_res_contact)
        print(df_res_count)

        df_all_clean = df_all_clean.drop_duplicates()

        df_all_clean.to_csv("{}/RESULTS/Quant_Data.csv".format(path), index=False)
        df_res_contact.to_csv("{}/RESULTS/SM_ResMap_Data.csv".format(path), index=False)
        df_acid_contact.to_csv("{}/RESULTS/SM_AcidMap_Data.csv".format(path), index=False)
        df_res_count.to_csv("{}/RESULTS/SM_ResCount_Data.csv".format(path), index=False)
        df_acid_count.to_csv("{}/RESULTS/SM_AcidCount_Data.csv".format(path), index=False)

    print(df_all_clean)



    # Residue DSM / NDSM Difference Standardized Array
    res_list = ["G3BP1",
                "PABP1",
                "TTP",
                "TIA1",
                "TDP43",
                "FUS",
                "RNA"]

    ndsm_res_arr = np.array(pd.read_csv("{}/RESULTS/Residue_Contacts_Standardized_Mean_NDSM.csv".format(path), header=None))
    dsm_res_arr = np.array(pd.read_csv("{}/RESULTS/Residue_Contacts_Standardized_Mean_DSM.csv".format(path), header=None))
    sg_res_arr = np.array(pd.read_csv("{}/RESULTS/Residue_Contacts_Standardized_Mean_sg_X.csv".format(path), header=None))

    res_contact_array = np.divide(np.subtract(dsm_res_arr, ndsm_res_arr), sg_res_arr)

    x_res_list = ["TDP43", "FUS", "TIA1", "G3BP1", "RNA", "PABP1", "TTP"]


    y_res_list = x_res_list


    afont = {'fontname': 'Arial'}
    sns.set_theme(style="ticks")
    sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=10)  # fontsize of the axes title
    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=10)  # legend fontsize
    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('axes', linewidth=2)

    plt.figure(figsize=(3.8, 3.0), tight_layout=True)
    ax = sns.heatmap(res_contact_array, xticklabels=x_res_list, yticklabels=y_res_list, cmap="coolwarm", square=True)
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=10)
    cbar.formatter.set_powerlimits((0, 0))

    # to get 10^3 instead of 1e3
    cbar.formatter.set_useMathText(True)

    file_name = "{}/IMAGES/Residue_DSM_NDSM_Difference_Standardized_HeatMap.png".format(path)

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    plt.savefig(file_name, format="png", dpi=400)

    df_acid = pd.DataFrame(res_contact_array)

    # save the dataframe as a csv file
    np.savetxt("{}/RESULTS/Residue_DSM_NDSM_Difference_Standardized_HeatMap".format(path), res_contact_array, delimiter=",")




    # Acid DSM / NDSM Difference Standardized Array

    ndsm_acid_arr = np.array(pd.read_csv("{}/RESULTS/Acid_Contacts_Standardized_Mean_NDSM.csv".format(path), header=None))
    dsm_acid_arr = np.array(pd.read_csv("{}/RESULTS/Acid_Contacts_Standardized_Mean_DSM.csv".format(path), header=None))
    sg_acid_arr = np.array(pd.read_csv("{}/RESULTS/Acid_Contacts_Standardized_Mean_sg_X.csv".format(path), header=None))

    acid_contact_array = np.divide(np.subtract(dsm_acid_arr, ndsm_acid_arr), sg_acid_arr)

    swap_contact_array = acid_contact_array

    acid_contact_array = swap_contact_array

    acid_list = [
        'ARG',
        'HIS',
        'LYS',
        'ASP',
        'GLU',
        'SER',
        'THR',
        'ASN',
        'GLN',
        'CYS',
        'GLY',
        'PRO',
        'ALA',
        'VAL',
        'ILE',
        'LEU',
        'MET',
        'PHE',
        'TYR',
        'TRP',
        'A',
        'U',
        'C',
        'G'
    ]

    afont = {'fontname': 'Arial'}
    sns.set_theme(style="ticks")
    sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=10)  # fontsize of the axes title
    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)  # fontsize of the tick labels
    plt.rc('legend', fontsize=10)  # legend fontsize
    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('axes', linewidth=2)

    plt.figure(figsize=(3.6, 2.8), tight_layout=True)
    ax = sns.heatmap(acid_contact_array, xticklabels=acid_list, yticklabels=acid_list, cmap="coolwarm", square=True)
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=10)
    cbar.formatter.set_powerlimits((0, 0))

    # to get 10^3 instead of 1e3
    cbar.formatter.set_useMathText(True)
    file_name = "{}/IMAGES/Acid_DSM_NDSM_Difference_Standardized_HeatMap.png".format(path)

    plt.savefig(file_name, format="png", dpi=400)
    df_acid = pd.DataFrame(acid_contact_array)

    # save the dataframe as a csv file
    np.savetxt("{}/RESULTS/Acid_DSM_NDSM_Difference_Standardized_HeatMap".format(path), acid_contact_array, delimiter=",")




    # SM Residue Contact Array

    df_res_contact = pd.read_csv("{}/RESULTS/SM_ResMap_Data.csv".format(path))
    print(df_res_contact)

    sm_list = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        "ND1",
        "ND2",
        "ND3",
        "ND4",
        "ND5",
        "ND6",
        "ND7",
        "ND8",
        "ND9",
        "ND10",
        "DSM",
        "NDSM"
    ]

    sm_res_contact_array = np.array(df_res_contact.iloc[:,1:])

    print(sm_res_contact_array)

    afont = {'fontname': 'Arial'}
    sns.set_theme(style="ticks")
    sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=10)  # fontsize of the axes title
    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=10)  # legend fontsize
    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('axes', linewidth=2)

    fig, ax1 = plt.subplots(figsize=(3.6, 3.2), tight_layout=True)
    sns.heatmap(sm_res_contact_array, xticklabels=sm_list, yticklabels=res_list, cmap="Blues", square=False)
    file_name = "{}/IMAGES/SM_Residue_DSM_NDSM_SUMMARY_Standardized_HeatMap.png".format(path)
    ax1.set_ylabel("")
    ax1.axes.get_yaxis().set_visible(False)



    # SM Acid Contact Array

    df_acid_contact = pd.read_csv("{}/RESULTS/SM_AcidMap_Data.csv".format(path))
    print(df_acid_contact)

    sm_list = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        "ND1",
        "ND2",
        "ND3",
        "ND4",
        "ND5",
        "ND6",
        "ND7",
        "ND8",
        "ND9",
        "ND10",
        "DSM",
        "NDSM"
    ]

    df_acid_contact.to_csv("{}/RESULTS/DF_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), index=False)

    sm_acid_contact_array = np.array(df_acid_contact.iloc[:,1:])

    np.savetxt("{}/RESULTS/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), sm_acid_contact_array, delimiter=",")

    afont = {'fontname': 'Arial'}
    sns.set_theme(style="ticks")
    sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=10)  # fontsize of the axes title
    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the tick labels
    plt.rc('legend', fontsize=10)  # legend fontsize
    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('axes', linewidth=2)

    fig, ax1 = plt.subplots(figsize=(3.6, 3.2), tight_layout=True)
    sns.heatmap(sm_acid_contact_array, xticklabels=sm_list, yticklabels=acid_list, cmap="Blues", square=False)
    file_name = "{}/IMAGES/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap.png".format(path)
    ax1.set_ylabel("")
    ax1.axes.get_yaxis().set_visible(False)

    plt.savefig(file_name, format="png", dpi=400)

    # save the dataframe as a csv file
    np.savetxt("{}/RESULTS/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap".format(path), sm_acid_contact_array, delimiter=",")

    # SM Residue Summary Contact Array
    b = df_res_contact.loc[:, "DSM"] - df_res_contact.loc[:, "NDSM"]
    sm_res_contact_array = np.transpose(np.asarray([b]))

    print(sm_res_contact_array)

    afont = {'fontname': 'Arial'}
    sns.set_theme(style="ticks")
    sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=10)  # fontsize of the axes title
    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=10)  # legend fontsize
    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('axes', linewidth=2)

    fig, ax1 = plt.subplots(figsize=(3.0, 2.8), tight_layout=True)
    ax = sns.heatmap(sm_res_contact_array, xticklabels=[""], yticklabels=res_list, cmap="coolwarm", square=False)
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20

    cbar.formatter.set_powerlimits((0, 0))

    # to get 10^3 instead of 1e3
    cbar.formatter.set_useMathText(True)
    cbar.ax.tick_params(labelsize=10)
    file_name = "{}/IMAGES/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap.png".format(path)
    ax1.set_ylabel("")
    file_name = "{}/IMAGES/SM_Residue_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap.png".format(path)
    ax1.set_ylabel("")
    # ax1.axes.get_yaxis().set_visible(False)

    plt.savefig(file_name, format="png", dpi=400)

    # save the dataframe as a csv file
    np.savetxt("{}/RESULTS/SM_Residue_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap".format(path),
               sm_res_contact_array, delimiter=",")

    # SM Acid Contact Array

    df_acid_contact = pd.read_csv("{}/RESULTS/SM_AcidMap_Data.csv".format(path))
    print(df_acid_contact)

    sm_list = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        "ND1",
        "ND2",
        "ND3",
        "ND4",
        "ND5",
        "ND6",
        "ND7",
        "ND8",
        "ND9",
        "ND10",
        "DSM",
        "NDSM"
    ]

    df_acid_contact.to_csv("{}/RESULTS/DF_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), index=False)

    sm_acid_contact_array = np.array(df_acid_contact.iloc[:, 1:])

    np.savetxt("{}/RESULTS/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), sm_acid_contact_array,
               delimiter=",")

    afont = {'fontname': 'Arial'}
    sns.set_theme(style="ticks")
    sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=10)  # fontsize of the axes title
    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the tick labels
    plt.rc('legend', fontsize=10)  # legend fontsize
    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('axes', linewidth=2)

    fig, ax1 = plt.subplots(figsize=(3.2, 3.0), tight_layout=True)
    sns.heatmap(sm_acid_contact_array, xticklabels=sm_list, yticklabels=acid_list, cmap="Blues", square=False)
    file_name = "{}/IMAGES/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap.png".format(path)
    ax1.set_ylabel("")
    ax1.axes.get_yaxis().set_visible(False)

    plt.savefig(file_name, format="png", dpi=400)

    # save the dataframe as a csv file
    np.savetxt("{}/RESULTS/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap".format(path), sm_acid_contact_array,
               delimiter=",")

    # SM Acid Contact Summary Array
    b = df_acid_contact.loc[:, "DSM"] - df_acid_contact.loc[:, "NDSM"]
    sm_acid_contact_array = np.transpose(np.asarray([b]))

    np.savetxt("{}/RESULTS/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap.csv".format(path),
               sm_acid_contact_array, delimiter=",")

    afont = {'fontname': 'Arial'}
    sns.set_theme(style="ticks")
    sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=10)  # fontsize of the axes title
    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)  # fontsize of the tick labels
    plt.rc('legend', fontsize=10)  # legend fontsize
    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('axes', linewidth=2)

    fig, ax1 = plt.subplots(figsize=(3.0, 2.8), tight_layout=True)
    ax = sns.heatmap(sm_acid_contact_array, xticklabels=[""], yticklabels=acid_list, cmap="coolwarm", square=False)
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20

    cbar.formatter.set_powerlimits((0, 0))

    # to get 10^3 instead of 1e3
    cbar.formatter.set_useMathText(True)
    cbar.ax.tick_params(labelsize=10)
    file_name = "{}/IMAGES/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap.png".format(path)
    ax1.set_ylabel("")
    # ax1.axes.get_yaxis().set_visible(False)

    plt.savefig(file_name, format="png", dpi=400)

    # save the dataframe as a csv file
    np.savetxt("{}/RESULTS/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap".format(path),
               sm_acid_contact_array, delimiter=",")

