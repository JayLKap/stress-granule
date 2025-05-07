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
import shutil
from tqdm import tqdm
from scipy.stats import sem
import warnings
warnings.filterwarnings("ignore")
from RDP_TIME import RDP

class Average:
    def __init__(self, category, tmin, tmax, dt):
        self.df = pd.DataFrame()
        self.category = category
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt

    def rdp_ave(self, biopolymer, sm):
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        for i in range(self.tmin,self.tmax,self.dt):
            try:
                distance_col = (pd.read_csv("{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, str(i)))["Distance from center of mass (A)"])
                if biopolymer == "SM":
                    density_col = list(pd.read_csv("{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, str(i)))["SM density (mg/mL)"])
                else:
                    density_col = list(
                        pd.read_csv("{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, str(i)))[
                            "Protein density (mg/mL)"])
                df_temp[str(i)]=density_col
            except:
                print("{}/DensityProfile_{}_{}_{}.csv File Missing".format(category,biopolymer, sm, str(i)))

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)
        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        df.to_csv("{}/{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm), index=False)

    def cluster_ave(self, biopolymer, sm):
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df_list = []
        index_list = []
        df_cols = pd.read_csv("{}/Cluster_{}_{}_{}.csv".format(category, biopolymer, sm, self.tmin)).columns.tolist()
        for i in range(self.tmin,self.tmax,self.dt):
            try:
                mean_values = pd.read_csv("{}/Cluster_{}_{}_{}.csv".format(category, biopolymer, sm, str(i))).mean()
                df_list.append(mean_values)
                index_list.append(i)
            except:
                print("Cluster_{}_{}_{}.csv File Missing".format(biopolymer, sm, str(i)))

        df = pd.DataFrame(columns=df_cols,data=df_list, index=index_list)
        df['Timestep'] = (np.array(df['Timestep'])+5.1)*5/1000
        df['Phi'] = (np.array(df['Chains in Largest Droplet'])/np.array(df['Total Chain Number']))
        return df


class Aggregate:
    def __init__(self, path, category):
        self.df = pd.DataFrame()
        self.path=path
        self.category = category

    def rdp_ave(self, biopolymer, sm_list):
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        for sm in sm_list:
            distance_col = (pd.read_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))["Distance from center of mass (A)"])
            if biopolymer == "SM":
                density_col = list(pd.read_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))["SM density (mg/mL)"])
            else:
                density_col = list(
                    pd.read_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))[
                        "Protein density (mg/mL)"])
            df_temp[str(sm)]=density_col

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)

        df["Distance from center of mass (A)"] = distance_col
        df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        df.to_csv("{}/ANALYSIS_{}_AGG/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False)

    def cluster_ave(self, biopolymer, sm_list):
        df_list = []
        index_list = []
        df_cols = pd.read_csv("{}/ANALYSIS_{}_AVE/Cluster_{}_{}.csv".format(self.path, self.category, biopolymer, sm_list[0])).columns.tolist()
        for sm in sm_list:
            mean_values = pd.read_csv("{}/ANALYSIS_{}_AVE/Cluster_{}_{}.csv".format(self.path, self.category, biopolymer, sm)).mean()
            df_list.append(mean_values)
            index_list.append(sm)

        df = pd.DataFrame(columns=df_cols, data=df_list, index=index_list)
        rg_sem = df["Largest Droplet Radius of Gyration"].sem()
        nd_sem = df["Number of Droplets"].sem()
        chains_largest_sem = df["Number of Droplets"].sem()
        mass_largest_sem = df["Mass of Largest Droplet (mg)"].sem()
        number_external_sem = df["Number of External Chains"].sem()
        mass_external_sem = df["Mass of External Chains"].sem()

        df_mean = df.mean()
        df = pd.DataFrame(df_mean).transpose()
        df["RG SEM"] = rg_sem
        df["ND SEM"] = nd_sem
        df["Chains Largest SEM"] = chains_largest_sem
        df["Mass Largest SEM"] = mass_largest_sem
        df["NE SEM"] = number_external_sem
        df["ME SEM"] = mass_external_sem
        df.to_csv("{}/ANALYSIS_{}_AGG/Cluster_{}_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=True)






class Average_Biopolymers():
    def __init__(self, path, category, tmin, tmax, dt):
        self.path = path
        self.category=category
        self.df = pd.DataFrame()
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt

    def rdp_ave(self, biopolymer, sm):
        category = sm.split("_")[0].upper()
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        for i in range(self.tmin, self.tmax, self.dt):
            #try:
            distance_col = (pd.read_csv("ANALYSIS_{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, str(i)))[
                "Distance from center of mass (A)"])
            if biopolymer == "SM":
                density_col = list(
                    pd.read_csv("ANALYSIS_{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, str(i)))[
                        "SM density (mg/mL)"])
            else:
                density_col = list(
                    pd.read_csv("ANALYSIS_{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, str(i)))[
                        "Protein density (mg/mL)"])
            df_temp[str(i)] = density_col
            #except:
            #    print("{}/DensityProfile_{}_{}_{}.csv File Missing".format(path, biopolymer, sm, str(i)))

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)

        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        if sm == "sg_X":
            sm = "SG"
        df.to_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm), index=False)

    def rdp_sm_ave(self, biopolymer, sm_list, sm_type):
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        for sm in sm_list:
            distance_col = (pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))[
                "Distance from center of mass (A)"])
            if biopolymer == "SM":
                density_col = list(
                    pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))[
                        "SM density (mg/mL)"])
            else:
                density_col = list(
                    pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))[
                        "Protein density (mg/mL)"])
            df_temp[str(sm)] = density_col
        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)
        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        df.to_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm_type), index=False)

    def gen_ave(self, sm_list, sm_type):
        biopolymer_list = ["RNA","ADENINE","UCG","Protein","G3BP1","TDP43","TTP","TIA1","FUS","PABP1"]
        for sm in tqdm(sm_list):
            for biopolymer in biopolymer_list:
                self.rdp_ave(biopolymer, sm)

    def gen_agg(self, sm_list, sm_type):
        biopolymer_list = ["RNA","ADENINE","UCG","Protein","G3BP1","TDP43","TTP","TIA1","FUS","PABP1"]
        for biopolymer in biopolymer_list:
            if sm_type != "SG":
                self.rdp_sm_ave(biopolymer,sm_list,sm_type)

if __name__ == '__main__':

    def cluster_ave(tmin, tmax, dt, biopolymer, sm):
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df_list = []
        index_list = []
        df_cols = pd.read_csv("{}/Cluster_{}_{}_{}.csv".format(category, biopolymer, sm, tmin)).columns.tolist()
        for i in range(tmin,tmax,dt):
            try:
                mean_values = pd.read_csv("{}/Cluster_{}_{}_{}.csv".format(category, biopolymer, sm, str(i))).mean()
                df_list.append(mean_values)
                index_list.append(i)
            except:
                print("Cluster_{}_{}_{}.csv File Missing".format(biopolymer, sm, str(i)))

        df = pd.DataFrame(columns=df_cols,data=df_list, index=index_list)
        df['Timestep'] = (np.array(df['Timestep'])+5.1)*5/1000
        df['Phi'] = (np.array(df['Chains in Largest Droplet'])/np.array(df['Total Chain Number']))
        df['Phi'].fillna(df['Phi'].mean(), inplace=True)
        df['Number of Droplets'].fillna(df['Number of Droplets'].mean(), inplace=True)
        return df

    def plot_cluster(timestep, mean_sg, mean_dsm, sem_dsm, mean_ndsm, sem_ndsm):
        col_pal_sg = sns.color_palette(["#808080"], 1)[0]
        col_pal_protein = sns.color_palette(["#C7383A"], 1)[0]
        col_pal_rna = sns.color_palette(["#0066ff"], 1)[0]
        col_pal_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)

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

        fig, ax1 = plt.subplots(figsize=(3.2, 3.2))

        ax1.plot(timestep, mean_sg, color=col_pal_sg, label='CONTROL', linewidth=4, zorder=1)
        sns.scatterplot(ax=ax1, x=timestep, y=mean_sg, color=col_pal_sg, legend=False,
                        s=40,
                        edgecolor="k", linewidth=1, zorder=3, clip_on=False)

        ax1.plot(timestep, mean_dsm, color=col_pal_sm[0], label='DSM', linewidth=4, zorder=1)
        sns.scatterplot(ax=ax1, x=timestep, y=mean_dsm, color=col_pal_sm[0], legend=False,
                        s=40,
                        edgecolor="k", linewidth=1, zorder=3, clip_on=False)
        ax1.errorbar(x=timestep, y=mean_dsm, yerr=sem_dsm, fmt=".", color=col_pal_sm[0],
                     zorder=2)

        ax1.plot(timestep, mean_ndsm, color=col_pal_sm[1], label='NDSM', linewidth=4, zorder=1)
        sns.scatterplot(ax=ax1, x=timestep, y=mean_ndsm, color=col_pal_sm[1], legend=False,
                        s=40,
                        edgecolor="k", linewidth=1, zorder=3, clip_on=False)
        ax1.errorbar(x=timestep, y=mean_ndsm, yerr=sem_ndsm, fmt=".", color=col_pal_sm[1],
                     zorder=2)

        ax1.set_xlabel("")
        ax1.set_ylabel("")
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)
        return fig, ax1



    def rdp_ave(tmin, tmax, dt, biopolymer, sm):
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        for i in range(tmin,tmax,dt):
            try:
                distance_col = (pd.read_csv("{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, str(i)))["Distance from center of mass (A)"])
                if biopolymer == "SM":
                    density_col = list(pd.read_csv("{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, str(i)))["SM density (mg/mL)"])
                else:
                    density_col = list(
                        pd.read_csv("{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, str(i)))[
                            "Protein density (mg/mL)"])
                df_temp[str(i)]=density_col
            except:
                print("{}/DensityProfile_{}_{}_{}.csv File Missing".format(category,biopolymer, sm, str(i)))

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)
        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        return df


    def rdf_ave(tmin, tmax, dt, sm):
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df = pd.read_csv("{}/RDF_{}_{}.csv".format(category, sm, tmin))
        for i in range(tmin + dt, tmax, dt):
            try:
                df_temp = pd.read_csv("{}/RDF_{}_{}.csv".format(category, sm, i))
                df = pd.concat([df, df_temp])
            except:
                print("{}/RDF_{}_{}.csv".format(category, sm, str(i)))

        df_mean = df.groupby(level=0).mean()
        return df_mean

    def plot_rdp(df_list, time_list, sm):
        pca_file_A = "CLASSIFY_50_2000/ANALYSIS_SG_AVE/PCA_Protein_sg_X.csv"
        cluster_file_A = "CLASSIFY_50_2000/ANALYSIS_SG_AVE/Cluster_Protein_sg_X.csv"
        init = [80, 80, 200, 80]
        T = 300
        linestyle_list = ['-', '--', ':']
        marker_list = ['o', 'D', 'v']

        col_pal_sg = sns.color_palette(["#808080"], 1)[0]
        col_pal_protein = sns.color_palette(["#C7383A"], 1)[0]
        col_pal_rna = sns.color_palette(["#0066ff"], 1)[0]
        col_pal_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)

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

        fig, ax1 = plt.subplots(figsize=(2.4, 2.4))

        for i, df_dict in enumerate(df_list):
            label_sg = f"SG {time_list[i]}"
            print(label_sg)
            fitterSG = RDP(df_dict["SG"], pca_file_A, cluster_file_A, T, init, label_sg)

            label_protein = f"Protein {time_list[i]}"
            print(label_protein)
            fitterProtein = RDP(df_dict["Protein"], pca_file_A, cluster_file_A, T, init, label_protein)

            label_rna = f"RNA {time_list[i]}"
            print(label_rna)
            fitterRNA = RDP(df_dict["RNA"], pca_file_A, cluster_file_A, T, init, label_rna)

            if sm == "DSM" or sm == "NDSM":
                label_sm = f"{sm} {time_list[i]}"
                print(label_sm)
                fitterSM = RDP(df_dict["SM"], pca_file_A, cluster_file_A, T, init, label_sm)

            else:
                fitterSM = None

            ax1.plot(fitterSG.fit_x, fitterSG.fit_rho, color=col_pal_sg, label=label_sg, linewidth=4, zorder=1, linestyle=linestyle_list[i])
            sns.scatterplot(ax=ax1, x=fitterSG.distances[0:24], y=fitterSG.densities[0:24], color=col_pal_sg, legend=False, s=40,
                            edgecolor="k", linewidth=1, zorder=3, clip_on=False, marker=marker_list[i])
            ax1.errorbar(x=fitterSG.distances, y=fitterSG.densities, yerr=fitterSG.errors, fmt=".", color=col_pal_sg,
                         zorder=2)

            ax1.plot(fitterProtein.fit_x, fitterProtein.fit_rho, color=col_pal_protein, label=label_protein, linewidth=4,
                     zorder=1, linestyle=linestyle_list[i])
            sns.scatterplot(ax=ax1, x=fitterProtein.distances[0:24], y=fitterProtein.densities[0:24], color=col_pal_protein,
                            legend=False,
                            s=40, edgecolor="k", linewidth=1, zorder=3, clip_on=False, marker=marker_list[i])
            ax1.errorbar(fitterProtein.distances, fitterProtein.densities, yerr=fitterSG.errors, fmt=".",
                         color=col_pal_protein, zorder=2)

            ax1.plot(fitterRNA.fit_x, fitterRNA.fit_rho, color=col_pal_rna, label=label_rna, linewidth=4, zorder=2, linestyle=linestyle_list[i])
            sns.scatterplot(ax=ax1, x=fitterRNA.distances[0:24], y=fitterRNA.densities[0:24], color=col_pal_rna, legend=False,
                            s=40,
                            edgecolor="k", linewidth=1, zorder=3, clip_on=False, marker=marker_list[i])
            ax1.errorbar(fitterRNA.distances, fitterRNA.densities, yerr=fitterRNA.errors, fmt=".", color=col_pal_rna,
                         zorder=2)

            if fitterSM is not None:
                ax2 = ax1.twinx()
                if "ND" not in sm:
                    col = col_pal_sm[0]
                else:
                    col = col_pal_sm[1]

                sns.scatterplot(ax=ax2, x=fitterSM.distances[0:24], y=fitterSM.densities[0:24], color=col, legend=False, s=40,
                                edgecolor="k", linewidth=1, zorder=3, clip_on=False, marker=marker_list[i])
                ax2.errorbar(fitterSM.distances, fitterSM.densities, yerr=fitterSM.errors, fmt=".", color=col, zorder=2)
                sns.lineplot(ax=ax2, x=fitterSM.fit_x, y=fitterSM.fit_rho, color=col, label=label_sm, linewidth=4, zorder=1, linestyle=linestyle_list[i])

                ax2.get_legend().remove()
                ax2.tick_params(left=False, right=True, top=False, bottom=False, labelbottom=False, direction='in',
                                length=4,
                                width=2)
                ax2.set_ylim(0.0, 0.4)

                ax1.tick_params(left=True, right=False, top=True, bottom=False, labelbottom=True, direction='in',
                                length=4,
                                width=2)
                ax2.spines['bottom'].set_visible(False)
                ax2.spines['left'].set_visible(False)

            else:
                ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                                length=4,
                                width=2)
                ax2=None

        return fig, ax1, ax2


    path = sys.argv[1]
    dt = int(sys.argv[2])
    start = int(sys.argv[3])
    end = int(sys.argv[4])

    df_sg = cluster_ave(start, end, dt, "SG", "sg_X")
    timestep_sg = np.array(df_sg['Timestep'])
    phi_sg = np.array(df_sg['Phi'])
    n_cluster = np.array(df_sg['Number of Droplets'])

    dsm_list = ["dsm_anisomycin", "dsm_daunorubicin", "dsm_dihydrolipoicacid", "dsm_hydroxyquinoline", "dsm_lipoamide",
                "dsm_lipoicacid", "dsm_mitoxantrone", "dsm_pararosaniline", "dsm_pyrivinium", "dsm_quinicrine"]

    phi_list = []
    n_cluster_list = []

    for dsm in dsm_list:
        df = cluster_ave(start, end, dt, "SG", dsm)
        phi_list.append(df['Phi'])
        n_cluster_list.append(np.array(df['Number of Droplets']))

    phi_arr = np.array(phi_list)
    phi_mean_dsm = np.mean(phi_arr, axis=0)
    phi_sem_dsm = sem(phi_arr, axis=0)


    n_cluster_arr = np.array(n_cluster_list)
    n_cluster_mean_dsm = np.mean(n_cluster_arr, axis=0)
    n_cluster_sem_dsm = sem(n_cluster_arr, axis=0)

    ndsm_list = ["ndsm_dmso", "ndsm_valeric", "ndsm_ethylenediamine", "ndsm_propanedithiol",
                 "ndsm_hexanediol", "ndsm_diethylaminopentane", "ndsm_aminoacridine", "ndsm_anthraquinone",
                 "ndsm_acetylenapthacene", "ndsm_anacardic"]

    phi_list = []
    n_cluster_list = []

    for ndsm in ndsm_list:
        df = cluster_ave(start, end, dt, "SG", ndsm)
        phi_list.append(df['Phi'])
        n_cluster_list.append(np.array(df['Number of Droplets']))

    phi_arr = np.array(phi_list)
    phi_mean_ndsm = np.mean(phi_arr, axis=0)
    phi_sem_ndsm = sem(phi_arr, axis=0)

    n_cluster_arr = np.array(n_cluster_list)
    n_cluster_mean_ndsm = np.mean(n_cluster_arr, axis=0)
    n_cluster_sem_ndsm = sem(n_cluster_arr, axis=0)

    fig, ax1 = plot_cluster(timestep_sg, phi_sg, phi_mean_dsm, phi_sem_dsm, phi_mean_ndsm, phi_sem_ndsm)
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0.5, 1)
    leg = plt.figlegend(loc='lower right', ncol=1, bbox_to_anchor=(0.7, 0.1, 0.2, 0.85))
    leg.get_frame().set_alpha(0)
    plt.savefig(f"ANALYSIS_AVE_TIME/TIME_PHI.png", format="png", dpi=400)

    fig, ax1 = plot_cluster(timestep_sg, n_cluster, n_cluster_mean_dsm, n_cluster_sem_dsm, n_cluster_mean_ndsm, n_cluster_sem_ndsm)
    #ax1.set_xlim(0, 2)
    #ax1.set_ylim(0.5, 1)
    leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
    leg.get_frame().set_alpha(0)
    plt.savefig(f"ANALYSIS_AVE_TIME/TIME_CLUSTER.png", format="png", dpi=400)


    df_1_dict = {}
    df_2_dict = {}
    df_3_dict = {}

    df_1_dict["SG"] = rdp_ave(0, 250, 50, "SG", "sg_X")
    df_1_dict["Protein"] = rdp_ave(0, 250, 50, "Protein", "sg_X")
    df_1_dict["RNA"] = rdp_ave(0, 250, 50, "RNA", "sg_X")

    df_2_dict["SG"] = rdp_ave(850, 1100, 50, "SG", "sg_X")
    df_2_dict["Protein"] = rdp_ave(850, 1100, 50, "Protein", "sg_X")
    df_2_dict["RNA"] = rdp_ave(850, 1100, 50, "RNA", "sg_X")

    df_3_dict["SG"] = rdp_ave(1700, 1950, 50, "SG", "sg_X")
    df_3_dict["Protein"] = rdp_ave(1700, 1950, 50, "Protein", "sg_X")
    df_3_dict["RNA"] = rdp_ave(1700, 1950, 50, "RNA", "sg_X")

    df_list = [df_1_dict, df_2_dict, df_3_dict]
    time_list = ["0.0-0.3 $mu$s", "0.85-1.15$mu$s", "1.7-2.0$mu$s"]
    fig, ax1, ax2 = plot_rdp(df_list, time_list, sm="sg_X")
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 800)
    #leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
    #leg.get_frame().set_alpha(0)
    plt.savefig(f"ANALYSIS_AVE_TIME/TIME_SG_RDP.png", format="png", dpi=400)

    df_1_dict_dsm_list = []
    df_2_dict_dsm_list = []
    df_3_dict_dsm_list = []

    for dsm in dsm_list:
        df_1_dict = {}
        df_2_dict = {}
        df_3_dict = {}

        df_1_dict["SG"] = rdp_ave(0, 250, 50, "SG", dsm)
        df_1_dict["Protein"] = rdp_ave(0, 250, 50, "Protein", dsm)
        df_1_dict["RNA"] = rdp_ave(0, 250, 50, "RNA", dsm)
        df_1_dict["SM"] = rdp_ave(0, 250, 50, "SM", dsm)

        df_1_dict_dsm_list.append(df_1_dict)

        df_2_dict["SG"] = rdp_ave(850, 1100, 50, "SG", dsm)
        df_2_dict["Protein"] = rdp_ave(850, 1100, 50, "Protein", dsm)
        df_2_dict["RNA"] = rdp_ave(850, 1100, 50, "RNA", dsm)
        df_2_dict["SM"] = rdp_ave(850, 1100, 50, "SM", dsm)
        df_2_dict_dsm_list.append(df_2_dict)

        df_3_dict["SG"] = rdp_ave(1700, 1950, 50, "SG", dsm)
        df_3_dict["Protein"] = rdp_ave(1700, 1950, 50, "Protein", dsm)
        df_3_dict["RNA"] = rdp_ave(1700, 1950, 50, "RNA", dsm)
        df_3_dict["SM"] = rdp_ave(1700, 1950, 50, "SM", dsm)
        df_3_dict_dsm_list.append(df_3_dict)

    # Initialize a dictionary to store the averaged DataFrames
    averaged_dfs_1 = {}
    averaged_dfs_2 = {}
    averaged_dfs_3 = {}

    # Extract the keys (assuming all dictionaries have the same keys)
    keys = df_1_dict_dsm_list[0].keys()

    # Iterate over each key
    for key in keys:
        # Concatenate DataFrames corresponding to the current key from all dictionaries
        concatenated_df_1 = pd.concat([d[key] for d in df_1_dict_dsm_list])
        concatenated_df_2 = pd.concat([d[key] for d in df_2_dict_dsm_list])
        concatenated_df_3 = pd.concat([d[key] for d in df_3_dict_dsm_list])

        # Group by the index and calculate the mean
        averaged_df_1 = concatenated_df_1.groupby(concatenated_df_1.index).mean()
        averaged_df_2 = concatenated_df_2.groupby(concatenated_df_2.index).mean()
        averaged_df_3 = concatenated_df_3.groupby(concatenated_df_3.index).mean()

        # Store the averaged DataFrame in the result dictionary
        averaged_dfs_1[key] = averaged_df_1
        averaged_dfs_2[key] = averaged_df_2
        averaged_dfs_3[key] = averaged_df_3

    print(averaged_dfs_1["SG"])
    print(averaged_dfs_2["SG"])
    print(averaged_dfs_3["SG"])

    df_list = [averaged_dfs_1, averaged_dfs_2, averaged_dfs_3]
    time_list = ["0.0-0.3 $mu$s", "0.85-1.15$mu$s", "1.7-2.0$mu$s"]
    fig, ax1, ax2 = plot_rdp(df_list, time_list, sm="DSM")
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 800)
    # leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
    # leg.get_frame().set_alpha(0)
    plt.savefig(f"ANALYSIS_AVE_TIME/TIME_DSM_RDP.png", format="png", dpi=400)

    df_1_dict_ndsm_list = []
    df_2_dict_ndsm_list = []
    df_3_dict_ndsm_list = []

    for ndsm in ndsm_list:
        df_1_dict = {}
        df_2_dict = {}
        df_3_dict = {}

        df_1_dict["SG"] = rdp_ave(0, 250, 50, "SG", ndsm)
        df_1_dict["Protein"] = rdp_ave(0, 250, 50, "Protein", ndsm)
        df_1_dict["RNA"] = rdp_ave(0, 250, 50, "RNA", ndsm)
        df_1_dict["SM"] = rdp_ave(0, 250, 50, "SM", ndsm)

        df_1_dict_ndsm_list.append(df_1_dict)

        df_2_dict["SG"] = rdp_ave(850, 1100, 50, "SG", ndsm)
        df_2_dict["Protein"] = rdp_ave(850, 1100, 50, "Protein", ndsm)
        df_2_dict["RNA"] = rdp_ave(850, 1100, 50, "RNA", ndsm)
        df_2_dict["SM"] = rdp_ave(850, 1100, 50, "SM", ndsm)
        df_2_dict_ndsm_list.append(df_2_dict)

        df_3_dict["SG"] = rdp_ave(1700, 1950, 50, "SG", ndsm)
        df_3_dict["Protein"] = rdp_ave(1700, 1950, 50, "Protein", ndsm)
        df_3_dict["RNA"] = rdp_ave(1700, 1950, 50, "RNA", ndsm)
        df_3_dict["SM"] = rdp_ave(1700, 1950, 50, "SM", ndsm)
        df_3_dict_ndsm_list.append(df_3_dict)

    # Initialize a dictionary to store the averaged DataFrames
    averaged_dfs_1 = {}
    averaged_dfs_2 = {}
    averaged_dfs_3 = {}

    # Extract the keys (assuming all dictionaries have the same keys)
    keys = df_1_dict_ndsm_list[0].keys()

    # Iterate over each key
    for key in keys:
        # Concatenate DataFrames corresponding to the current key from all dictionaries
        concatenated_df_1 = pd.concat([d[key] for d in df_1_dict_ndsm_list])
        concatenated_df_2 = pd.concat([d[key] for d in df_2_dict_ndsm_list])
        concatenated_df_3 = pd.concat([d[key] for d in df_3_dict_ndsm_list])

        # Group by the index and calculate the mean
        averaged_df_1 = concatenated_df_1.groupby(concatenated_df_1.index).mean()
        averaged_df_2 = concatenated_df_2.groupby(concatenated_df_2.index).mean()
        averaged_df_3 = concatenated_df_3.groupby(concatenated_df_3.index).mean()

        # Store the averaged DataFrame in the result dictionary
        averaged_dfs_1[key] = averaged_df_1
        averaged_dfs_2[key] = averaged_df_2
        averaged_dfs_3[key] = averaged_df_3

    print(averaged_dfs_1["SG"])
    print(averaged_dfs_2["SG"])
    print(averaged_dfs_3["SG"])

    df_list = [averaged_dfs_1, averaged_dfs_2, averaged_dfs_3]
    time_list = ["0.0-0.3 $mu$s", "0.85-1.15$mu$s", "1.7-2.0$mu$s"]
    fig, ax1, ax2 = plot_rdp(df_list, time_list, sm="NDSM")
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 800)
    leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
    leg.get_frame().set_alpha(0)
    plt.savefig(f"ANALYSIS_AVE_TIME/TIME_NDSM_RDP.png", format="png", dpi=400)