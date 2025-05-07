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

class Average:
    def __init__(self, path, category, tmin, tmax, dt):
        self.path=path
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

    def rdf_ave(self, sm):
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df = pd.read_csv("{}/RDF_{}_{}.csv".format(category, sm, self.tmin))
        for i in range(self.tmin+self.dt,self.tmax,self.dt):
            try:
                df_temp = pd.read_csv("{}/RDF_{}_{}.csv".format(category, sm, i))
                df = pd.concat([df,df_temp])
            except:
                print("{}/RDF_{}_{}.csv".format(category, sm, str(i)))

        df_mean = df.groupby(level=0).mean()
        df_mean.to_csv("{}/{}_AVE/RDF_{}.csv".format(self.path, self.category, sm), index=False)


    def contact_ave(self, biopolymer, sm):
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df = pd.read_csv("{}/{}_Contacts_Count_{}_{}.csv".format(category, biopolymer, sm, str(self.tmin)), header=None)
        count = 1
        for i in range(int(self.tmin+self.dt),self.tmax,self.dt):
            try:
                df_temp = pd.read_csv("{}/{}_Contacts_Count_{}_{}.csv".format(category, biopolymer, sm, str(i)), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}_Contacts_Count_{}_{}.csv.csv File Missing".format(biopolymer, sm, str(i)))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/{}_AVE/{}_Contacts_Mean_{}.csv".format(self.path,self.category, biopolymer, sm), index=False, header=False)

    def contact_sm_ave(self, biopolymer, sm):
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df = pd.read_csv("{}/{}_SM_Contacts_Count_{}_{}.csv".format(category, biopolymer, sm, self.tmin), header=None)

        count = 1
        for i in range(int(self.tmin+self.dt),self.tmax,self.dt):
            try:
                df_temp = pd.read_csv("{}/{}_SM_Contacts_Count_{}_{}.csv".format(category, biopolymer, sm, str(i)), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}_SM_Contacts_Count_{}_{}.csv.csv File Missing".format(biopolymer, sm, str(i)))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/{}_AVE/{}_SM_Contacts_Mean_{}.csv".format(self.path,self.category, biopolymer, sm), index=False, header=False)

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
        df.to_csv("{}/{}_AVE/Cluster_{}_{}.csv".format(self.path,self.category, biopolymer, sm), index=False, header=True)

    def diffusivity_ave(self, sm, n_prot):
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df_list = []
        index_list = []
        for i in range(self.tmin,self.tmax,self.dt):
            for j in range(n_prot):
                try:
                    df_temp = pd.read_csv("{}/G3BP1_Diffusivity_{}_{}_{}.csv".format(category, sm, str(i), str(j)))
                    df_list.append(df_temp)
                    index_list.append(i)
                except:
                    print("G3BP1_Diffusivity_{}_{}.csv File Missing".format(sm, str(i)))

        df = (
            # combine dataframes into a single dataframe
            pd.concat(df_list)
                # replace 0 values with nan to exclude them from mean calculation
                .reset_index()
                # group by the row within the original dataframe
                .groupby("index")
                # calculate the mean
                .mean()
        )
        df_sem = (
            # combine dataframes into a single dataframe
            pd.concat(df_list)
                # replace 0 values with nan to exclude them from mean calculation
                .reset_index()
                # group by the row within the original dataframe
                .groupby("index")
                # calculate the mean
                .sem()
        )

        df["MSD_SEM"] = df_sem["MSD (um)"]
        df["Rg_SEM"] = df_sem["Rg"]
        df.to_csv("{}/{}_AVE/G3BP1_Diffusivity_{}.csv".format(self.path, self.category, sm), index=False, header=True)

    def pca_ave(self, biopolymer, sm):
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df_list = []
        index_list = []
        for i in range(self.tmin,self.tmax,self.dt):
            mean_values = pd.read_csv("{}/PCA_{}_{}_{}.csv".format(category, biopolymer, sm, str(i))).mean()
            df_list.append(mean_values)
            index_list.append(i)
        df = pd.DataFrame(df_list, index=index_list)
        df.to_csv("{}/{}_AVE/PCA_{}_{}.csv".format(self.path,self.category, biopolymer, sm), index=False, header=True)

    def collect_diffusion(self, prefix, sm):
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        if self.tmin > 0:
            tmin = self.tmin - self.dt
        else:
            tmin = self.tmin
        if self.tmax < 2000:
            tmax = self.tmax + self.dt
        else:
            tmax = self.tmax

        diff_list = list(np.arange(tmin,tmax,self.dt))

        for filename in os.listdir(category):
            # Check if the filename starts with the given prefix
            if "Diffusivity" in filename and "G3BP1" in filename:
                if int(filename.split("_")[4]) in diff_list:
                    if sm in filename:
                        source_file = os.path.join(category, filename)
                        dest_file = os.path.join("{}/{}_AVE/DIFFUSIVITY".format(self.path,self.category), filename)
                        # Copy the file
                        shutil.copy2(source_file, dest_file)


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

    def rdf_ave(self, sm_list):
        category = "ANALYSIS_{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/{}_AVE/RDF_{}.csv".format(self.path, category, sm_list[0]))
        for i in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/{}_AVE/RDF_{}.csv".format(self.path, category, i))
                df = pd.concat([df,df_temp])
            except:
                print("{}/{}_AVE/RDF_{}.csv".format(self.path, category, i))

        df_mean = df.groupby(level=0).mean()
        df_mean.to_csv("{}/ANALYSIS_{}_AGG/RDF_{}.csv".format(self.path, self.category, self.category), index=False)


    def contact_ave(self, biopolymer, sm_list):
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/{}_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/{}_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS_{}_Contacts_Mean_{}.csv File Missing".format(self.path, biopolymer, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AGG/{}_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=False)

    def contact_sm_ave(self, biopolymer, sm_list):
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/{}_SM_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/{}_SM_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS_{}_SM_Contacts_Mean_{}_{}.csv File Missing".format(self.path, self.category, biopolymer, sm))

        df = df.div(count, fill_value=0)
        df.to_csv("{}/ANALYSIS_{}_AGG/{}_SM_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=False)

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

    def pca_ave(self, biopolymer, sm_list):
        df_list = []
        index_list = []
        for sm in sm_list:
            mean_values = pd.read_csv("{}/ANALYSIS_{}_AVE/PCA_{}_{}.csv".format(self.path, self.category, biopolymer, sm)).mean()
            df_list.append(mean_values)
            index_list.append(sm)
        df = pd.DataFrame(df_list, index=index_list)
        df.to_csv("{}/ANALYSIS_{}_AGG/PCA_{}_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=True)

    def stress_ave(self, biopolymer, sm_list):
        df_list = []
        index_list = []
        for sm in sm_list:
            mean_values = pd.read_csv("{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv".format(self.path, self.category, sm)).mean()
            df_list.append(mean_values)
            index_list.append(sm)
        df = pd.DataFrame(df_list, index=index_list)
        df.to_csv("{}/ANALYSIS_{}_AGG/PCA_{}_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=True)

    def stress_ave(self, sm_list):
        df_xx = pd.DataFrame()
        df_yy = pd.DataFrame()
        df_zz = pd.DataFrame()
        df_xy = pd.DataFrame()
        df_xz = pd.DataFrame()
        df_yz = pd.DataFrame()
        df = pd.DataFrame()
        for sm in sm_list:
            time_col = (
            pd.read_csv("{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv".format(self.path, self.category, sm))[
                "Timestep"])
            Pxx_col = list(
                pd.read_csv("{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv".format(self.path, self.category, sm))[
                    "Pxx"])
            Pyy_col = list(
                pd.read_csv("{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv".format(self.path, self.category, sm))[
                    "Pyy"])
            Pzz_col = list(
                pd.read_csv("{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv".format(self.path, self.category, sm))[
                    "Pzz"])
            Pxy_col = list(
                pd.read_csv("{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv".format(self.path, self.category, sm))[
                    "Pxy"])
            Pxz_col = list(
                pd.read_csv("{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv".format(self.path, self.category, sm))[
                    "Pxz"])
            Pyz_col = list(
                pd.read_csv("{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv".format(self.path, self.category, sm))[
                    "Pyz"])
            df_xx[("Pxx_" + str(sm))] = Pxx_col
            df_yy[("Pyy_" + str(sm))] = Pyy_col
            df_zz[("Pzz_"+str(sm))] = Pzz_col
            df_xy[("Pxy_" + str(sm))] = Pxy_col
            df_xz[("Pxz_" + str(sm))] = Pxz_col
            df_yz[("Pyz_" + str(sm))] = Pyz_col

        pxx_avg = df_xx.mean(axis=1)
        pxx_sem = df_xx.sem(axis=1)
        pyy_avg = df_yy.mean(axis=1)
        pyy_sem = df_yy.sem(axis=1)
        pzz_avg = df_zz.mean(axis=1)
        pzz_sem = df_zz.sem(axis=1)
        pxy_avg = df_xy.mean(axis=1)
        pxy_sem = df_xy.sem(axis=1)
        pxz_avg = df_xz.mean(axis=1)
        pxz_sem = df_xz.sem(axis=1)
        pyz_avg = df_yz.mean(axis=1)
        pyz_sem = df_yz.sem(axis=1)

        df["Timestep"] = time_col
        df["Pxx"] = pxx_avg
        df["Pyy"] = pyy_avg
        df["Pzz"] = pzz_avg
        df["Pxy"] = pxy_avg
        df["Pxz"] = pxz_avg
        df["Pyz"] = pyz_avg

        df.to_csv("{}/ANALYSIS_{}_AGG/Stress_Tensor_{}.csv".format(self.path, self.category, self.category),
                  index=False)

    def collect_diffusion(self, category):
        source_file = "{}/ANALYSIS_{}_AVE/DIFFUSIVITY".format(self.path, self.category)
        dest_file = "{}/ANALYSIS_{}_AGG/DIFFUSIVITY".format(self.path, self.category)
        # Copy the file
        shutil.copytree(source_file, dest_file)




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

    def additive_conc(self, sm):
        category = sm.split("_")[0].upper()
        distances = np.array((pd.read_csv("{}/BIOPOLYMER_ANALYSIS_{}/Density_Profile_{}_{}.csv".format(self.path, category, "RNA", sm))[
            "Distance from center of mass (A)"]))

        rna_density = np.array(
            pd.read_csv("{}/BIOPOLYMER_ANALYSIS_{}/Density_Profile_{}_{}.csv".format(self.path, category, "RNA", sm))[
                "Protein density (mg/mL)"])
        rna_sme = np.array(
            pd.read_csv("{}/BIOPOLYMER_ANALYSIS_{}/Density_Profile_{}_{}.csv".format(self.path, category, "RNA", sm))[
                "Standard mean error"])

        protein_density = np.zeros(rna_density.shape)
        protein_sme = np.zeros(rna_sme.shape)

        df = pd.DataFrame()

        protein_list = ["G3BP1", "FUS", "PABP1", "TDP43", "TIA1", "TTP"]
        for protein in protein_list:
            try:
                density_col = np.array(
                    pd.read_csv("{}/BIOPOLYMER_ANALYSIS_{}/Density_Profile_{}_{}.csv".format(self.path, category, protein, sm))[
                        "Protein density (mg/mL)"])
                error_col = np.array(
                    pd.read_csv("{}/BIOPOLYMER_ANALYSIS_{}/Density_Profile_{}_{}.csv".format(self.path, category, protein, sm))[
                        "Standard mean error"])
                protein_density = protein_density + density_col
                protein_sme = protein_sme + error_col ** 2
            except:
                print("Density_Profile_{}_{}.csv File Missing".format(protein, sm))

        protein_sme = np.sqrt(protein_sme)

        sg_density = protein_density + rna_density
        sg_sme = np.sqrt(protein_sme ** 2 + rna_sme ** 2)

        df_sg = pd.DataFrame()
        df_sg["Distance from center of mass (A)"] = distances
        df_sg["Protein density (mg/mL)"] = sg_density
        df_sg["Standard mean error"] = sg_sme
        df_sg.to_csv("{}/BIOPOLYMER_SUMMARY/Density_Profile_SG_{}.csv".format(self.path, sm), index=False)

        df_sg = pd.DataFrame()
        df_sg["Distance from center of mass (A)"] = distances
        df_sg["Protein density (mg/mL)"] = protein_density
        df_sg["Standard mean error"] = protein_sme
        df_sg.to_csv("{}/BIOPOLYMER_SUMMARY/Density_Profile_Protein_{}.csv".format(self.path, sm), index=False)

        df_sg = pd.DataFrame()
        df_sg["Distance from center of mass (A)"] = distances
        df_sg["Protein density (mg/mL)"] = rna_density
        df_sg["Standard mean error"] = rna_sme
        df_sg.to_csv("{}/BIOPOLYMER_SUMMARY/Density_Profile_RNA_{}.csv".format(self.path, sm), index=False)


if __name__ == '__main__':

    def gen_path(path, folder):
        full_path = "{}/{}".format(path,folder)
        if not os.path.exists(full_path):
            # Create the folder
            os.makedirs(full_path)
        else:
            shutil.rmtree(full_path)
            os.makedirs(full_path)

    def gen_avg(path, dt, start, end, dsm_names, ndsm_names):
        if not os.path.exists(path):
            # Create the folder
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
            pass

        # SG
        print("SG BLOCK AVERAGES")
        folder = "ANALYSIS_SG_AVE"
        gen_path(path, folder)

        ave = Average(path,"ANALYSIS_SG", start, end, dt)

        ave.rdp_ave("Protein", "sg_X")
        ave.rdp_ave("RNA", "sg_X")
        ave.rdp_ave("SG", "sg_X")

        ave.rdf_ave("sg_X")

        ave.contact_ave("Acid", "sg_X")
        ave.contact_ave("Residue", "sg_X")

        ave.cluster_ave("RNA", "sg_X")
        ave.cluster_ave("Protein", "sg_X")
        ave.cluster_ave("SG", "sg_X")

        ave.pca_ave("SG", "sg_X")
        ave.pca_ave("Protein", "sg_X")
        ave.pca_ave("RNA", "sg_X")

        shutil.copyfile('ANALYSIS_{}/Stress_Tensor_{}_0.csv'.format("SG", "sg_X"),
                        '{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv'.format(path,"SG", "sg_X"))

        gen_path(path, "{}/DIFFUSIVITY".format(folder))

        ave.collect_diffusion("G3BP1", "sg_X")




        # DSM
        category = "DSM"
        folder = "ANALYSIS_DSM_AVE"
        gen_path(path, folder)

        ave = Average(path, "ANALYSIS_DSM", start, end, dt)

        print("DSM BLOCK AVERAGES")

        gen_path(path, "{}/DIFFUSIVITY".format(folder))

        for i in tqdm(dsm_names):
            ave.rdp_ave("Protein", i)
            ave.rdp_ave("RNA", i)
            ave.rdp_ave("SG", i)
            ave.rdp_ave("SM", i)

            ave.rdf_ave(i)

            ave.contact_ave("Acid", i)
            ave.contact_ave("Residue", i)
            ave.contact_sm_ave("Residue", i)
            ave.contact_sm_ave("Acid", i)

            ave.cluster_ave("RNA", i)
            ave.cluster_ave("Protein", i)
            ave.cluster_ave("SG", i)

            ave.pca_ave("SG", i)
            ave.pca_ave("Protein", i)
            ave.pca_ave("RNA", i)

            ave.collect_diffusion("G3BP1",i)

            shutil.copyfile('ANALYSIS_{}/Stress_Tensor_{}_0.csv'.format(i.split("_")[0].upper(), i), '{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv'.format(path,category, i))

        print("DSM AGGREGATE")
        folder = "ANALYSIS_DSM_AGG"
        gen_path(path, folder)

        ave = Aggregate(path,"DSM")

        ave.rdp_ave("SG", dsm_names)
        ave.rdp_ave("RNA", dsm_names)
        ave.rdp_ave("Protein", dsm_names)
        ave.rdp_ave("SM", dsm_names)

        ave.rdf_ave(dsm_names)

        ave.contact_ave("Acid", dsm_names)
        ave.contact_ave("Residue", dsm_names)
        ave.contact_sm_ave("Residue", dsm_names)
        ave.contact_sm_ave("Acid", dsm_names)

        ave.cluster_ave("RNA", dsm_names)
        ave.cluster_ave("Protein", dsm_names)
        ave.cluster_ave("SG", dsm_names)

        ave.pca_ave("SG", dsm_names)
        ave.pca_ave("Protein", dsm_names)
        ave.pca_ave("RNA", dsm_names)

        ave.stress_ave(dsm_names)

        ave.collect_diffusion(dsm_names[0])




        # NDSM
        category = "NDSM"
        folder = "ANALYSIS_NDSM_AVE"
        gen_path(path, folder)

        ave = Average(path, "ANALYSIS_NDSM", start, end, dt)

        print("NDSM BLOCK AVERAGES")

        gen_path(path, "{}/DIFFUSIVITY".format(folder))

        for i in tqdm(ndsm_names):
            ave.rdp_ave("Protein", i)
            ave.rdp_ave("RNA", i)
            ave.rdp_ave("SG", i)
            ave.rdp_ave("SM", i)

            ave.rdf_ave(i)

            ave.contact_ave("Acid", i)
            ave.contact_ave("Residue", i)
            ave.contact_sm_ave("Residue", i)
            ave.contact_sm_ave("Acid", i)

            ave.cluster_ave("RNA", i)
            ave.cluster_ave("Protein", i)
            ave.cluster_ave("SG", i)

            ave.pca_ave("SG", i)
            ave.pca_ave("Protein", i)
            ave.pca_ave("RNA", i)

            ave.collect_diffusion("G3BP1",i)

            shutil.copyfile('ANALYSIS_{}/Stress_Tensor_{}_0.csv'.format(i.split("_")[0].upper(), i),
                            '{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv'.format(path,category, i))

        print("NDSM AGGREGATE")
        folder = "ANALYSIS_NDSM_AGG"
        gen_path(path, folder)
        ave = Aggregate(path, "NDSM")

        ave.rdp_ave("SG", ndsm_names)
        ave.rdp_ave("RNA", ndsm_names)
        ave.rdp_ave("Protein", ndsm_names)
        ave.rdp_ave("SM", ndsm_names)

        ave.rdf_ave(ndsm_names)

        ave.contact_ave("Acid", ndsm_names)
        ave.contact_ave("Residue", ndsm_names)
        ave.contact_sm_ave("Residue", ndsm_names)
        ave.contact_sm_ave("Acid", ndsm_names)

        ave.cluster_ave("RNA", ndsm_names)
        ave.cluster_ave("Protein", ndsm_names)
        ave.cluster_ave("SG", ndsm_names)

        ave.pca_ave("SG", ndsm_names)
        ave.pca_ave("Protein", ndsm_names)
        ave.pca_ave("RNA", ndsm_names)

        ave.stress_ave(ndsm_names)

        ave.collect_diffusion(ndsm_names[0])


        # BIOPOLYMER
        gen_path(path, "BIOPOLYMER_SUMMARY")
        print("SG BIOPOLYMER AVERAGES")

        folder = "BIOPOLYMER_ANALYSIS_SG"
        gen_path(path, folder)
        sm = ["sg_X"]
        ave_bio = Average_Biopolymers(path, folder, start, end, dt)
        ave_bio.gen_ave(sm,"SG")
        ave_bio.gen_agg(sm, "SG")
        ave_bio.additive_conc("SG")

        print("DSM BIOPOLYMER AVERAGES")
        folder = "BIOPOLYMER_ANALYSIS_DSM"
        gen_path(path, folder)
        ave_bio_dsm = Average_Biopolymers(path, folder, start, end, dt)
        ave_bio_dsm.gen_ave(dsm_names, "DSM")

        print("NDSM BIOPOLYMER AVERAGES")
        folder = "BIOPOLYMER_ANALYSIS_NDSM"
        gen_path(path, folder)
        ave_bio_ndsm = Average_Biopolymers(path, folder, start, end, dt)
        ave_bio_ndsm.gen_ave(ndsm_names, "NDSM")

        ave_bio_dsm.gen_agg(dsm_names, "DSM")
        ave_bio_ndsm.gen_agg(ndsm_names, "NDSM")
        ave_bio_dsm.additive_conc("DSM")
        ave_bio_ndsm.additive_conc("NDSM")

        # SG
        shutil.copyfile('{}/ANALYSIS_SG_AVE/Density_Profile_Protein_sg_X.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_Protein_SG_AVG.csv'.format(path))
        shutil.copyfile('{}/ANALYSIS_SG_AVE/Density_Profile_RNA_sg_X.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_RNA_SG_AVG.csv'.format(path))
        shutil.copyfile('{}/ANALYSIS_SG_AVE/Density_Profile_SG_sg_X.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_SG_SG_AVG.csv'.format(path))

        # NDSM
        shutil.copyfile('{}/ANALYSIS_NDSM_AGG/Density_Profile_Protein_NDSM.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_Protein_NDSM_AVG.csv'.format(path))
        shutil.copyfile('{}/ANALYSIS_NDSM_AGG/Density_Profile_RNA_NDSM.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_RNA_NDSM_AVG.csv'.format(path))
        shutil.copyfile('{}/ANALYSIS_NDSM_AGG/Density_Profile_SG_NDSM.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_SG_NDSM_AVG.csv'.format(path))
        shutil.copyfile('{}/ANALYSIS_NDSM_AGG/Density_Profile_SM_NDSM.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_SM_NDSM.csv'.format(path))

        # DSM
        shutil.copyfile('{}/ANALYSIS_DSM_AGG/Density_Profile_Protein_DSM.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_Protein_DSM_AVG.csv'.format(path))
        shutil.copyfile('{}/ANALYSIS_DSM_AGG/Density_Profile_RNA_DSM.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_RNA_DSM_AVG.csv'.format(path))
        shutil.copyfile('{}/ANALYSIS_DSM_AGG/Density_Profile_SG_DSM.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_SG_DSM_AVG.csv'.format(path))
        shutil.copyfile('{}/ANALYSIS_DSM_AGG/Density_Profile_SM_DSM.csv'.format(path),
                        '{}/BIOPOLYMER_SUMMARY/Density_Profile_SM_DSM.csv'.format(path))

        gen_path(path, "FIGURES")
        gen_path(path, "IMAGES")
        gen_path(path, "RESULTS")
        gen_path(path, "BIOPOLYMER_SUMMARY/FIGURES")
        gen_path(path, "BIOPOLYMER_SUMMARY/IMAGES")
        gen_path(path, "BIOPOLYMER_SUMMARY/RESULTS")


    path = sys.argv[1]
    dt = int(sys.argv[2])
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    expt = sys.argv[5]
    path = "{}_{}_{}".format(path, start, end)

    if expt == "T":
        dsm_list = ["dsm_anisomycin", "dsm_daunorubicin", "dsm_dihydrolipoicacid", "dsm_hydroxyquinoline", "dsm_lipoamide",
               "dsm_lipoicacid", "dsm_mitoxantrone", "dsm_pararosaniline", "dsm_pyrivinium", "dsm_quinicrine"]

        ndsm_list = ["ndsm_dmso", "ndsm_valeric", "ndsm_ethylenediamine", "ndsm_propanedithiol",
                      "ndsm_hexanediol", "ndsm_diethylaminopentane", "ndsm_aminoacridine", "ndsm_anthraquinone",
                      "ndsm_acetylenapthacene", "ndsm_anacardic"]

        gen_avg(path, dt, start, end, dsm_list, ndsm_list)

    else:
        dsm_list = []
        with open('{}/dsm_list.txt'.format(path), 'r') as f:
            for i in f.readlines():
                dsm_list.append(i.strip())

        ndsm_list = []
        with open('{}/ndsm_list.txt'.format(path), 'r') as f:
            for i in f.readlines():
                ndsm_list.append(i.strip())

        path = "{}_THEORETICAL".format(path)
        gen_avg(path, dt, start, end, dsm_list, ndsm_list)






