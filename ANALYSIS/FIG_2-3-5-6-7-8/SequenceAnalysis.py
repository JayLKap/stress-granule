#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:27:50 2023

@author: jaykaplan
"""


import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

acid_dict = {
    'R': 0,
    'H': 0,
    'K': 0,
    'D': 0,
    'E': 0,
    
    'S': 0,
    'T': 0,
    'N': 0,
    'Q': 0,
    
    'C': 0,
    'G': 0,
    'P': 0,
    'A': 0,
    'V': 0,
    'I': 0,
    'L': 0,
    'M': 0,
    
    'F': 0,
    'Y': 0,
    'W': 0,
    
    'NA':0,
    'NU':0,
    'NC':0,
    'NG':0
    }


# SEQUENCES
df_proteins = pd.DataFrame(columns=[""])

for file in os.scandir("SEQUENCES"):
    if "PRDOS" in file.name:
        df = pd.read_csv(file)
        name = file.name.split(".")[0].split("_")[0]
       
        
        afont = {'fontname':'Arial'}
        
        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=10)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)
        
        # Disorder
        fig, axs = plt.subplots(figsize=(2.7, 0.96), tight_layout=True)
        g1 = sns.lineplot(ax=axs, x=df.iloc[:,0], y=df.iloc[:,3], linewidth=1, markers=True, dashes=False)
        
        straight_line = np.divide(np.ones(len(df.iloc[:,0])),2)
        
        g2 = plt.plot(df.iloc[:,0], straight_line, "k--", linewidth=2)
        
        plt.yticks(np.arange(0, 1+0.1, 0.5))
        
        plt.xticks(np.arange(0, len(df.iloc[:,0]), 100))
        
        g1.set(xlabel=None)
        g1.set(ylabel=None)
        
        axs.set_ylim(0, 1)
       
        axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
        
        plt.xlim(0, len(df.iloc[:,0]))
        
        plt.savefig("SEQUENCES/FIGURES/{}.png".format(name), format="png", dpi=400)
        
        plt.show()
        
        nRes = {
            "TDP43": 32,
            "TTP": 16,
            "TIA1": 16,
            "PABP1": 16,
            "G3BP1": 16,
            "FUS": 16
            }
        



        # AA Distribution
        aa_dict = {
            'R': 0,
            'H': 0,
            'K': 0,
            'D': 0,
            'E': 0,
            
            'S': 0,
            'T': 0,
            'N': 0,
            'Q': 0,
            
            'C': 0,
            'G': 0,
            'P': 0,
            'A': 0,
            'V': 0,
            'I': 0,
            'L': 0,
            'M': 0,
            
            'F': 0,
            'Y': 0,
            'W': 0
            }
        
        
        total = 0
        
        for i in df.iloc[:, 1]:
            aa_dict[i.strip()] += 1
            acid_dict[i.strip()] += nRes[name]
            total+=1
        

            
        acids = aa_dict.keys()
        
        num = aa_dict.values()
        
        classify = [
            "Electrostatic",
            "Electrostatic",
            "Electrostatic",
            "Electrostatic",
            "Electrostatic",
            
            "Polar",
            "Polar",
            "Polar",
            "Polar",
            
            "Hydrophobic",
            "Hydrophobic",
            "Hydrophobic",
            "Hydrophobic",
            "Hydrophobic",
            "Hydrophobic",
            "Hydrophobic",
            "Hydrophobic",
            
            "Aromatic",
            "Aromatic",
            "Aromatic"          
            ]
        
        
        df = pd.DataFrame({"AA": acids, "Number": num, "Class": classify})
        
        fig, axs = plt.subplots(figsize=(3.3, 1.24), tight_layout=True)
        
        cols = sns.color_palette(palette="hls", n_colors=8)
        col_pal = [cols[0], cols[2], cols[4], cols[6]]
        
        g1 = sns.barplot(ax=axs, data=df, x="AA", y="Number", hue="Class", palette=col_pal, saturation=100, width = 0.6, dodge=False, edgecolor="k")
        
        #sns.move_legend(axs, "upper right", ncol=1, title=None, frameon=False)
        axs.get_legend().remove()
        
        g1.set(xlabel=None)
        g1.set(ylabel=None)
        
        maximum = df["Number"].loc[df["Number"].idxmax()]
        print(maximum)
        max_val = maximum+20-maximum%20
        print(max_val)
        
        axs.set_ylim(0, 80)
        plt.yticks(np.arange(0, 81, 20))
       
        axs.tick_params(left=True, right=True, top=True, bottom=False, labelbottom=True, direction='in', length=4, width=2)
        
        plt.savefig("SEQUENCES/FIGURES/{}_Histogram.png".format(file.name), format="png", dpi=400)
        
        plt.show()

#%%
file = "SEQUENCES/{}_PRDOS.csv".format("FUS")
df = pd.read_csv(file)


afont = {'fontname':'Arial'}

sns.set_theme(style="ticks")
sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=10)  # fontsize of the axes title
plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
plt.rc('legend', fontsize=10)  # legend fontsize
plt.rc('font', size=10)  # controls default text sizes
plt.rc('axes', linewidth=2)

# AA Distribution
aa_dict = {
    'R': 0,
    'H': 0,
    'K': 0,
    'D': 0,
    'E': 0,
    'S': 0,
    'T': 0,
    'N': 0,
    'Q': 0,
    'C': 0,
    'G': 0,
    'P': 0,
    'A': 0,
    'V': 0,
    'I': 0,
    'L': 0,
    'M': 0,
    'F': 0,
    'Y': 0,
    'W': 0
    }
total = 0

for i in df.iloc[:, 1]:
    aa_dict[i.strip()] += 1

    
acids = aa_dict.keys()

num = aa_dict.values()

classify = [
    "Charged",
    "Charged",
    "Charged",
    "Charged",
    "Charged",
    "Polar",
    "Polar",
    "Polar",
    "Polar",
    "Special",
    "Special",
    "Special",
    "Hydrophobic",
    "Hydrophobic",
    "Hydrophobic",
    "Hydrophobic",
    "Hydrophobic",
    "Hydrophobic",
    "Hydrophobic",
    "Hydrophobic"          
    ]


df = pd.DataFrame({"AA": acids, "Number": num, "Class": classify})

fig, axs = plt.subplots(figsize=(3.4, 1.2), tight_layout=True)

cols = sns.color_palette(palette="Spectral", n_colors=20)

g1 = sns.barplot(ax=axs, data=df, x="AA", y="Number", hue="Class", palette="Blues", saturation=100, width = 0.6, dodge=False, edgecolor="k")

#sns.move_legend(axs, "upper right", ncol=1, title=None, frameon=False)
axs.get_legend().remove()

g1.set(xlabel=None)
g1.set(ylabel=None)

axs.tick_params(left=True, right=True, top=True, bottom=False, labelbottom=True, direction='in', length=4, width=2)
axs.set_ylim(0, 160)
plt.yticks(np.arange(0, 161, 40))

plt.savefig("SEQUENCES/FIGURES/{}_Histogram.png".format(file.name), format="png", dpi=400)

plt.show()






#%%

# NA Distribution

file = "Sequences/RNA.txt"
with open(file, "r") as f:
    nucleic_acids = f.read().replace('\n', '')
    
    
aa_dict = {
    'A': 0,
    'U': 0,
    'C': 0,
    'G': 0
    }

for i in nucleic_acids:
    print(i)
    if i == "T":
        i = "U"
        
    aa_dict[i.strip()] += 1
    acid_dict["N"+i.strip()] += 22
    
acids = aa_dict.keys()
num = aa_dict.values()

print(len(acids))
print(len(classify))
df = pd.DataFrame({"AA": acids, "Number": num})

fig, axs = plt.subplots(figsize=(6.7, 1.4), tight_layout=True)

cols = sns.color_palette(palette="Set2", n_colors=8)


g1 = sns.barplot(ax=axs, data=df, x="AA", y="Number", color=cols[7], saturation=100, width = 0.8, dodge=False, edgecolor="k")

g1.set(xlabel=None)
g1.set(ylabel=None)

axs.set_ylim(0, 400)
plt.yticks(np.arange(0, 401, 100))

axs.tick_params(left=True, right=True, top=True, bottom=False, labelbottom=True, direction='in', length=4, width=2)

#axs.set_ylim(0, 120)


plt.savefig("SEQUENCES/FIGURES/RNA_Histogram.png", format="png", dpi=400)

plt.show()
            


#%%

df = pd.read_csv("R2_Dataset_4.csv")  

afont = {'fontname':'Arial'}

sns.set_theme(style="ticks")
sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=10)  # fontsize of the axes title
plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
plt.rc('legend', fontsize=10)  # legend fontsize
plt.rc('font', size=10)  # controls default text sizes
plt.rc('axes', linewidth=2)

fig, axs = plt.subplots(figsize=(3.4, 2.4), tight_layout=True)
plt.ylim(0,1)
plt.xlim(0,16)
plt.xticks(np.arange(0, 17, 2))
sns.lineplot(ax=axs, data = df, x="x", y="R^2", linewidth=4, hue="Parameter", palette="rocket", zorder=1)
sns.scatterplot(ax=axs, data = df, x="x", y="R^2", hue="Parameter", palette="rocket", legend=False, s=60, edgecolor="k", linewidth=2, zorder=2)
axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
sns.move_legend(axs, "lower right", bbox_to_anchor=(0.94, 0.07), ncol=1, title=None, frameon=False)
plt.xlabel(None)
plt.ylabel(None)
plt.savefig("R2_Plot_4.png", format="png", dpi=400)
plt.show()

#%%
sns.set_theme(style="ticks")
sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=10)  # fontsize of the axes title
plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
plt.rc('legend', fontsize=8)  # legend fontsize
plt.rc('font', size=10)  # controls default text sizes
plt.rc('axes', linewidth=2)


df_params = pd.read_csv("parameters.csv")

print(df_params)

col_pal_1 = sns.color_palette(palette="rocket", n_colors=6)
col_pal_2 = sns.color_palette(palette="Blues", n_colors=6)


df = df_params[["Molecule", "E", "S", "V", "U", "R"]].groupby(by=["Molecule"]).mean()

print(df)

fig, axs = plt.subplots(figsize=(3.4, 2.4), tight_layout=True)

axs.tick_params(left=True, bottom=True)

axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)



r = np.linspace(4, 22, 100)

for mol in range(12):
    
    eps = df.iloc[mol,0]
    sig = df.iloc[mol,1]
    v = df.iloc[mol,2]
    mu = df.iloc[mol,3]
    rc = df.iloc[mol,4]
    
    
    alpha = 2 * v * np.power(rc / sig, 2 * mu) * np.power(
               ((1 + 2 * v) / (2 * v * (np.power(rc / sig, 2 * mu) - 1))),
               2 * v + 1)
    
    
    phi = eps * alpha * (np.power(sig / r, 2 * mu) - 1) * np.power(
        (np.power(rc / r, 2 * mu) - 1), 2 * v)
    
    print(phi)
    
    if mol<6:
        name = "D"+str(mol+1)
        plt.plot(r, phi, label=name, color=col_pal_2[mol], linewidth=1)
        
    else:
        name = "ND"+str(mol-5)
        plt.plot(r, phi, label=name, color=col_pal_1[mol-6], linewidth=1)


leg = plt.figlegend(loc='upper right', ncol=2, bbox_to_anchor=(0.9, 0.9))
leg.get_frame().set_alpha(0)
plt.ylim(-0.4,0.6)
plt.xlim(4,20)
plt.xticks(np.arange(4, 21, 4))

plt.savefig("MIX_PARAMS.png", format="png", dpi=400)
plt.show()

 



