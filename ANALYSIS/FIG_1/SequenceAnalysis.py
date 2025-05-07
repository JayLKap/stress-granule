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
        print(name)
        
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
        
        plt.savefig("SEQUENCES/FIGURES/{}_PRDOS.png".format(name), format="png", dpi=400)
        
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
        
        fig, axs = plt.subplots(figsize=(3.3, 1), tight_layout=True)
        
        cols = sns.color_palette(palette="hls", n_colors=8)
        col_pal = [cols[0], cols[2], cols[4], cols[6]]
        
        g1 = sns.barplot(ax=axs, data=df, x="AA", y="Number", hue="Class", palette=col_pal, saturation=100, width = 0.6, dodge=False, edgecolor="k")
        
        #sns.move_legend(axs, "upper right", ncol=1, title=None, frameon=False)
        axs.get_legend().remove()
        
        g1.set(xlabel=None)
        g1.set(ylabel=None)
        
        maximum = df["Number"].loc[df["Number"].idxmax()]
        max_val = maximum+20-maximum%20
        
        axs.set_ylim(0, 80)
        plt.yticks(np.arange(0, 81, 20))
       
        axs.tick_params(left=True, right=True, top=True, bottom=False, labelbottom=True, direction='in', length=4, width=2)
        
        plt.savefig("SEQUENCES/FIGURES/{}_Histogram.png".format(name), format="png", dpi=400)


# NA Distribution
print("RNA")
file = "SEQUENCES/RNA.txt"
with open(file, "r") as f:
    nucleic_acids = f.read().replace('\n', '')
    
    
aa_dict = {
    'A': 0,
    'U': 0,
    'C': 0,
    'G': 0
    }

for i in nucleic_acids:
    if i == "T":
        i = "U"
        
    aa_dict[i.strip()] += 1
    acid_dict["N"+i.strip()] += 22
    
acids = aa_dict.keys()
num = aa_dict.values()

df = pd.DataFrame({"AA": acids, "Number": num})

fig, axs = plt.subplots(figsize=(6.72, 1.2), tight_layout=True)

cols = sns.color_palette(palette="Set2", n_colors=8)


g1 = sns.barplot(ax=axs, data=df, x="AA", y="Number", color=cols[7], saturation=100, width = 0.8, dodge=False, edgecolor="k")

g1.set(xlabel=None)
g1.set(ylabel=None)

axs.set_ylim(0, 400)
plt.yticks(np.arange(0, 401, 100))

axs.tick_params(left=True, right=True, top=True, bottom=False, labelbottom=True, direction='in', length=4, width=2)

#axs.set_ylim(0, 120)


plt.savefig("SEQUENCES/FIGURES/RNA_Histogram.png", format="png", dpi=400)

 



