import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


class Analyze:

    def __init__(self, seqFile):
        self.seqFile = seqFile
        self.initial_cluster = 1

    def read_file(self):
        with open(self.seqFile, 'r') as f:
            contents = f.readlines()
        contents = contents[3:]
        return contents

    def read_csv_file(self):
        df = pd.read_csv(self.csvFile)
        return np.mean(df["Total Chain Number"])

    def generate_data(self, name, species, effect, df, stepSize):
        lines = self.read_file()
        fileLength = len(lines)
        lineNum = 0
        biggest = 0
        timeStep = 0
        clusterNum = 0
        LargestNum = 0
        run_once = True
        total_atoms = 0

        while lineNum < fileLength:
            sum = 0
            if lineNum == fileLength:
                break
            if timeStep == 0:
                lineNum += 1
            while int(lines[lineNum].split()[0]) % stepSize != 0:
                if lineNum == fileLength - 1:
                    break
                line = lines[lineNum].split()
                Rg = float(line[1])

                if Rg > 0:
                    clusterNum += 1
                if Rg > biggest:
                    biggest = Rg

                NumAtoms = int(line[2])
                if NumAtoms > LargestNum:
                    LargestNum = NumAtoms
                sum += NumAtoms
                lineNum += 1

            if run_once:
                self.initial_cluster = clusterNum
                total_atoms = sum
                run_once = False

            d_cluster = (self.initial_cluster - clusterNum) / self.initial_cluster * 100
            lineNum += 1
            timeStep += stepSize * 200 / 10000000
            if species == "Small_Molecules":
                tempDF = pd.DataFrame([{'Small Molecule': str(name), 'Time': timeStep, 'Cluster': clusterNum, 'RG': biggest,
                                "Phi": LargestNum / total_atoms, "Compound Class": effect}])
                df = pd.concat([df, tempDF], ignore_index=True)
            elif species == "Percent_Proteins_RNA":
                lab = "$\phi_P={}$".format(name)
                tempDF = pd.DataFrame(
                    [{'Percent': str(name), 'Phi_Label': lab, 'Time': timeStep, 'Cluster': clusterNum, 'RG': biggest,
                     "Phi": LargestNum / total_atoms}])
                df = pd.concat([df, tempDF], ignore_index=True)
            else:
                tempDF = pd.DataFrame([{'Protein': str(name), 'Time': timeStep, 'Cluster': clusterNum, 'RG': biggest,
                                "Phi": LargestNum / total_atoms, "Compound Class": effect}])
                df = pd.concat([df, tempDF], ignore_index=True)
            biggest = 0
            LargestNum = 0
            clusterNum = 0
        return df


if __name__ == '__main__':
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=16)  # fontsize of the axes title
    plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('font', size=16)  # controls default text sizes
    # Figure Gen
    
    
    def gen_SM_plots(df, name, window, folder, start, end):
        
        col_pall = sns.color_palette("Blues_r", n_colors=3)
        fig, axs = plt.subplots(3, sharex=True, figsize=(12, 10), tight_layout=True)
        df["Rolling_Cluster"] = df["Cluster"].rolling(window).mean()
        df["Rolling_Phi"] = df["Phi"].rolling(window).mean()
        df["Rolling_RG"] = df["RG"].rolling(window).mean()
        sns.lineplot(ax=axs[0], data=df, x="Time", y="Rolling_Cluster", hue="Compound Class", linewidth=4, palette="Blues_r", marker='o')
        sns.lineplot(ax=axs[1], data=df, x="Time", y="Rolling_Phi", hue="Compound Class", linewidth=4, palette="Blues_r", marker='o', legend=False)
        sns.lineplot(ax=axs[2], data=df, x="Time", y="Rolling_RG", hue="Compound Class", linewidth=4, palette="Blues_r", marker='o', legend=False)
        sns.move_legend(axs[0], "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
        plt.xlabel('Time (ns)')
        axs[0].set_ylabel("$N_{D}$")
        axs[1].set_ylabel("$\phi_{D}$")
        axs[2].set_ylabel("$R_{g}$ $(\AA)$")
        axs[0].spines[['right', 'top']].set_visible(False)
        axs[1].spines[['right', 'top']].set_visible(False)
        axs[2].spines[['right', 'top']].set_visible(False)
        plt.xlim(start, end)
        plt.savefig(folder + "/FIGURES/" + folder + "_" + name + "_Rolling.png", format="png", dpi=400)
        plt.show()
        
        df_cluster_mean = df.loc[:,["Time", "Cluster", "RG", "Phi", "Compound Class"]].groupby("Compound Class").mean()
        df_cluster_std = df.loc[:,["Time", "Cluster", "RG", "Phi", "Compound Class"]].groupby("Compound Class").sem()
        
        print(df_cluster_mean)
        print(df_cluster_std)
        
        df_cluster_sg = pd.concat([df_cluster_mean, df_cluster_std], axis=1).reindex(df_cluster_mean.index)
        
        df_cluster_sg.to_csv("Small_Molecules_1uM/Cluster_SG_Analysis_1uM.csv")
        print(df_cluster_sg)
        
        
        df = df.sort_values(by=["Compound Class", "Small Molecule"])
        fig, axs = plt.subplots(3, sharex=True, figsize=(12, 8), tight_layout=True)
        plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
        g1 = sns.barplot(ax=axs[0], data=df, x="Small Molecule", y="Cluster", hue="Compound Class", palette="rocket", dodge=False, width=0.5, saturation=100)
        g2 = sns.barplot(ax=axs[1], data=df, x="Small Molecule", y="Phi", hue="Compound Class", palette="rocket", dodge=False, width=0.5, saturation=100)
        g3 = sns.barplot(ax=axs[2], data=df, x="Small Molecule", y="RG", hue="Compound Class", palette="rocket", dodge=False, width=0.5, saturation=100)
        sns.move_legend(axs[0], "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
        axs[1].legend([], [], frameon=False)
        axs[2].legend([], [], frameon=False)
        g1.set(xlabel=None)
        plt.xlabel('Small Molecule Class')
        axs[0].set_xlabel("")
        axs[0].set_ylabel("$N_{D}$")
        axs[1].set_xlabel("")
        axs[0].spines[['right', 'top']].set_visible(False)
        axs[1].spines[['right', 'top']].set_visible(False)
        axs[2].spines[['right', 'top']].set_visible(False)
        axs[1].set_ylabel("$\phi_{D}$")
        axs[1].set_ylim(0.4, 1)
        
        axs[1].yaxis.set_ticks(np.arange(0.4, 1.05, 0.1))
        axs[2].set_ylabel("$R_{g}$ $(\AA)$")
        plt.savefig(folder + "/FIGURES/" + folder + "_" + name + "_Bar.png", format="png", dpi=400)
        plt.show()

        df = df.sort_values(by=["Compound Class", "Small Molecule"])
        fig, axs = plt.subplots(3, sharex=True, figsize=(12, 8), tight_layout=True)
        plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
        g1 = sns.barplot(ax=axs[0], data=df, x="Compound Class", y="Cluster", hue="Compound Class", palette="rocket",
                         dodge=False, width=0.5, saturation=100)
        g2 = sns.barplot(ax=axs[1], data=df, x="Compound Class", y="Phi", hue="Compound Class", palette="rocket",
                         dodge=False, width=0.5, saturation=100)
        g3 = sns.barplot(ax=axs[2], data=df, x="Compound Class", y="RG", hue="Compound Class", palette="rocket",
                         dodge=False, width=0.5, saturation=100)
        sns.move_legend(axs[0], "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
        axs[1].legend([], [], frameon=False)
        axs[2].legend([], [], frameon=False)
        g1.set(xlabel=None)
        g2.set(xlabel=None)
        axs[0].set_xticks([])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("$N_{D}$")
        axs[1].set_xticks([])
        axs[1].set_xlabel("")
        axs[1].set_ylabel("$\phi_{D}$")
        axs[1].set_ylim(0.6, 1)
        axs[1].yaxis.set_ticks(np.arange(0.6, 1.05, 0.1))
        axs[0].spines[['right', 'top']].set_visible(False)
        axs[1].spines[['right', 'top']].set_visible(False)
        axs[2].spines[['right', 'top']].set_visible(False)
        axs[2].set_xticks([])
        axs[2].set_ylabel("$R_{g}$ $(\AA)$")
        plt.savefig(folder + "/FIGURES/" + folder + "_" + name + "_Avg_Bar.png", format="png", dpi=400)
        plt.show()
        
        
        # Phi
        fig, axs = plt.subplots(figsize=(6, 6), tight_layout=True)
        
        g1 = sns.barplot(ax=axs, data=df, x="Compound Class", y="Phi", hue="Compound Class", palette="Blues_r",
                         dodge=False, saturation=100, width = 0.5)

        
        axs.get_legend().remove()
        axs.set_ylabel("$\phi_{D}$")
        plt.xlabel("Small Molecule Class")
        axs.spines[['right', 'top']].set_visible(False)
       
        axs.tick_params(left=True, labelbottom=True)
        
        fig_name = "SG_Phi_Bar_Plot"
        
        plt.savefig("SG/FIGURES/{}.png".format(fig_name), format="png", dpi=400)
        
        plt.show()
        
        fig = plt.figure(figsize=(3, 6), tight_layout=True)
        
        
        # N
        fig, axs = plt.subplots(figsize=(6, 6), tight_layout=True)
        
        g1 = sns.barplot(ax=axs, data=df, x="Compound Class", y="Cluster", hue="Compound Class", palette="Blues_r",
                         dodge=False, saturation=100, width = 0.5)

        
        axs.get_legend().remove()
        axs.set_ylabel("$N_{D}$")
        plt.xlabel('Small Molecule Class')
        axs.spines[['right', 'top']].set_visible(False)
       
        axs.tick_params(left=True, labelbottom=True)
        
        fig_name = "SG_ND_Bar_Plot"
        
        plt.savefig("SG/FIGURES/{}.png".format(fig_name), format="png", dpi=400)
        
        plt.show()
        
        fig = plt.figure(figsize=(3, 6), tight_layout=True)
        
        # RG
        fig, axs = plt.subplots(figsize=(6, 6), tight_layout=True)
        
        g1 = sns.barplot(ax=axs, data=df, x="Compound Class", y="RG", hue="Compound Class", palette="Blues_r",
                         dodge=False, saturation=100, width = 0.5)

        
        axs.get_legend().remove()
        axs.set_ylabel("$R_{g}$ $(\AA)$")
        plt.xlabel('Small Molecule Class')
        axs.spines[['right', 'top']].set_visible(False)
       
        axs.tick_params(left=True, labelbottom=True)
        
        fig_name = "SG_RG_Bar_Plot"
        
        plt.savefig("SG/FIGURES/{}.png".format(fig_name), format="png", dpi=400)
        
        plt.show()
        
        fig = plt.figure(figsize=(3, 6), tight_layout=True)
        
        
        
        
    def gen_Percent_plots(df):
        afont = {'fontname':'Arial'}
        
        #custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=10)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)
        # Figure Gen
        
        # Phi
        fig, axs = plt.subplots(figsize=(3.42, 2.3), tight_layout=True)
        
        g1 = sns.barplot(ax=axs, data=df, x="Percent", y="Phi", hue="Percent", palette="Blues",
                         dodge=False, saturation=100, width = 0.5, errorbar='se', capsize=0.1, errcolor='k', errwidth=1, edgecolor="k")

        
        #axs.get_legend().remove()
        
        #axs.spines[['right', 'top']].set_visible(False)
        g1.set(xlabel=None)
        g1.set(ylabel=None)
       
        axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
        
        fig_name = "SG_Phi_Bar_Plot"
        
        axs.set_ylim(0, 1)
        
        plt.savefig("Percent_Proteins_RNA/FIGURES/{}.png".format(fig_name), format="png", dpi=400)
        
        
        plt.show()
                
        
        # N
        fig, axs = plt.subplots(figsize=(4.8, 3.2), tight_layout=True)
        
        g1 = sns.barplot(ax=axs, data=df, x="Percent", y="Cluster", hue="Percent", palette="Blues",
                         dodge=False, saturation=100, width = 0.5, errorbar='se', capsize=0.1, errcolor='k', errwidth=1, edgecolor="k")

        
        g1.set(xlabel=None)
        g1.set(ylabel=None)
        
        axs.get_legend().remove()
        
        
        
        #axs.spines[['right', 'top']].set_visible(False)
       
        axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
        
        fig_name = "SG_ND_Bar_Plot"
        
        axs.set_ylim(0, 80)
        
        plt.savefig("Percent_Proteins_RNA/FIGURES/{}.png".format(fig_name), format="png", dpi=400)
        
        plt.show()
        
        fig = plt.figure(figsize=(3, 6), tight_layout=True)
        
        
        # RG
        fig, axs = plt.subplots(figsize=(4.8, 3.2), tight_layout=True)
        
        g1 = sns.barplot(ax=axs, data=df, x="Percent", y="RG", hue="Percent", palette="Blues",
                         dodge=False, saturation=100, width = 0.5, errorbar='se', capsize=0.1, errcolor='k', errwidth=1, edgecolor="k")
        
        g1.set(xlabel=None)
        g1.set(ylabel=None)

        
        axs.get_legend().remove()
        #axs.spines[['right', 'top']].set_visible(False)
      
        axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
        
        axs.set_ylim(0, 8000)
        
        fig_name = "SG_RG_Bar_Plot"
        
        plt.savefig("Percent_Proteins_RNA/FIGURES/{}.png".format(fig_name), format="png", dpi=400)
        
        plt.show()
        
        fig = plt.figure(figsize=(3, 6), tight_layout=True)
        
    
    def gen_composition_plots(df):
        
        
        afont = {'fontname':'Arial'}
        
        #custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=10)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)
        # Figure Gen
        
        # Phi
                
        fig, axs = plt.subplots(figsize=(3.42, 2.58), tight_layout=True)
        
        cols = sns.color_palette(palette="Blues", n_colors=11)
        col_pal = [cols[5]]
        col_pal.append(cols[10])
        
        g1 = sns.barplot(ax=axs, data=df, x="Protein", y="Phi", hue="Compound Class", palette=col_pal,
                         saturation=100, width = 0.5, errorbar='se', capsize=0.05, errcolor='k', errwidth=1, edgecolor="k", hue_order=["$\phi_{P}=0.5$", "$\phi_{P}=1.0$"])

        sns.move_legend(axs, "upper center", bbox_to_anchor=(.5, 1.025), ncol=2, title=None, frameon=False)

        axs.get_legend().remove()

        plt.yticks(np.arange(0, 1.1, 0.2))
        
        g1.set(xlabel=None)
        g1.set(ylabel=None)
        
        axs.set_ylim(0, 1)
       
        axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
        
        plt.xticks(rotation = 45)
        
        fig_name = "SG_Phi_Bar_Plot"
        
        plt.savefig("Type_Protein_RNA/FIGURES/{}.png".format(fig_name), format="png", dpi=800)
        
        plt.show()
                
        
        # N
        fig, axs = plt.subplots(figsize=(4.8, 3.2), tight_layout=True)
        
        cols = sns.color_palette(palette="Blues_r", n_colors=11)
        col_pal = [cols[5]]
        col_pal.append(cols[10])
        
        g1 = sns.barplot(ax=axs, data=df, x="Protein", y="Cluster", hue="Compound Class", palette=col_pal,
                         saturation=100, width = 0.5, errorbar='se', capsize=0.05, errcolor='k', errwidth=1, edgecolor="k")
        
        axs.get_legend().remove()
        axs.set_ylabel("$N_{D}$")
        plt.xlabel("Protein")
        
        
        
        axs.set_ylim(0, 120)
       
        axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
        
        plt.xticks(rotation = 45)
        
        fig_name = "SG_ND_Bar_Plot"
        
        plt.savefig("Type_Protein_RNA/FIGURES/{}.png".format(fig_name), format="png", dpi=400)
        
        plt.show()
        
        
        # RG
        fig, axs = plt.subplots(figsize=(4.8, 3.2), tight_layout=True)
        
        g1 = sns.barplot(ax=axs, data=df, x="Protein", y="RG", hue="Compound Class", palette="Blues_r",
                         saturation=100, width = 0.5, errorbar='se', capsize=0.05, errcolor='k', errwidth=1, edgecolor="k")

        
        axs.get_legend().remove()
       
        g1.set(xlabel=None)
        g1.set(ylabel=None)
        
        axs.set_ylim(0, 10000)
      
        axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
        
        plt.xticks(rotation = 45)
        
        fig_name = "SG_RG_Bar_Plot"
        
        plt.savefig("Type_Protein_RNA/FIGURES/{}.png".format(fig_name), format="png", dpi=400)
        
        plt.show()
        
        fig = plt.figure(figsize=(3, 6), tight_layout=True)
        

    












def gen_SM(folder, start, end, window, stepSize):
        # Small Molecule Systems
        df = pd.DataFrame(columns=["Small Molecule", "Time", "Cluster", "RG", "Phi", "Compound Class"])
        n_n = 1
        n_y = 1
        for file in (os.scandir("{}/CLUSTER".format(folder))):
            file_name = file.name.split(".")[0].split("_")
            file_name_save = " ".join(file_name)
            if "llps" in file.name:
                if len(file_name) > 5:
                    mol_name = ' '.join(file_name[3]).upper()
                else:
                    mol_name = file_name[3].upper()
                effect = file_name[2]

                if effect == "N":
                    compound = "Non-Dissolving"
                    name = "ND-{}".format(n_n)
                    n_n += 1
                elif effect == "Y":
                    compound = "Dissolving"
                    name = "D-{}".format(n_y)
                    n_y += 1
                else:
                    compound = "SG Control"
                    name = "SG"
                analyzer = Analyze(file)
                df = analyzer.generate_data(name, "Small_Molecules", compound, df, stepSize)

        print(df)

        df = df.loc[df["Time"] >= start]
        df = df.loc[df["Time"] <= end]
        
        df = df.sort_values(by=["Small Molecule"])
        
        print(df)
        gen_SM_plots(df, name=file_name_save, window=window, folder=folder, start=start, end=end)


# Protein Percent Systems
def gen_Percent():
    df = pd.DataFrame(columns=["Percent", "Time", "Cluster", "RG", "Phi"])
    stepSize = 200000
    start = 4
    window = 5
    end = 120
    folder = "Percent_Proteins_RNA"
    for file in (os.scandir("Percent_Proteins_RNA/CLUSTER")):
        file_name = file.name.split(".")[0].split("_")
        if "llps" in file.name:
            name = (int(file_name[2]) / 100)
            compound = ""
            analyzer = Analyze(file)
            df = analyzer.generate_data(name=name, species=folder, effect=compound, df=df, stepSize=stepSize)

    print(df)

    df = df.loc[df["Time"] >= start]
    df = df.loc[df["Time"] <= end]

    df = df.sort_values(by=["Percent"])
    gen_Percent_plots(df)




# Protein Percent Systems
def gen_comp():
    df = pd.DataFrame(columns=["Protein", "Time", "Cluster", "RG", "Phi"])
    stepSize = 200000
    start = 4
    window = 5
    end = 120
    folder = "Type_Protein_RNA"
    
    for file in (os.scandir("Type_Protein_RNA/CLUSTER")):
        file_name = file.name.split(".")[0].split("_")
        if "llps" in file.name:
            name = file_name[2]
            compound = "$\phi_{P}=0.5$"
            analyzer = Analyze(file)
            df = analyzer.generate_data(name=name, species=folder, effect=compound, df=df, stepSize=stepSize)
            
    
    for file in (os.scandir("Type_Protein_Pure/CLUSTER")):
        file_name = file.name.split(".")[0].split("_")
        if "llps" in file.name:
            name = file_name[2]
            compound = "$\phi_{P}=1.0$"
            analyzer = Analyze(file)
            df = analyzer.generate_data(name=name, species=folder, effect=compound, df=df, stepSize=stepSize)
            
    

    print(df)

    df = df.loc[df["Time"] >= start]
    df = df.loc[df["Time"] <= end]
    
    #df.group_by().mean()AA
    
    df = df.sort_values(by=("Protein"))
    
    
    df = df.sort_values(by=("Phi"))

   
    gen_composition_plots(df)





#gen_Percent()

#df_sm_exp_com = pd.DataFrame(columns = ["Small Molecule", "SG Area Reduction", "Phi"])

#gen_SM(folder="Small_Molecules_1uM", start=20, end=120, window = 5, stepSize=200000)

gen_comp()






#gen_Percent()

"""
    def value_line_plot(df, column, start, end, window, species):
        leg = True
        plt.figure(figsize=(10, 6), tight_layout=True)
        df.sort_values(by=["Legend"])
        df = df.loc[df["Time"] >= start]
        df = df.loc[df["Time"] <= end]
        df = df.loc[df["Species"] == species]

        if species == "Percent_Proteins_RNA":
            if window > 1:
                df["Rolling"] = df[column].rolling(window).mean()
                sns.lineplot(data=df, x="Time", y="Rolling", linewidth=2, palette="viridis",
                             marker='o',
                             legend=leg)
            else:
                sns.lineplot(data=df, x="Time", y=column, linewidth=2, palette="viridis",
                             marker='o',
                             legend=leg)
        if species == "Small_Molecules":
            if window > 1:
                df["Rolling"] = df[column].rolling(window).mean()
                sns.lineplot(data=df, x="Time", y="Rolling", hue="Compound Class", linewidth=2, palette="viridis", marker='o',
                             legend=leg)
            else:
                sns.lineplot(data=df, x="Time", y=column, hue="Compound Class", linewidth=2, palette="viridis", marker='o',
                             legend=leg)
        else:
            if window > 1:
                df["Rolling"] = df[column].rolling(window).mean()
                sns.lineplot(data=df, x="Time", y="Rolling", hue="Legend", linewidth=2, palette="viridis", marker='o', legend=leg)
            else:
                sns.lineplot(data=df, x="Time", y=column, hue="Legend", linewidth=2, palette="viridis", marker='o', legend=leg)

        if species == "Percent_Proteins_RNA":
            plt.legend(sorted(legend))

        plt.xlabel('t  (ns)')
        if column == "Cluster":
            plt.ylabel('$N_D$')
        elif column == "RG":
            plt.ylabel('$R_g$  $(\AA)$')
        elif column == "Phi":
            plt.ylabel('$\phi_D$')
        plt.xlim(start+4*(window-1), end)
        if window > 1:
            plt.savefig(fold + "/FIGURES/" + column + "_" + species + "_Rolling.png", format="png", dpi=400)
        else:
            plt.savefig(fold + "/FIGURES/" + column + "_" + species + ".png", format="png", dpi=400)


    def bar_plot(df, column, start, end):
        plt.figure(figsize=(10, 6), tight_layout=True)
        df = df.loc[df["Time"] >= start]
        df = df.loc[df["Time"] <= end]
        if len(folders) > 1:
            sns.barplot(data=df, x="Legend", y=column, hue="Species", palette="viridis", width=0.8)
            plt.xlabel("Protein Species")
        elif folders[0] == "Percent_Proteins_RNA":
            sns.barplot(data=df, x="Legend", y=column, hue ="Compound Class", palette="viridis", width=1)
            plt.xlabel("$\phi_P$")
        elif folders[0] == "Small_Molecules":
            sns.barplot(data=df, x="Legend", y=column, hue="Species", palette="viridis", width=1)
            plt.xlabel("$Compound$")
            plt.tick_params(bottom=False)  # labels along the bottom edge are off

        if column == "Cluster":
            plt.ylabel('$N_D$')
        elif column == "RG":
            plt.ylabel('$R_g$  $(\AA)$')
        elif column == "Phi":
            plt.ylabel('$\phi_D$')
        plt.savefig(fold + "/FIGURES/" + column + "_Bar.png", format="png", dpi=400)

    for fold in folders:
        folder = fold+"/CLUSTER/"
        for file in (os.scandir(folder)):
            if "llps" in file.name:

                if len(folders) > 1:
                    name = file.name.split(".")[0].split("_")[2]
                    if name == "RNA":
                        name = "RNA + Protein"
                    elif name == "Pure":
                        name == "Protein"
                    species = "Protein"
                    if species == "RNA":
                        species = "RNA + Protein"
                    effect = "Proteins"

                elif fold == "Small_Molecules":
                    species = fold
                    if len(file.name.split(".")[0].split("_")) == 4:
                        name = file.name.split(".")[0].split("_")[-1].upper()
                        effect = file.name.split(".")[0].split("_")[2]
                    elif len(file.name.split(".")[0].split("_")) > 4:
                        name = file.name.split(".")[0].split("_")[-1].upper() + " " + file.name.split(".")[0].split("_")[-1].upper()
                        effect = file.name.split(".")[0].split("_")[2]
                    else:
                        name = file.name.split(".")[0].split("_")[-1].upper()
                        effect = "Control SG"
                    if effect == "Y":
                        effect = "Dissolving"
                    elif effect == "N":
                        effect = "Non-Dissolving"
                elif fold == "Percent_Proteins_RNA":
                    name = (float(file.name.split(".")[0].split("_")[2]) / 100)
                    effect = "$\phi_P$"+str(name)
                    species = fold

                analyzer = Analyze(file)
                df = analyzer.generate_data(name, species, df, effect)

        df.sort_values(["Legend"])

        print(folders)

        print(df)

        value_line_plot(df, "Cluster", start, end, window, species)
        plt.show()

        value_line_plot(df, "RG", start, end, window, species)
        plt.show()

        value_line_plot(df, "Phi", start, end, window, species)
        plt.show()

    avg_start = int(input("Enter Average Calculation Starting Time"))

    bar_plot(df, "Cluster", avg_start, end)
    plt.show()

    bar_plot(df, "RG", avg_start, end)
    plt.show()

    bar_plot(df, "Phi", avg_start, end)
    plt.show()




    # Radius of Gyration
    plt.figure(figsize=(10, 6), tight_layout=True)
    i = 0
    for name in names:
        leg.append(str(name)+"%")
        file = folder+"/llps_rg_"+str(name)+".out"
        analyzer = Analyze(file)
        analyzer.generate_data(x_lim, cluster, atoms, rg, str(name))
        color = sns.color_palette("viridis", col_num)[i]
        analyzer.gen_rg_plot(color)
        i += 1
    plt.xlim(start, end)
    plt.legend(leg)
    plt.savefig(folder + "_Radius_of_Gyration.png", format="png", dpi=400)
    #plt.show()





    # Average Radius of Gyration
    plt.figure(figsize=(10, 6), tight_layout=True)
    i = 0
    for name in names:
        file = folder + "/llps_rg_" + str(name) + ".out"
        analyzer = Analyze(file)
        analyzer.generate_data(x_lim, cluster, atoms, rg, str(name))
        color = sns.color_palette("viridis", col_num)[i]
        analyzer.gen_rg_avg_plot(color, window)
        i += 1
    plt.xlim(start+4*(window-1), end)
    plt.legend(leg)
    plt.savefig(folder + "_Radius_of_Gyration_Avg.png", format="png", dpi=400)
    #plt.show()





    # Cluster Number
    plt.figure(figsize=(10, 6), tight_layout=True)
    i = 0
    for name in names:
        file = folder + "/llps_rg_" + str(name) + ".out"
        analyzer = Analyze(file)
        analyzer.generate_data(x_lim, cluster, atoms, rg, str(name))
        color = sns.color_palette("viridis", col_num)[i]
        analyzer.gen_cluster_plot(color)
        i+=1
    plt.xlim(start, end)
    plt.legend(leg)
    plt.savefig(folder + "_Cluster_Number.png",  format="png", dpi=400)
    #plt.show()





    # Average Cluster Number
    plt.figure(figsize=(10, 6), tight_layout=True)
    i = 0
    for name in names:
        file = folder + "/llps_rg_" + str(name) + ".out"
        analyzer = Analyze(file)
        analyzer.generate_data(x_lim, cluster, atoms, rg, str(name))
        color = sns.color_palette("viridis", col_num)[i]
        analyzer.gen_cluster_avg_plot(color, window)
        i += 1
    plt.legend(leg)
    plt.xlim(start+4*(window-1), end)
    plt.savefig(folder + "_Cluster_Number_Avg.png", format="png", dpi=400)
    #plt.show()





    # Atom Number
    plt.figure(figsize=(10, 6), tight_layout=True)
    i = 0
    for name in names:
        file = folder + "/llps_rg_" + str(name) + ".out"
        analyzer = Analyze(file)
        color = sns.color_palette("viridis", col_num)[i]
        analyzer.generate_data(x_lim, cluster, atoms, rg, str(name))
        analyzer.gen_atom_plot(color)
        i += 1
    plt.xlim(start, end)
    plt.legend(leg)
    plt.savefig(folder + "_Atom_Number.png", format="png", dpi=400)
    #plt.show()





    # Average Atom Number
    plt.figure(figsize=(10, 6), tight_layout=True)
    i = 0
    for name in names:
        file = folder + "/llps_rg_" + str(name) + ".out"
        analyzer = Analyze(file)
        color = sns.color_palette("viridis", col_num)[i]
        analyzer.generate_data(x_lim, cluster, atoms, rg, str(name))
        analyzer.gen_atom_avg_plot(color, window)
        i += 1
    plt.xlim(start+4*(window-1), end)
    plt.legend(leg)
    plt.savefig(folder + "_Atom_Number_Avg.png", format="png", dpi=400)
    plt.show()

    def cluster_percent_plot(cluster):
        clust_avg_arr = []
        clust_std_arr = []
        percent_arr = []
        for key in (cluster.keys()):
            clust = np.array(list(cluster[key].values()))
            ave = np.mean(clust[60:])
            std = np.std(clust[60:])
            clust_avg_arr.append(ave)
            clust_std_arr.append(std)
            percent_arr.append((key))



        sns.barplot(x=percent_arr, y=clust_avg_arr, palette="viridis", width=1)
        plt.errorbar(x=percent_arr, y=clust_avg_arr, yerr=clust_std_arr)
        #plt.title('Number of Clusters as a Function of Protein Percentage')
        plt.xlabel('$\phi_P$ (%)')
        plt.ylabel('$N_D$')


    def atoms_percent_plot(atoms):
        atoms_arr = []
        percent_arr = []
        for key in (atoms.keys()):
            sum = 0
            n = 0
            ind = 80
            while ind <= 116:
                sum += atoms[key][ind]
                n += 1
                ind += 4
            ave = sum / n
            atoms_arr.append(ave)
            percent_arr.append(key)
        sns.barplot(x=percent_arr, y=atoms_arr, palette="viridis", width=1,  errorbar="se")
        #plt.title('Number of Atoms in the Largest Cluster as a Function of Protein Percentage')
        plt.xlabel('$\phi_P$ (%)')
        plt.ylabel('$\phi_D$')


    def rg_percent_plot(rg):
        rg_dict = {}
        rg_arr = []
        percent_arr = []
        for key in (rg.keys()):
            sum = 0
            n = 0
            ind = 20
            while ind <= 116:
                sum += rg[key][ind]
                n += 1
                ind += 4
            ave = sum / n
            rg_arr.append(ave)
            percent_arr.append(key)
            rg_dict[key] = ave
        sns.barplot(x=percent_arr, y=rg_arr, palette="viridis", width=1, errorbar="se")
        #plt.title('Radius of Gyration of the Largest Cluster as a Function of Protein Percentage')
        plt.xlabel('$\phi_P$ (%)')
        plt.ylabel('$R_g$  $(\AA)$')


    plt.figure(figsize=(10, 6), tight_layout=True)
    cluster_percent_plot(cluster)
    plt.savefig(folder + "_Cluster_Number_Bar.png", format="png", dpi=400)
    plt.show()

    plt.figure(figsize=(10, 6), tight_layout=True)
    atoms_percent_plot(atoms)
    plt.savefig(folder + "_Atom_Number_Bar.png", format="png", dpi=400)
    plt.show()

    plt.figure(figsize=(10, 6), tight_layout=True)
    rg_percent_plot(rg)
    plt.savefig(folder + "_RG_Bar.png", format="png", dpi=400)
    plt.show()
    """
