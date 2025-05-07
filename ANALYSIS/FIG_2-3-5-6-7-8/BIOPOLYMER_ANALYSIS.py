import sys

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from RDP_PLOT import RDP
from RDP_NORMALIZE import RDP_NORMALIZE
from scipy.interpolate import UnivariateSpline

class BIOPOLYMER_ANALYSIS():
    def __init__(self, path):
        self.path = path

    def gen_biopolymer_fitters(self, folder, sm):
        pca_file_A = "{}/ANALYSIS_SG_AVE/PCA_Protein_sg_X.csv".format(self.path)
        cluster_file_A = "{}/ANALYSIS_SG_AVE/Cluster_Protein_sg_X.csv".format(self.path)
        init = [80, 80, 200, 80]
        T = 300

        label = "G3BP1"
        print("{}    {}".format(sm,label))
        density_file = "{}/{}/Density_Profile_G3BP1_{}.csv".format(self.path,folder,sm)
        #fitterG3BP1 = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        fitterG3BP1 = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        #fitterG3BP1 = RDP_NORMALIZE_CONTROL(density_file, pca_file_A, cluster_file_A, T, init, label, "{}/{}/Density_Profile_G3BP1_{}.csv".format(path,"BIOPOLYMER_ANALYSIS","SG"))

        label = "TDP43"
        print("{}    {}".format(sm, label))
        density_file = "{}/{}/Density_Profile_TDP43_{}.csv".format(self.path,folder,sm)
        #fitterTDP43 = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        fitterTDP43 = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        #fitterTDP43 = RDP_NORMALIZE_CONTROL(density_file, pca_file_A, cluster_file_A, T, init, label,"{}/{}/Density_Profile_TDP43_{}.csv".format(path,"BIOPOLYMER_ANALYSIS", "SG"))

        label = "FUS"
        print("{}    {}".format(sm, label))
        density_file = "{}/{}/Density_Profile_FUS_{}.csv".format(self.path,folder,sm)
        #fitterFUS = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        fitterFUS = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        #fitterFUS = RDP_NORMALIZE_CONTROL(density_file, pca_file_A, cluster_file_A, T, init, label,"{}/{}/Density_Profile_FUS_{}.csv".format(path,"BIOPOLYMER_ANALYSIS", "SG"))

        label = "PABP1"
        print("{}    {}".format(sm, label))
        density_file = "{}/{}/Density_Profile_PABP1_{}.csv".format(self.path,folder,sm)
        #fitterPABP1 = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        fitterPABP1 = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        #fitterPABP1 = RDP_NORMALIZE_CONTROL(density_file, pca_file_A, cluster_file_A, T, init, label,"{}/{}/Density_Profile_PABP1_{}.csv".format(path,"BIOPOLYMER_ANALYSIS", "SG"))

        label = "TIA1"
        print("{}    {}".format(sm, label))
        density_file = "{}/{}/Density_Profile_TIA1_{}.csv".format(self.path,folder,sm)
        #fitterTIA1 = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        fitterTIA1 = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        #fitterTIA1 = RDP_NORMALIZE_CONTROL(density_file, pca_file_A, cluster_file_A, T, init, label,"{}/{}/Density_Profile_TIA1_{}.csv".format(path,"BIOPOLYMER_ANALYSIS", "SG"))

        label = "TTP"
        print("{}    {}".format(sm, label))
        density_file = "{}/{}/Density_Profile_TTP_{}.csv".format(self.path,folder,sm)
        #fitterTTP = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        fitterTTP = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        #fitterTTP = RDP_NORMALIZE_CONTROL(density_file, pca_file_A, cluster_file_A, T, init, label,"{}/{}/Density_Profile_TTP_{}.csv".format(path,"BIOPOLYMER_ANALYSIS", "SG"))

        label = "RNA"
        print("{}    {}".format(sm, label))
        density_file = "{}/{}/Density_Profile_RNA_{}.csv".format(self.path,folder,sm)
        #fitterRNA = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        fitterRNA = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
        #fitterRNA = RDP_NORMALIZE_CONTROL(density_file, pca_file_A, cluster_file_A, T, init, label, "{}/{}/Density_Profile_RNA_{}.csv".format(path,"BIOPOLYMER_ANALYSIS", "SG"))

        return fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA


    def plot_biopolymer_curve(self, fitter, ax1, fig, col, lab, offset, line_style, line_width):
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

        densities = np.array(fitter.densities) + offset
        temp_dens = list(densities.copy())
        temp_dens.insert(0,0)
        temp_dist = fitter.distances.copy()
        temp_dist.insert(0,0)

        axis_dens = np.zeros_like(temp_dist) + offset

        axis_dist = temp_dist

        ax1.plot(fitter.distances, densities, color=col, label=lab, zorder=1, linewidth=line_width, linestyle=line_style)

        if lab != "":
            spl = UnivariateSpline(fitter.distances, fitter.densities, k=5)
            spl.set_smoothing_factor(1)
            xs = np.linspace(30, 330, 1000)
            spl_vals = spl(xs)
            index = (np.abs(np.array(spl_vals[10:]) - 0.5)).argmin()
            x_pos = xs[index]
            print(x_pos)
            plt.plot([x_pos,x_pos],[offset, offset+spl_vals[index]],color=col,linewidth=2,zorder=1,linestyle=(0, (2, 0.5)))
            #plt.plot(xs, spl(xs)+offset, 'g', lw=3)



            if offset != 0:
                ax1.plot(axis_dist,axis_dens,color='k',linewidth=2)
                #ax1.plot(axis_dist, axis_dens+0.5, color='k', linewidth=1)

            #if offset == 0:
                #ax1.plot(axis_dist, axis_dens + 0.5, color='k', linewidth=1)

            sns.scatterplot(ax=ax1, x=fitter.distances[0:16], y=densities[0:16], color=col, legend=False, s=40,
                            edgecolor="k", linewidth=1, zorder=10, clip_on=False)

            ax1.errorbar(x=fitter.distances, y=densities, yerr=fitter.errors, fmt=".", color=col,
                         zorder=2)

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)

        ax1.set_xlim(10, 400)
        ax1.set_ylim(0, 7)

        return fig, ax1


    def plot_biopolymer_curve(self, fitter, ax1, fig, col, lab, offset, line_style, line_width):
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

        densities = np.array(fitter.densities)
        temp_dens = list(densities.copy())
        temp_dens.insert(0,0)
        temp_dist = fitter.distances.copy()
        temp_dist.insert(0,0)

        axis_dens = np.zeros_like(temp_dist)

        axis_dist = temp_dist

        ax1.plot(fitter.distances, densities, color=col, label=lab, zorder=1, linewidth=line_width, linestyle=line_style)

        if lab != "":

            sns.scatterplot(ax=ax1, x=fitter.distances[0:20], y=densities[0:20], color=col, legend=False, s=40,
                            edgecolor="k", linewidth=1, zorder=10, clip_on=False)

            ax1.errorbar(x=fitter.distances, y=densities, yerr=fitter.errors, fmt=".", color=col,
                            zorder=2)

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)

        ax1.set_xlim(10, 400)
        ax1.set_ylim(0, 200)

        return fig, ax1




    def gen_biopolymer_plots(self, fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA, sm):
        col_pall = sns.color_palette("rocket", n_colors=14)
        col_pall2 = sns.color_palette(["#0066ff"], 1)
        col_pal_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)
        #col_pall2 = sns.color_palette()
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

        fig1, ax1 = plt.subplots(figsize=(3.2, 3.2))

        if sm != "SG":
            offset_arr = np.arange(1,8,1)
        else:
            offset_arr = np.arange(0, 7, 1)


        label = "TDP43"
        offset = offset_arr[6]
        col = col_pall[0]
        fig1, ax1 = self.plot_biopolymer_curve(fitterTDP43, ax1, fig1, col, label, offset, line_width=3, line_style='solid')

        label = "FUS"
        offset = offset_arr[5]
        col = col_pall[2]
        fig1, ax1 = self.plot_biopolymer_curve(fitterFUS, ax1, fig1, col, label, offset, line_width=3, line_style='solid')

        label = "TIA1"
        offset = offset_arr[4]
        col = col_pall[4]
        fig1, ax1 = self.plot_biopolymer_curve(fitterTIA1, ax1, fig1, col, label, offset, line_width=3, line_style='solid')

        label = "G3BP1"
        offset = offset_arr[3]
        col = col_pall[6]
        fig1, ax1 = self.plot_biopolymer_curve(fitterG3BP1, ax1, fig1, col, label, offset, line_width=3, line_style='solid')

        label = "PABP1"
        offset = offset_arr[2]
        col = col_pall[8]
        fig1, ax1 = self.plot_biopolymer_curve(fitterPABP1, ax1, fig1, col, label, offset, line_width=3, line_style='solid')

        label = "TTP"
        offset = offset_arr[1]
        col = col_pall[10]
        fig1, ax1 = self.plot_biopolymer_curve(fitterTTP, ax1, fig1, col, label, offset, line_width=3, line_style='solid')

        label = "RNA"
        offset = offset_arr[0]
        col = col_pall2[0]
        fig1, ax1 = self.plot_biopolymer_curve(fitterRNA, ax1, fig1, col, label, offset, line_width=3, line_style='solid')

        if sm != "SG":
            label = sm
            pca_file_A = "{}/ANALYSIS_SG_AVE/PCA_Protein_sg_X.csv".format(self.path)
            cluster_file_A = "{}/ANALYSIS_SG_AVE/Cluster_Protein_sg_X.csv".format(self.path)
            init = [80, 80, 200, 80]
            T = 300
            density_file = "{}/BIOPOLYMER_SUMMARY/Density_Profile_SM_{}.csv".format(self.path, sm)
            fitterSM = RDP_NORMALIZE(density_file, pca_file_A, cluster_file_A, T, init, label)
            offset = 0
            if sm == "DSM":
                col = col_pal_sm[0]
            else:
                col = col_pal_sm[1]
            #fig1, ax1 = self.plot_biopolymer_curve(fitterSM, ax1, fig1, col, label, offset, line_width=3,
            #                                       line_style='solid')

        leg = fig1.legend(loc='upper right', ncol=1, bbox_to_anchor=(0.66, 0.11, 0.25, 0.78),labelspacing=0.4) #1.8
        leg.get_frame().set_alpha(0)

        if sm != "SG":

            fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA = self.gen_biopolymer_fitters("BIOPOLYMER_ANALYSIS_SG", "SG")

            label = ""
            offset = offset_arr[6]
            col = col_pall[0]
            fig1, ax1 = self.plot_biopolymer_curve(fitterTDP43, ax1, fig1, col, label, offset, line_width=2, line_style=(0, (1, 1)))

            label = ""
            offset = offset_arr[5]
            col = col_pall[2]
            fig1, ax1 = self.plot_biopolymer_curve(fitterFUS, ax1, fig1, col, label, offset, line_width=2, line_style=(0, (1, 1)))

            label = ""
            offset = offset_arr[4]
            col = col_pall[4]
            fig1, ax1 = self.plot_biopolymer_curve(fitterTIA1, ax1, fig1, col, label, offset, line_width=2, line_style=(0, (1, 1)))

            label = ""
            offset = offset_arr[3]
            col = col_pall[6]
            fig1, ax1 = self.plot_biopolymer_curve(fitterG3BP1, ax1, fig1, col, label, offset, line_width=2, line_style=(0, (1, 1)))

            label = ""
            offset = offset_arr[2]
            col = col_pall[8]
            fig1, ax1 = self.plot_biopolymer_curve(fitterPABP1, ax1, fig1, col, label, offset, line_width=2, line_style=(0, (1, 1)))

            label = ""
            offset = offset_arr[1]
            col = col_pall[10]
            fig1, ax1 = self.plot_biopolymer_curve(fitterTTP, ax1, fig1, col, label, offset, line_width=2, line_style=(0, (1, 1)))

            label = ""
            offset = offset_arr[0]
            col = col_pall2[0]
            fig1, ax1 = self.plot_biopolymer_curve(fitterRNA, ax1, fig1, col, label, offset, line_width=2, line_style=(0, (1, 1)))

        #ax1.set_xlim(10, 330)
        #ax1.set_ylim(0, 8)

        fig1.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/PROTEIN_ANALYSIS_NCC_{}.png".format(self.path,sm), format="png", dpi=400)




    def plot_rna_curve(self, fitterA, fitterUCG, fitterRNA, sm):
        col_pal = sns.color_palette(["#0066ff", "#99c2ff", "#38C7C5"], 3)

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

        ax1.plot(fitterRNA.distances, fitterRNA.densities, color=col_pal[0], label='RNA', linewidth=4,
                 zorder=1)
        sns.scatterplot(ax=ax1, x=fitterRNA.distances[0:24], y=fitterRNA.densities[0:24], color=col_pal[0],
                        legend=False,
                        s=40, edgecolor="k", linewidth=1, zorder=3, clip_on=False)
        ax1.errorbar(fitterRNA.distances, fitterRNA.densities, yerr=fitterRNA.errors, fmt=".",
                     color=col_pal[0], zorder=2)

        ax1.plot(fitterA.distances, fitterA.densities, color=col_pal[1], label='A', linewidth=4, zorder=1)
        sns.scatterplot(ax=ax1, x=fitterA.distances[0:24], y=fitterA.densities[0:24], color=col_pal[1], legend=False, s=40,
                        edgecolor="k", linewidth=1, zorder=3, clip_on=False)
        ax1.errorbar(x=fitterA.distances, y=fitterA.densities, yerr=fitterA.errors, fmt=".", color=col_pal[1],
                     zorder=2)

        ax1.plot(fitterUCG.distances, fitterUCG.densities, color=col_pal[2], label='UCG', linewidth=4,
                 zorder=1)
        sns.scatterplot(ax=ax1, x=fitterUCG.distances[0:24], y=fitterUCG.densities[0:24], color=col_pal[2],
                        legend=False,
                        s=40, edgecolor="k", linewidth=1, zorder=3, clip_on=False)
        ax1.errorbar(fitterUCG.distances, fitterUCG.densities, yerr=fitterUCG.errors, fmt=".",
                     color=col_pal[2], zorder=2)


        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)

        ax1.set_xlim(10, 500)

        if sm != "SG":
            ax1.set_ylim(0, 180)

            label = "A"
            T = 300
            pca_file_A = "{}/ANALYSIS_SG_AVE/PCA_RNA_sg_X.csv".format(self.path)
            cluster_file_A = "{}/ANALYSIS_SG_AVE/Cluster_RNA_sg_X.csv".format(self.path)
            init = [80, 80, 200, 80]

            density_file_A = "{}/BIOPOLYMER_ANALYSIS_SG/Density_Profile_ADENINE_SG.csv".format(self.path)
            print("{}    {}".format(sm, label))
            fitterA = RDP(density_file_A, pca_file_A, cluster_file_A, T, init, label)

            density_file_UCG = "{}/BIOPOLYMER_ANALYSIS_SG/Density_Profile_UCG_SG.csv".format(self.path)
            print("{}    {}".format(sm, label))
            fitterUCG = RDP(density_file_UCG, pca_file_A, cluster_file_A, T, init, label)

            density_file_RNA = "{}/BIOPOLYMER_ANALYSIS_SG/Density_Profile_RNA_SG.csv".format(self.path)
            print("{}    {}".format(sm, label))
            fitterRNA = RDP(density_file_RNA, pca_file_A, cluster_file_A, T, init, label)

            ax1.plot(fitterRNA.distances, fitterRNA.densities, color=col_pal[0], label='RNA Control', zorder=1, linewidth=2, linestyle=(0, (1, 1)))
            ax1.plot(fitterA.distances, fitterA.densities, color=col_pal[1], label='A Control', zorder=1,
                     linewidth=2, linestyle=(0, (1, 1)))
            ax1.plot(fitterUCG.distances, fitterUCG.densities, color=col_pal[2], label='UCG Control', zorder=1,
                     linewidth=2, linestyle=(0, (1, 1)))






        else:
            ax1.set_ylim(0, 140)

        leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
        leg.get_frame().set_alpha(0)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/RNA_AUCG_NCC_{}.png".format(self.path,sm), format="png", dpi=400)




    def gen_rna_plots(self, folder, sm):
        label = "A"
        T = 300
        pca_file_A = "{}/ANALYSIS_SG_AVE/PCA_RNA_sg_X.csv".format(self.path)
        cluster_file_A = "{}/ANALYSIS_SG_AVE/Cluster_RNA_sg_X.csv".format(self.path)
        init = [80, 80, 200, 80]

        density_file_A = "{}/{}/Density_Profile_ADENINE_{}.csv".format(self.path,folder,sm)
        print("{}    {}".format(sm, label))
        fitterA = RDP(density_file_A, pca_file_A, cluster_file_A, T, init, label)

        density_file_UCG = "{}/{}/Density_Profile_UCG_{}.csv".format(self.path,folder,sm)
        print("{}    {}".format(sm, label))
        fitterUCG = RDP(density_file_UCG, pca_file_A, cluster_file_A, T, init, label)

        density_file_RNA = "{}/{}/Density_Profile_RNA_{}.csv".format(self.path,folder,sm)
        print("{}    {}".format(sm, label))
        fitterRNA = RDP(density_file_RNA, pca_file_A, cluster_file_A, T, init, label)

        fig = self.plot_rna_curve(fitterA, fitterUCG, fitterRNA, sm)

        return fig


    def plot_rdp(self, sm):
        pca_file_A = "{}/ANALYSIS_SG_AVE/PCA_Protein_sg_X.csv".format(self.path)
        cluster_file_A = "{}/ANALYSIS_SG_AVE/Cluster_Protein_sg_X.csv".format(self.path)
        init = [80, 80, 200, 80]
        T = 300

        label = "SG"
        print("{}    {}".format(sm,label))
        density_file = "{}/BIOPOLYMER_SUMMARY/Density_Profile_SG_{}.csv".format(self.path,sm)
        fitterSG = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "Protein"
        print("{}    {}".format(sm, label))
        density_file = "{}/BIOPOLYMER_SUMMARY/Density_Profile_Protein_{}.csv".format(self.path,sm)
        fitterProtein = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "RNA"
        print("{}    {}".format(sm, label))
        density_file = "{}/BIOPOLYMER_SUMMARY/Density_Profile_RNA_{}.csv".format(self.path,sm)
        fitterRNA = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)

        if sm == "DSM" or sm == "NDSM":
            label = "RNA"
            print("{}    {}".format(sm, label))
            density_file = "{}/BIOPOLYMER_SUMMARY/Density_Profile_SM_{}.csv".format(self.path,sm)
            fitterSM = RDP(density_file, pca_file_A, cluster_file_A, T, init, label)
            sg_control_density_file = "{}/BIOPOLYMER_SUMMARY/Density_Profile_SG_SG.csv".format(self.path)
            protein_control_density_file = "{}/BIOPOLYMER_SUMMARY/Density_Profile_Protein_SG.csv".format(self.path)
            rna_control_density_file = "{}/BIOPOLYMER_SUMMARY/Density_Profile_RNA_SG.csv".format(self.path)

            fitter_SG_Control = RDP(sg_control_density_file, pca_file_A, cluster_file_A, T, init, label)
            fitter_Protein_Control = RDP(protein_control_density_file, pca_file_A, cluster_file_A, T, init, label)
            fitter_RNA_Control = RDP(rna_control_density_file, pca_file_A, cluster_file_A, T, init, label)

        else:
            fitterSM = None

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

        fig, ax1 = plt.subplots(figsize=(1.8, 1.8))

        ax1.plot(fitterSG.fit_x, fitterSG.fit_rho, color=col_pal_sg, label='SG', linewidth=4, zorder=1)
        sns.scatterplot(ax=ax1, x=fitterSG.distances[0:24], y=fitterSG.densities[0:24], color=col_pal_sg, legend=False, s=40,
                        edgecolor="k", linewidth=1, zorder=3, clip_on=True)
        ax1.errorbar(x=fitterSG.distances, y=fitterSG.densities, yerr=fitterSG.errors, fmt=".", color=col_pal_sg,
                     zorder=2)

        ax1.plot(fitterProtein.fit_x, fitterProtein.fit_rho, color=col_pal_protein, label='Protein', linewidth=4,
                 zorder=1)
        sns.scatterplot(ax=ax1, x=fitterProtein.distances[0:24], y=fitterProtein.densities[0:24], color=col_pal_protein,
                        legend=False,
                        s=40, edgecolor="k", linewidth=1, zorder=3, clip_on=True)
        ax1.errorbar(fitterProtein.distances, fitterProtein.densities, yerr=fitterSG.errors, fmt=".",
                     color=col_pal_protein, zorder=2)

        ax1.plot(fitterRNA.fit_x, fitterRNA.fit_rho, color=col_pal_rna, label='RNA', linewidth=4, zorder=2)
        sns.scatterplot(ax=ax1, x=fitterRNA.distances[0:24], y=fitterRNA.densities[0:24], color=col_pal_rna, legend=False,
                        s=40,
                        edgecolor="k", linewidth=1, zorder=3, clip_on=True)
        ax1.errorbar(fitterRNA.distances, fitterRNA.densities, yerr=fitterRNA.errors, fmt=".", color=col_pal_rna,
                     zorder=2)

        if fitterSM is not None:
            ax2 = ax1.twinx()
            if "ND" not in sm:
                col = col_pal_sm[0]
            else:
                col = col_pal_sm[1]

            sns.scatterplot(ax=ax2, x=fitterSM.distances[0:24], y=fitterSM.densities[0:24], color=col, legend=False, s=40,
                            edgecolor="k", linewidth=1, zorder=3, clip_on=False)
            ax2.errorbar(fitterSM.distances, fitterSM.densities, yerr=fitterSM.errors, fmt=".", color=col, zorder=2)
            sns.lineplot(ax=ax2, x=fitterSM.fit_x, y=fitterSM.fit_rho, color=col, label=sm, linewidth=4, zorder=1)

            ax1.plot(fitter_SG_Control.fit_x, fitter_SG_Control.fit_rho, color=col_pal_sg, label='SG Control',
                     linewidth=2, zorder=1, linestyle=(0, (1, 1)))

            ax1.plot(fitter_Protein_Control.fit_x, fitter_Protein_Control.fit_rho, color=col_pal_protein, label='Protein Control',
                     linewidth=2, zorder=1, linestyle=(0, (1, 1)))

            ax1.plot(fitter_RNA_Control.fit_x, fitter_RNA_Control.fit_rho, color=col_pal_rna,
                     label='RNA Control',
                     linewidth=2, zorder=1, linestyle=(0, (1, 1)))


            ax2.get_legend().remove()
            ax2.tick_params(left=False, right=True, top=False, bottom=False, labelbottom=False, direction='in',
                            length=4,
                            width=2)
            ax2.set_ylim(0.0, 0.4)

            ax1.tick_params(left=True, right=False, top=True, bottom=False, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_ylim(0, 600)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)

        else:
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_ylim(0, 500)

        ax1.set_xlim(200, 300)
        ax1.set_ylim(0, 50)

        #leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
        #leg.get_frame().set_alpha(0)

        plt.savefig("{}/BIOPOLYMER_SUMMARY/IMAGES/{}_RDP.png".format(self.path,sm), format="png", dpi=400)
        return fig

    def plot_res_cpm(self, res_contact_array, col, min, max, mid):
        x_res_list = ["G3BP1", "PABP1", "TTP", "TIA1", "TDP43", "FUS", "RNA"]

        x_res_list[4], x_res_list[0] = x_res_list[0], x_res_list[4]
        x_res_list[2], x_res_list[1] = x_res_list[1], x_res_list[2]
        x_res_list[1], x_res_list[5] = x_res_list[5], x_res_list[1]
        x_res_list[3], x_res_list[4] = x_res_list[4], x_res_list[3]
        x_res_list[4], x_res_list[6] = x_res_list[6], x_res_list[4]
        x_res_list[5], x_res_list[6] = x_res_list[6], x_res_list[5]
        x_res_list[2], x_res_list[5] = x_res_list[5], x_res_list[2]


        res_contact_array[[4, 0]] = res_contact_array[[0, 4]]
        res_contact_array[[2, 1]] = res_contact_array[[1, 2]]
        res_contact_array[[1, 5]] = res_contact_array[[5, 1]]
        res_contact_array[[3, 4]] = res_contact_array[[4, 3]]
        res_contact_array[[4, 6]] = res_contact_array[[6, 4]]
        res_contact_array[[5, 6]] = res_contact_array[[6, 5]]
        res_contact_array[[2, 5]] = res_contact_array[[5, 2]]
        res_contact_array[:, [4, 0]] = res_contact_array[:, [0, 4]]
        res_contact_array[:, [2, 1]] = res_contact_array[:, [1, 2]]
        res_contact_array[:, [1, 5]] = res_contact_array[:, [5, 1]]
        res_contact_array[:, [3, 4]] = res_contact_array[:, [4, 3]]
        res_contact_array[:, [4, 6]] = res_contact_array[:, [6, 4]]
        res_contact_array[:, [5, 6]] = res_contact_array[:, [6, 5]]
        res_contact_array[:, [2, 5]] = res_contact_array[:, [5, 2]]

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
        #x_res_list = ["TDP43", "FUS", "TIA1", "G3BP1", "RNA", "PABP1", "TTP"]

        y_res_list = x_res_list

        fig, ax1 = plt.subplots(figsize=(3.8, 3.0), tight_layout=True)
        ax = sns.heatmap(res_contact_array, xticklabels=x_res_list, yticklabels=y_res_list, cmap=col, square=True, center=mid)
        #                 vmin=min, vmax=max, center=mid)
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=10)
        cbar.formatter.set_powerlimits((0, 0))

        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)

        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        return fig

    def plot_res_sm_cpm(self, res_contact_array, col, min, max, mid, sm_list):

        x_res_list = ["G3BP1", "PABP1", "TTP", "TIA1", "TDP43", "FUS", "RNA"]

        x_res_list[4], x_res_list[0] = x_res_list[0], x_res_list[4]
        x_res_list[2], x_res_list[1] = x_res_list[1], x_res_list[2]
        x_res_list[1], x_res_list[5] = x_res_list[5], x_res_list[1]
        x_res_list[3], x_res_list[4] = x_res_list[4], x_res_list[3]
        x_res_list[4], x_res_list[6] = x_res_list[6], x_res_list[4]
        x_res_list[5], x_res_list[6] = x_res_list[6], x_res_list[5]
        x_res_list[2], x_res_list[5] = x_res_list[5], x_res_list[2]

        res_contact_array[[4, 0]] = res_contact_array[[0, 4]]
        res_contact_array[[2, 1]] = res_contact_array[[1, 2]]
        res_contact_array[[1, 5]] = res_contact_array[[5, 1]]
        res_contact_array[[3, 4]] = res_contact_array[[4, 3]]
        res_contact_array[[4, 6]] = res_contact_array[[6, 4]]
        res_contact_array[[5, 6]] = res_contact_array[[6, 5]]
        res_contact_array[[2, 5]] = res_contact_array[[5, 2]]


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

        y_res_list = x_res_list

        fig, ax1 = plt.subplots(figsize=(2.0, 3.2), tight_layout=True)
        ax = sns.heatmap(res_contact_array, xticklabels=sm_list, yticklabels=x_res_list, cmap=col, square=False)#, center=mid)
        #                 vmin=min, vmax=max, )
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=10)
        cbar.formatter.set_powerlimits((0, 0))

        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)

        #plt.yticks(rotation=0)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        return fig

    def gen_residue_cpms(self):

        # Residue DSM / NDSM Difference Standardized Array

        res_list = ["G3BP1",
                    "PABP1",
                    "TTP",
                    "TIA1",
                    "TDP43",
                    "FUS",
                    "RNA"]

        nBio_dsm = np.loadtxt("{}/ANALYSIS_DSM_AGG/BioPolNum_DSM.csv".format(self.path), delimiter=",", dtype=float)
        nBio_ndsm = np.loadtxt("{}/ANALYSIS_NDSM_AGG/BioPolNum_NDSM.csv".format(self.path), delimiter=",", dtype=float)
        nBio_sg = np.loadtxt("{}/ANALYSIS_SG_AVE/BioPolNum_sg_X.csv".format(self.path), delimiter=",", dtype=float)
        nBio = np.loadtxt("CM_NORM/MAPS/BioPolNum_SYSTEM.csv", delimiter=",", dtype=float)

        ndsm_res_arr = np.array(
            pd.read_csv("{}/ANALYSIS_NDSM_AGG/Residue_Contacts_Mean_NDSM.csv".format(self.path), header=None))
        dsm_res_arr = np.array(
            pd.read_csv("{}/ANALYSIS_DSM_AGG/Residue_Contacts_Mean_DSM.csv".format(self.path), header=None))
        sg_res_arr = np.array(
            pd.read_csv("{}/ANALYSIS_SG_AVE/Residue_Contacts_Mean_sg_X.csv".format(self.path), header=None))

        # INDIVIDUAL MAPS
        # UNNORMALIZED
        res_contact_array = np.divide(sg_res_arr, (np.sum(sg_res_arr)))
        #res_contact_array = sg_res_arr
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=0, max=0.07, mid=0.035)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_SG_UNNORMALIZED_HeatMap.png".format(
            self.path), format="png", dpi=400)

        res_contact_array = np.divide(dsm_res_arr, (np.sum(dsm_res_arr)))
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=0, max=0.07, mid=0.035)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_DSM_UNNORMALIZED_HeatMap.png".format(
            self.path), format="png", dpi=400)

        res_contact_array = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr)))
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=0, max=0.07, mid=0.035)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_NDSM_UNNORMALIZED_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # NORMALIZED SYSTEM
        res_contact_array = np.divide(sg_res_arr, (np.sum(sg_res_arr)*nBio))
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=0, max=0.0004, mid=0.0002)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_SG_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path), format="png", dpi=400)

        res_contact_array = np.divide(dsm_res_arr, (np.sum(dsm_res_arr)*nBio))
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=0, max=0.0004, mid=0.0002)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_DSM_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path), format="png", dpi=400)

        res_contact_array = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr)*nBio))
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=0, max=0.0004, mid=0.0002)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_NDSM_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # NORMALIZED CLUSTER
        res_contact_array = np.divide(sg_res_arr, (np.sum(sg_res_arr) * nBio_sg))
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=0, max=0.0003, mid=0.00015)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_SG_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path), format="png", dpi=400)

        res_contact_array = np.divide(dsm_res_arr, (np.sum(dsm_res_arr) * nBio_dsm))
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=0, max=0.0003, mid=0.00015)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_DSM_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path), format="png", dpi=400)

        res_contact_array = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr) * nBio_ndsm))
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=0, max=0.0003, mid=0.00015)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_NDSM_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # DIFFERENCE MAPS
        # UNNORMALIZED
        res_contact_array = np.divide(np.subtract(dsm_res_arr, ndsm_res_arr), sg_res_arr)
        fig = self.plot_res_cpm(res_contact_array, col="Blues_r", min=-0.6, max=0.1, mid=-0.35)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_DIFF_UNNORMALIZED_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # UNNORMALIZED PERCENTAGE
        dsm_sum = np.divide(dsm_res_arr,np.sum(dsm_res_arr))
        ndsm_sum = np.divide(ndsm_res_arr,np.sum(ndsm_res_arr))
        sg_sum = np.divide(sg_res_arr,np.sum(sg_res_arr))
        res_contact_array = np.divide(np.subtract(dsm_sum, ndsm_sum), sg_sum)
        fig = self.plot_res_cpm(res_contact_array, col="coolwarm", min=-0.6, max=0.1, mid=0)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_DIFF_UNNORMALIZED_PERCENTAGE_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # NORMALIZED SYSTEM
        sg_norm_arr = np.divide(sg_res_arr, (np.sum(sg_res_arr))*nBio)
        dsm_norm_arr = np.divide(dsm_res_arr, (np.sum(dsm_res_arr))*nBio)
        ndsm_norm_arr = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr))*nBio)
        res_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
        fig = self.plot_res_cpm(res_contact_array, col="coolwarm", min=-0.25, max=0.25, mid=0)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_DIFF_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # NORMALIZED CLUSTER
        sg_norm_arr = np.divide(sg_res_arr, (np.sum(sg_res_arr)*nBio_sg))
        dsm_norm_arr = np.divide(dsm_res_arr, (np.sum(dsm_res_arr)*nBio_dsm))
        ndsm_norm_arr = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr)*nBio_ndsm))
        res_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=-0.25, max=0.25, mid=0.6)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_DIFF_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # NORMALIZED CLUSTER
        sg_norm_arr = nBio_sg
        dsm_norm_arr = nBio_dsm
        ndsm_norm_arr = nBio_ndsm
        res_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
        fig = self.plot_res_cpm(res_contact_array, col="Reds", min=-0.25, max=0.25, mid=3)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_DIFF_COUNT_HeatMap.png".format(
            self.path), format="png", dpi=400)


        dsm_norm_arr = np.loadtxt("{}/ANALYSIS_DSM_AGG/BioNum_DSM.csv".format(self.path), delimiter=",", dtype=float).transpose()
        ndsm_norm_arr = np.loadtxt("{}/ANALYSIS_NDSM_AGG/BioNum_NDSM.csv".format(self.path), delimiter=",",
                                  dtype=float).transpose()
        sg_norm_arr = np.loadtxt("{}/ANALYSIS_SG_AVE/BioNum_sg_X.csv".format(self.path), delimiter=",",
                                   dtype=float).transpose()
        dsm_norm_arr = np.divide(dsm_norm_arr, np.sum(dsm_norm_arr))
        ndsm_norm_arr = np.divide(ndsm_norm_arr, np.sum(ndsm_norm_arr))
        sg_norm_arr = np.divide(sg_norm_arr, np.sum(sg_norm_arr))
        sm_list = [""]
        res_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr).reshape(-1, 1)
        fig = self.plot_res_sm_cpm(res_contact_array, col="coolwarm", min=0, max=8, mid=0, sm_list=sm_list)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Residue_DIFF_COUNT_1D_HeatMap.png".format(
            self.path), format="png", dpi=400)











        # SM RESIDUE MAPS
        # INDIVIDUAL SM MAPS
        df_res_contact = pd.read_csv("{}/RESULTS/SM_ResMap_Data.csv".format(self.path))
        df_res_count = pd.read_csv("{}/RESULTS/SM_ResCount_Data.csv".format(self.path))
        sm_list = [
            "D-1",
            "D-2",
            "D-3",
            "D-4",
            "D-5",
            "D-6",
            "D-7",
            "D-8",
            "D-9",
            "D-10",
            "N-1",
            "N-2",
            "N-3",
            "N-4",
            "N-5",
            "N-6",
            "N-7",
            "N-8",
            "N-9",
            "N-10",
            "DSM",
            "NDSM"
        ]


        # UNNORMALIZED
        sm_res_contact_array = np.array(df_res_contact.iloc[:, 1:])
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Residue_IND_UNNORMALIZED_HeatMap_OLD.png".format(
            self.path)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Reds", min=0, max=180, mid=90, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)

        # SYSTEM NORMALIZED
        nprot = np.loadtxt("CM_NORM/MAPS/BioNum_SYSTEM.csv", delimiter=",", dtype=float).transpose().reshape(-1, 1)
        sm_res_contact_array = np.divide(np.array(df_res_contact.iloc[:, 1:]),nprot)
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Residue_IND_NORMALIZED_SYSTEM_HeatMap_OLD.png".format(
            self.path)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Reds", min=0, max=8, mid=4, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)

        # CLUSTER NORMALIZED
        sm_res_count_array = np.array(df_res_count.iloc[:, 1:])
        sm_res_contact_array = np.divide(np.array(df_res_contact.iloc[:, 1:]), sm_res_count_array)
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Residue_IND_NORMALIZED_CLUSTER_HeatMap_OLD.png".format(
            self.path)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Reds", min=0, max=20, mid=10, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)





        # DIFFERENCE SM RESIDUE MAPS

        # DIFFERENCE SM RESIDUE MAPS
        #UNNORMALIZED
        sm_arr = np.divide(df_res_contact.loc[:, "DSM"],np.sum(df_res_contact.loc[:, "DSM"])) - np.divide(df_res_contact.loc[:, "NDSM"], np.sum(df_res_contact.loc[:, "NDSM"]))
        #sm_arr = df_res_contact.loc[:, "DSM"] - df_res_contact.loc[:, "NDSM"]

        sm_res_contact_array = np.transpose(np.asarray([sm_arr]))
        sm_list = [""]
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Residue_DIFF_UNNORMALIZED_HeatMap.png".format(
            self.path)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="coolwarm", min=-0.4, max=-10, mid=0, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)

        # SYSTEM NORMALIZED
        nprot = np.loadtxt("CM_NORM/MAPS/BioNum_SYSTEM.csv", delimiter=",", dtype=float).transpose()
        sm_arr = np.divide(df_res_contact.loc[:, "DSM"], nprot) - np.divide(
            df_res_contact.loc[:, "NDSM"], nprot)
        sm_res_contact_array = np.transpose(np.asarray([sm_arr]))
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Residue_DIFF_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Blues_r", min=-1.5, max=-0.5, mid=-1, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)

        # CLUSTER NORMALIZED
        df_quant = pd.read_csv("{}/RESULTS/Quant_Data.csv".format(self.path)).drop_duplicates()
        dsm_conc = df_quant[df_quant["Small Molecule ID"]=="DSM"].loc[:,"$P_{SM}$"].values[0]
        ndsm_conc = df_quant[df_quant["Small Molecule ID"] == "NDSM"].loc[:,"$P_{SM}$"].values[0]

        sm_arr = np.divide(df_res_contact.loc[:, "DSM"], df_res_count.loc[:, "DSM"]*dsm_conc) - np.divide(df_res_contact.loc[:, "NDSM"], df_res_count.loc[:, "NDSM"]*ndsm_conc)
        sm_arr = (np.divide(df_res_contact.loc[:, "DSM"], df_res_count.loc[:, "DSM"] * np.sum(df_res_contact.loc[:, "DSM"]) * dsm_conc)
                  - np.divide(df_res_contact.loc[:, "NDSM"], df_res_count.loc[:, "NDSM"] * np.sum(df_res_contact.loc[:, "NDSM"]) * ndsm_conc))
        sm_arr = (np.divide(df_res_contact.loc[:, "DSM"],
                            df_res_count.loc[:, "DSM"] * np.sum(df_res_contact.loc[:, "DSM"]))
                  - np.divide(df_res_contact.loc[:, "NDSM"],
                              df_res_count.loc[:, "NDSM"] * np.sum(df_res_contact.loc[:, "NDSM"])))

        print(sm_res_contact_array)
        sm_res_contact_array = np.transpose(np.asarray([sm_arr]))

        x_res_list = ["", "", "", "", "", "", ""]
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Residue_DIFF_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Reds", min=-1.4, max=0, mid=0, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)







    def plot_acid_cpm(self, acid_contact_array, col, min, max, mid):
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

        acid_contact_array[:, [4, 0]] = acid_contact_array[:, [0, 4]]
        acid_contact_array[:, [15, 1]] = acid_contact_array[:, [1, 15]]
        acid_contact_array[:, [2, 2]] = acid_contact_array[:, [2, 2]]
        acid_contact_array[:, [6, 3]] = acid_contact_array[:, [3, 6]]
        acid_contact_array[:, [16, 4]] = acid_contact_array[:, [4, 16]]
        acid_contact_array[:, [7, 4]] = acid_contact_array[:, [4, 7]]
        acid_contact_array[:, [14, 5]] = acid_contact_array[:, [5, 14]]
        acid_contact_array[:, [11, 8]] = acid_contact_array[:, [8, 11]]
        acid_contact_array[:, [18, 9]] = acid_contact_array[:, [9, 18]]
        acid_contact_array[:, [15, 10]] = acid_contact_array[:, [10, 15]]
        acid_contact_array[:, [17, 11]] = acid_contact_array[:, [11, 17]]
        acid_contact_array[:, [14, 12]] = acid_contact_array[:, [12, 14]]
        acid_contact_array[:, [18, 13]] = acid_contact_array[:, [13, 18]]
        acid_contact_array[:, [19, 14]] = acid_contact_array[:, [14, 19]]
        acid_contact_array[:, [17, 18]] = acid_contact_array[:, [18, 17]]
        acid_contact_array[[4, 0]] = acid_contact_array[[0, 4]]
        acid_contact_array[[15, 1]] = acid_contact_array[[1, 15]]
        acid_contact_array[[2, 2]] = acid_contact_array[[2, 2]]
        acid_contact_array[[6, 3]] = acid_contact_array[[3, 6]]
        acid_contact_array[[16, 4]] = acid_contact_array[[4, 16]]
        acid_contact_array[[7, 4]] = acid_contact_array[[4, 7]]
        acid_contact_array[[14, 5]] = acid_contact_array[[5, 14]]
        acid_contact_array[[11, 8]] = acid_contact_array[[8, 11]]
        acid_contact_array[[18, 9]] = acid_contact_array[[9, 18]]
        acid_contact_array[[15, 10]] = acid_contact_array[[10, 15]]
        acid_contact_array[[17, 11]] = acid_contact_array[[11, 17]]
        acid_contact_array[[14, 12]] = acid_contact_array[[12, 14]]
        acid_contact_array[[18, 13]] = acid_contact_array[[13, 18]]
        acid_contact_array[[19, 14]] = acid_contact_array[[14, 19]]
        acid_contact_array[[17, 18]] = acid_contact_array[[18, 17]]


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

        fig, ax1 = plt.subplots(figsize=(3.6, 2.8), tight_layout=True)
        ax = sns.heatmap(acid_contact_array, xticklabels=acid_list, yticklabels=acid_list, square=True, cmap=col)
                         #vmin=min, vmax=max, center=mid)
        cbar = ax.collections[0].colorbar

        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=10)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)

        return fig

    def plot_acid_sm_cpm(self, acid_contact_array, col, min, max, mid, sm_list):
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

        acid_contact_array[[4, 0]] = acid_contact_array[[0, 4]]
        acid_contact_array[[15, 1]] = acid_contact_array[[1, 15]]
        acid_contact_array[[2, 2]] = acid_contact_array[[2, 2]]
        acid_contact_array[[6, 3]] = acid_contact_array[[3, 6]]
        acid_contact_array[[16, 4]] = acid_contact_array[[4, 16]]
        acid_contact_array[[7, 4]] = acid_contact_array[[4, 7]]
        acid_contact_array[[14, 5]] = acid_contact_array[[5, 14]]
        acid_contact_array[[11, 8]] = acid_contact_array[[8, 11]]
        acid_contact_array[[18, 9]] = acid_contact_array[[9, 18]]
        acid_contact_array[[15, 10]] = acid_contact_array[[10, 15]]
        acid_contact_array[[17, 11]] = acid_contact_array[[11, 17]]
        acid_contact_array[[14, 12]] = acid_contact_array[[12, 14]]
        acid_contact_array[[18, 13]] = acid_contact_array[[13, 18]]
        acid_contact_array[[19, 14]] = acid_contact_array[[14, 19]]
        acid_contact_array[[17, 18]] = acid_contact_array[[18, 17]]


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

        fig, ax1 = plt.subplots(figsize=(2, 3.2), tight_layout=True)
        ax = sns.heatmap(acid_contact_array, xticklabels=sm_list, yticklabels=acid_list, cmap=col, square=False)#, center=mid)#cbar_kws={'aspect': 200}) #center=mid)
        #                 vmin=min, vmax=max, center=mid)
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=10)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)

        return fig


    def gen_acid_cpms(self):
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

        nAcid_dsm = np.loadtxt("{}/ANALYSIS_DSM_AGG/AcidPolNum_DSM.csv".format(self.path), delimiter=",", dtype=float)
        nAcid_ndsm = np.loadtxt("{}/ANALYSIS_NDSM_AGG/AcidPolNum_NDSM.csv".format(self.path), delimiter=",", dtype=float)
        nAcid_sg = np.loadtxt("{}/ANALYSIS_SG_AVE/AcidPolNum_sg_X.csv".format(self.path), delimiter=",", dtype=float)
        nAcid = np.loadtxt("CM_NORM/MAPS/AcidPolNum_SYSTEM.csv", delimiter=",", dtype=float)

        nBond_dsm = np.loadtxt("{}/ANALYSIS_DSM_AGG/BondNum_DSM.csv".format(self.path), delimiter=",", dtype=float)
        nBond_ndsm = np.loadtxt("{}/ANALYSIS_NDSM_AGG/BondNum_NDSM.csv".format(self.path), delimiter=",",
                                dtype=float)
        nBond_sg = np.loadtxt("{}/ANALYSIS_SG_AVE/BondNum_sg_X.csv".format(self.path), delimiter=",", dtype=float)
        nBond = np.loadtxt("CM_NORM/MAPS/BondNum_SYSTEM.csv", delimiter=",", dtype=float)

        ndsm_acid_arr = np.array(
            pd.read_csv("{}/ANALYSIS_NDSM_AGG/Acid_Contacts_Mean_NDSM.csv".format(self.path), header=None))
        dsm_acid_arr = np.array(
            pd.read_csv("{}/ANALYSIS_DSM_AGG/Acid_Contacts_Mean_DSM.csv".format(self.path), header=None))
        sg_acid_arr = np.array(
            pd.read_csv("{}/ANALYSIS_SG_AVE/Acid_Contacts_Mean_sg_X.csv".format(self.path), header=None))

        # INDIVIDUAL MAPS
        # UNNORMALIZED
        acid_contact_array = np.divide(sg_acid_arr-nBond_sg, (np.sum(sg_acid_arr-nBond_sg)))
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0, max=0.012, mid=0.006)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_SG_UNNORMALIZED_HeatMap.png".format(
            self.path), format="png", dpi=400)

        acid_contact_array = np.divide(dsm_acid_arr-nBond, (np.sum(dsm_acid_arr)))
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0, max=0.012, mid=0.006)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_DSM_UNNORMALIZED_HeatMap.png".format(
            self.path), format="png", dpi=400)

        acid_contact_array = np.divide(ndsm_acid_arr-nBond, (np.sum(ndsm_acid_arr)))
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0, max=0.012, mid=0.006)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_NDSM_UNNORMALIZED_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # NORMALIZED SYSTEM
        acid_contact_array = np.divide(sg_acid_arr-nBond, (np.sum(sg_acid_arr) * nAcid))
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0, max=0.6*10**-9, mid=0.3*10**-9)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_SG_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path), format="png", dpi=400)

        acid_contact_array = np.divide(dsm_acid_arr-nBond, (np.sum(dsm_acid_arr) * nAcid))
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0, max=0.6*10**-9, mid=0.3*10**-9)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_DSM_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path), format="png", dpi=400)

        acid_contact_array = np.divide(ndsm_acid_arr-nBond, (np.sum(ndsm_acid_arr) * nAcid))
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0, max=0.6*10**-9, mid=0.3*10**-9)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_NDSM_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # NORMALIZED CLUSTER
        acid_contact_array = np.divide(sg_acid_arr-nBond_sg, (np.sum(sg_acid_arr) * nAcid_sg))
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0, max=1.5*10**-9, mid=0.75*10**-9)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_SG_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path), format="png", dpi=400)

        acid_contact_array = np.divide(dsm_acid_arr-nBond_dsm, (np.sum(dsm_acid_arr) * nAcid_dsm))
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0, max=1.5*10**-9, mid=0.75*10**-9)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_DSM_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path), format="png", dpi=400)

        acid_contact_array = np.divide(ndsm_acid_arr-nBond_ndsm, (np.sum(ndsm_acid_arr) * nAcid_ndsm))
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0, max=1.5*10**-9, mid=0.75*10**-9)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_NDSM_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # DIFFERENCE MAPS
        # UNNORMALIZED
        dsm_arr = np.divide(dsm_acid_arr-nBond_dsm,np.sum(dsm_acid_arr-nBond_dsm))
        ndsm_arr = np.divide(ndsm_acid_arr - nBond_ndsm, np.sum(ndsm_acid_arr - nBond_ndsm))
        sg_arr = np.divide(sg_acid_arr - nBond_sg, np.sum(sg_acid_arr - nBond_sg))
        dsm_arr = dsm_acid_arr - nBond_dsm
        ndsm_arr = ndsm_acid_arr - nBond_ndsm
        sg_arr = sg_acid_arr - nBond_sg
        acid_contact_array = np.divide(np.subtract(dsm_arr, ndsm_arr), sg_arr)
        fig = self.plot_acid_cpm(acid_contact_array, col="coolwarm", min=-0.07, max=0.07, mid=0)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_DIFF_UNNORMALIZED_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # NORMALIZED SYSTEM
        sg_norm_arr = np.divide(sg_acid_arr, (np.sum(sg_acid_arr)))
        dsm_norm_arr = np.divide(dsm_acid_arr, (np.sum(dsm_acid_arr)))
        ndsm_norm_arr = np.divide(ndsm_acid_arr, (np.sum(ndsm_acid_arr)))
        acid_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
        fig = self.plot_acid_cpm(acid_contact_array, col="coolwarm", min=-0.07, max=0.07, mid=0)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_DIFF_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path), format="png", dpi=400)

        sg_norm_arr = np.divide(sg_acid_arr - nBond_sg, (np.sum(sg_acid_arr - nBond_sg)))
        dsm_norm_arr = np.divide(dsm_acid_arr - nBond_dsm, (np.sum(dsm_acid_arr - nBond_dsm)))
        ndsm_norm_arr = np.divide(ndsm_acid_arr - nBond_ndsm, (np.sum(ndsm_acid_arr - nBond_ndsm)))
        acid_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
        fig = self.plot_acid_cpm(acid_contact_array, col="coolwarm", min=0.3, max=1, mid=0.8)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_DIFF_NORMALIZED_SYSTEM_NEW_HeatMap.png".format(
            self.path), format="png", dpi=400)

        # NORMALIZED CLUSTER
        sg_norm_arr = np.divide(sg_acid_arr-nBond_sg, (np.sum(sg_acid_arr-nBond_sg) * nAcid_sg))
        dsm_norm_arr = np.divide(dsm_acid_arr-nBond_dsm, (np.sum(dsm_acid_arr-nBond_dsm) * nAcid_dsm))
        ndsm_norm_arr = np.divide(ndsm_acid_arr-nBond_ndsm, (np.sum(ndsm_acid_arr-nBond_ndsm) * nAcid_ndsm))
        acid_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
        fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0.3, max=1, mid=0.8)
        plt.savefig("{}/BIOPOLYMER_SUMMARY/FIGURES/Acid_DIFF_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path), format="png", dpi=400)









        # SM Acid Contact Array
        df_acid_contact = pd.read_csv("{}/RESULTS/SM_AcidMap_Data.csv".format(self.path))
        df_acid_count = pd.read_csv("{}/RESULTS/SM_AcidCount_Data.csv".format(self.path))
        sm_list = [
            "D-1",
            "D-2",
            "D-3",
            "D-4",
            "D-5",
            "D-6",
            "D-7",
            "D-8",
            "D-9",
            "D-10",
            "N-1",
            "N-2",
            "N-3",
            "N-4",
            "N-5",
            "N-6",
            "N-7",
            "N-8",
            "N-9",
            "N-10",
            "DSM",
            "NDSM"
        ]

        # SM ACID MAPS
        # INDIVIDUAL SM MAPS

        # UNNORMALIZED
        sm_acid_contact_array = np.divide(np.array(df_acid_contact.iloc[:, 1:]),np.sum(np.array(df_acid_contact.iloc[:, 1:])))
        print(sm_acid_contact_array)
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Acid_IND_UNNORMALIZED_HeatMap.png".format(
            self.path)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="coolwarm", min=0, max=50, mid=25, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)

        # SYSTEM NORMALIZED
        nprot = np.loadtxt("CM_NORM/MAPS/AcidNum_SYSTEM.csv", delimiter=",", dtype=float).transpose().reshape(-1, 1)
        sm_acid_contact_array = np.divide(np.array(df_acid_contact.iloc[:, 1:]), nprot)
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Acid_IND_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="coolwarm", min=0, max=0.014, mid=0, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)

        # CLUSTER NORMALIZED
        sm_acid_count_array = np.array(df_acid_count.iloc[:, 1:])
        sm_acid_contact_array = np.divide(np.array(df_acid_contact.iloc[:, 1:]), sm_acid_count_array)
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Acid_IND_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="Reds", min=0, max=0.028, mid=0.014, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)

        # DIFFERENCE SM RESIDUE MAPS




        # DIFFERENCE SM RESIDUE MAPS
        # UNNORMALIZED
        sm_arr = np.divide(df_acid_contact.loc[:, "DSM"],np.sum(df_acid_contact.loc[:, "DSM"])) - np.divide(df_acid_contact.loc[:, "NDSM"], np.sum(df_acid_contact.loc[:, "NDSM"]))
        #sm_arr = df_acid_contact.loc[:, "DSM"] - df_acid_contact.loc[:, "NDSM"]

        sm_acid_contact_array = np.transpose(np.asarray([sm_arr]))
        sm_list = [""]
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Acid_DIFF_UNNORMALIZED_HeatMap.png".format(
            self.path)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="coolwarm", min=-10, max=0, mid=0, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)

        # SYSTEM NORMALIZED
        nprot = np.loadtxt("CM_NORM/MAPS/AcidNum_SYSTEM.csv", delimiter=",", dtype=float).transpose()
        sm_arr = np.divide(df_acid_contact.loc[:, "DSM"], nprot) - np.divide(
            df_acid_contact.loc[:, "NDSM"], nprot)
        sm_acid_contact_array = np.transpose(np.asarray([sm_arr]))
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Acid_DIFF_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.path)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="Blues_r", min=-0.0026, max=-0.0012, mid=-0.0019, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)

        # CLUSTER NORMALIZED
        df_quant = pd.read_csv("{}/RESULTS/Quant_Data.csv".format(self.path)).drop_duplicates()
        dsm_conc = df_quant[df_quant["Small Molecule ID"] == "DSM"].loc[:, "$P_{SM}$"].values[0]
        ndsm_conc = df_quant[df_quant["Small Molecule ID"] == "NDSM"].loc[:, "$P_{SM}$"].values[0]
        sm_arr = np.divide(df_acid_contact.loc[:, "DSM"], df_acid_count.loc[:, "DSM"]*dsm_conc) - np.divide(
            df_acid_contact.loc[:, "NDSM"], df_acid_count.loc[:, "NDSM"]*ndsm_conc)
        sm_arr = np.divide(df_acid_contact.loc[:, "DSM"],
                           df_acid_count.loc[:, "DSM"] * np.sum(df_acid_contact.loc[:, "DSM"])) - np.divide(
            df_acid_contact.loc[:, "NDSM"], df_acid_count.loc[:, "NDSM"] * np.sum(df_acid_contact.loc[:, "NDSM"]))
        sm_arr = np.divide(df_acid_contact.loc[:, "DSM"],
                           df_acid_count.loc[:, "DSM"] * np.sum(df_acid_contact.loc[:, "DSM"])) - np.divide(
            df_acid_contact.loc[:, "NDSM"], df_acid_count.loc[:, "NDSM"] * np.sum(df_acid_contact.loc[:, "NDSM"]))
        sm_acid_contact_array = np.transpose(np.asarray([sm_arr]))
        x_res_list = ["", "", "", "", "", "", ""]
        file_name = "{}/BIOPOLYMER_SUMMARY/FIGURES/SM_Acid_DIFF_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.path)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="Reds", min=-0.001, max=0.0, mid=0.0001, sm_list=sm_list)
        plt.savefig(file_name, format="png", dpi=400)



    def plot_rdf(self, folder, sm, protein):
        df_rdf = pd.read_csv("{}/{}/RDF_{}.csv".format(self.path,folder,sm)).iloc[1:,:]
        print(df_rdf.head(29))
        dist = df_rdf["Distance"]
        prot_list = ["TDP43","FUS","TIA1","G3BP1","RNA","PABP1","TTP"]
        rdf_list = []
        prots = prot_list.remove(protein)
        labs = []
        for i in prot_list:
            labs.append("{}-{}".format(protein, i))
            for j in list(df_rdf.columns.tolist()):
                if protein in j and i in j:
                    rdf_list.append(df_rdf[j])

        col_pall = sns.color_palette("rocket", n_colors=14)
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

        for i in range(len(rdf_list)):
            ax1.plot(dist, rdf_list[i], label=labs[i], linewidth=4, zorder=1, color=col_pall[2*i])
        leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
        leg.get_frame().set_alpha(0)
        file_name = "{}/BIOPOLYMER_SUMMARY/IMAGES/RDF_{}_{}.png".format(self.path,sm,protein)
        ax1.set_ylabel("")
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)
        # ax1.axes.get_yaxis().set_visible(False)

        plt.savefig(file_name, format="png", dpi=400)



if __name__ == '__main__':
    # Biopolymer Curves
    path = sys.argv[1]
    bio_analysis = BIOPOLYMER_ANALYSIS(path)
    # Contact Maps
    bio_analysis.gen_residue_cpms()
    bio_analysis.gen_acid_cpms()

    #bio_analysis.plot_rdf("ANALYSIS_SG_AVE", "sg_X", "TDP43")
    #bio_analysis.plot_rdf("ANALYSIS_SG_AVE", "sg_X", "FUS")
    #bio_analysis.plot_rdf("ANALYSIS_SG_AVE", "sg_X", "TIA1")
    #bio_analysis.plot_rdf("ANALYSIS_SG_AVE", "sg_X", "G3BP1")
    #bio_analysis.plot_rdf("ANALYSIS_SG_AVE", "sg_X", "RNA")
    #bio_analysis.plot_rdf("ANALYSIS_SG_AVE", "sg_X", "PABP1")
    #bio_analysis.plot_rdf("ANALYSIS_SG_AVE", "sg_X", "PABP1")

    # SG
    sm = "SG"
    bio_analysis.plot_rdp(sm)
    # Biopolymer
    folder = "BIOPOLYMER_ANALYSIS_SG"
    fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA = bio_analysis.gen_biopolymer_fitters(folder, "SG")
    bio_analysis.gen_biopolymer_plots(fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA, "SG")
    # RNA
    bio_analysis.gen_rna_plots(folder, "SG")

    # DSM
    # SG
    sm = "DSM"
    bio_analysis.plot_rdp(sm)
    # Biopolymer
    folder = "BIOPOLYMER_ANALYSIS_DSM"
    fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA = bio_analysis.gen_biopolymer_fitters(folder, "DSM")
    bio_analysis.gen_biopolymer_plots(fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA, "DSM")
    # RNA
    bio_analysis.gen_rna_plots(folder, "DSM")

    # NDSM
    # SG
    sm = "NDSM"
    bio_analysis.plot_rdp(sm)
    # Biopolymer
    folder = "BIOPOLYMER_ANALYSIS_NDSM"
    fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA = bio_analysis.gen_biopolymer_fitters(folder, "NDSM")
    bio_analysis.gen_biopolymer_plots(fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA, "NDSM")
    # RNA
    bio_analysis.gen_rna_plots(folder, "NDSM")


