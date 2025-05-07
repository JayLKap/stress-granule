import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

class SummaryPlots():
    def __init__(self, path):
        self.path = path
        self.df_og = pd.read_csv("{}/RESULTS/Quant_Data.csv".format(path), delimiter=",")
        self.variables = ["$P_{SG}$",
                     "$R_{cond}$ $(\AA)$",
                     "$W_{interface}$ $(\AA)$",
                     "$\gamma_{1}$ $(mN/m)$",
                     "$\gamma_{2}$ $(mN/m)$",
                     "$\gamma_ave$ $(mN/m)$",
                     "$\Delta G_{trans}$ $(kJ/mol)$",
                     "$P_{SM}$",
                     "$\phi_{D}$",
                     "$N_{D}$",
                     "$R_{g}$",
                     "$\phi_{R}$",
                     "$D$ $\mu m^{2} / s$",
                     "$tau$ $ns$",
                     "$\l_{Cond}$ A",
                     "$\eta_{D}$ Pa s",
                     "$\eta_{GK}$ Pa s",
                     ]

        self.names = ["P_SG",
                 "R_cond",
                 "W_int",
                 "gamma_1",
                 "gamma_2",
                 "gamma_ave",
                 "Delta G",
                 "P_SM",
                 "phi_D",
                 "N_D",
                 "R_g",
                 "phi_R",
                 "D",
                 "tau",
                 "l_cond",
                 "eta_D",
                 "eta_GK",
                 ]

    def clean_df(self):
        df = self.df_og.copy()
        """
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

        mask = df["Small Molecule ID"].isin(row_sel)
        # Convert the column 'A' to a categorical type with the predefined order
        df['Small Molecule ID'] = pd.Categorical(df['Small Molecule ID'], categories=row_sel, ordered=True)

        # Sort the DataFrame by column 'A' using the predefined order
        df = df.sort_values(by='Small Molecule ID')
        """
        print(df)

        df = df[df['Small Molecule ID'] != "DSM"]
        df = df[df['Small Molecule ID'] != "DSM_AVG"]
        df = df[df['Small Molecule ID'] != "NDSM"]
        df = df[df['Small Molecule ID'] != "NDSM_AVG"]

        df = df.drop_duplicates()

        df["Compound Class"].replace('Dissolving',"DSM", inplace=True)
        df["Compound Class"].replace('Non-Dissolving',"NDSM", inplace=True)

        print(df)

        features = df.loc[:,
                    ["Compound Class",
                     "$P_{SG}$",
                     "$R_{cond}$ $(\AA)$",
                     "$W_{interface}$ $(\AA)$",
                     "$\gamma_{1}$ $(mN/m)$",
                     "$\gamma_{2}$ $(mN/m)$",
                     "$\gamma_ave$ $(mN/m)$",
                     "$\Delta G_{trans}$ $(kJ/mol)$",
                     "$P_{SM}$",
                     "$\phi_{D}$",
                     "$N_{D}$",
                     "$R_{g}$",
                     "$\phi_{R}$",
                     "$D$ $\mu m^{2} / s$",
                     "$tau$ $ns$",
                     "$\l_{Cond}$ A",
                     "$\eta_{D}$ Pa s",
                     "$\eta_{GK}$ Pa s"]]

        df_mean = features.groupby("Compound Class",observed=False).mean()
        df_std = features.groupby("Compound Class",observed=False).std()

        df_mean["ID"] = ["DSM","NDSM","SG"]
        df_std["ID"] = ["DSM","NDSM","SG"]

        #df_mean.iloc[0], df_mean.iloc[1] = df_mean.iloc[1], df_mean.iloc[0].copy()
        #df_std.iloc[0], df_std.iloc[1] = df_std.iloc[1], df_std.iloc[0].copy()

        #df_mean.iloc[0], df_mean.iloc[2] = df_mean.iloc[2], df_mean.iloc[0].copy()
        #df_std.iloc[0], df_std.iloc[2] = df_std.iloc[2], df_std.iloc[0].copy()

        self.df = df
        print(self.df)
        categories_to_plot = ["DSM","NDSM"]
        self.df_filtered = df[df['Compound Class'].isin(categories_to_plot)]
        print(self.df_filtered)
        self.df_mean = df_mean
        self.df_std = df_std
        print(self.df_mean)


    def plot_mean(self):
        col_pall = sns.color_palette(["#808080","#bfe49b","#40641b"], 3)
        df = self.df
        df_mean = self.df_mean
        df_std = self.df_std
        variables = self.variables

        # Mean Calculation
        for var in range(len(self.variables)):
            sns.set_theme(style="ticks")
            sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
            plt.rc('axes', titlesize=10)  # fontsize of the axes title
            plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
            plt.rc('legend', fontsize=8)  # legend fontsize
            plt.rc('font', size=10)  # controls default text sizes
            plt.rc('axes', linewidth=2)

            fig, ax1 = plt.subplots(figsize=(2.8, 2.8))
            plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
            sns.barplot(ax=ax1, data=df_mean, x="ID", y=variables[var], hue="ID", palette=col_pall,
                             dodge=False,
                             width=0.5, saturation=100, linewidth=2, edgecolor="k")
            plotline, caps, barlinecols = ax1.errorbar(df_mean["ID"], df_mean[variables[var]], yerr=df_std[variables[var]], fmt="none", color='k', elinewidth=1, capsize=2, capthick=1)
            plt.setp(barlinecols[0], capstyle='round')
            ax1.get_legend().remove()
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            plt.tight_layout()
            plt.savefig("{}/FIGURES/{}_AVG.png".format(self.path, self.names[var]), format="png", dpi=400)
            # add lables to each point

    def plot_full(self):
        col_pall = sns.color_palette(["#808080","#bfe49b","#40641b"], 3)
        # Full Bar Calculation
        for var in range(len(self.variables)):
            sns.set_theme(style="ticks")
            sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
            plt.rc('axes', titlesize=10)  # fontsize of the axes title
            plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
            plt.rc('legend', fontsize=8)  # legend fontsize
            plt.rc('font', size=10)  # controls default text sizes
            plt.rc('axes', linewidth=2)

            fig, ax1 = plt.subplots(figsize=(2.8, 2.8))
            plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
            sns.barplot(ax=ax1, data=self.df, x="Small Molecule ID", y=self.variables[var], hue="Compound Class", palette=col_pall,
                        dodge=False,
                        width=0.5, saturation=100, linewidth=2, edgecolor="k")
            plotline, caps, barlinecols = ax1.errorbar(self.df["Small Molecule ID"], self.df[self.variables[var]],
                                                       yerr=self.df["SIG" + self.variables[var]], fmt="none", color='k', elinewidth=1,
                                                       capsize=2, capthick=1)
            plt.setp(barlinecols[0], capstyle='round')
            ax1.get_legend().remove()
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            plt.xticks(rotation=90)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            plt.tight_layout()
            plt.savefig("{}/FIGURES/{}_ALL.png".format(self.path, self.names[var]), format="png", dpi=400)
            # add lables to each point


    # Violin Plot Calculation
    def plot_violin(self):
        for var in range(len(self.variables)):
            col_pal_sm = sns.color_palette(["#40641b","#bfe49b"], 2)
            sns.set_theme(style="ticks")
            sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
            plt.rc('axes', titlesize=10)  # fontsize of the axes title
            plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
            plt.rc('legend', fontsize=8)  # legend fontsize
            plt.rc('font', size=10)  # controls default text sizes
            plt.rc('axes', linewidth=2)

            fig, ax1 = plt.subplots(figsize=(2.1, 3.1))
            plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
            sns.violinplot(ax=ax1, data=self.df_filtered, x="Compound Class",
                            y=self.variables[var], hue="Compound Class", palette=col_pal_sm,
                            dodge=False, width=0.35, saturation=100,
                           linewidth=2, edgecolor="k", inner="box",
                           cut=0)
            y_line = self.df_mean.loc[self.df_mean["ID"]=="SG", self.variables[var]][0]
            #print(y_line[0])
            plt.axhline(y=y_line, color=sns.color_palette(["#808080"], 1)[0], linewidth=2)
            #ax1.get_legend().remove()
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            if self.variables[var] == "$P_{SG}$":
                plt.ylim([80, 220])
            plt.savefig("{}/FIGURES/{}_VP.png".format(self.path, self.names[var]), format="png", dpi=400)


    def k_means_simulation(self, feature_list):
        print(self.df)
        df = self.df[1:]
        df_part1 = df.iloc[0:10]  # Rows 2-11 (Python indexing: includes 2-10)
        df_part2 = df.iloc[10:20]  # Rows 11-20 (Python indexing: includes 11-19)

        # Swap and reassemble the DataFrame
        df = pd.concat([df_part1, df_part2]).reset_index(drop=True)
        print(df)
        features = df.loc[:,feature_list]

        features = np.array(features)

        labels = list((df.loc[:, "D_Binary"]))

        for i in range(len(labels)):
            labels[i] = int(np.round(labels[i]))

        kmeans = KMeans(
            init="random",
            n_clusters=2,
            n_init=20,
            max_iter=400,
            random_state=None)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        pca = PCA(n_components=2)
        pca_features = pd.DataFrame(pca.fit_transform(scaled_features),
                                    columns=["Principle Component 1", "Principle Component 2"])

        #mds = MDS(random_state=0)
        #scaled_df = mds.fit_transform(features)
        #kmeans.fit(pca_features)

        kmeans.fit(pca_features)
        loadings = pd.DataFrame(pca.components_, columns=feature_list)

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        feature_labels = ["$R_{cond}$",
                        "$P_{SM}$",
                        "$\phi_{D}$",
                        "$D$",
                        "$\eta_{D}$"]

        print(loadings)

        fig, ax1 = plt.subplots(figsize=(3.2, 3.2))
        plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
        sns.barplot(ax=ax1, x=feature_labels, y=np.abs(loadings.iloc[0]),
                    dodge=True,
                    width=0.6, saturation=100, color = 'grey', edgecolor ="k")
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        plt.savefig("{}/FIGURES/KMeans_PCA1_Loading.png".format(self.path), format="png", dpi=400)

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
        plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
        sns.barplot(ax=ax1, x=feature_labels, y=np.abs(loadings.iloc[1]),
                    dodge=True,
                    width=0.6, saturation=100, color = 'grey', edgecolor ="k")
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        plt.savefig("{}/FIGURES/KMeans_PCA2_Loading.png".format(self.path), format="png", dpi=400)


        print(f"Lowest SSE: {kmeans.inertia_:.6f}")

        print(f"Lowest SSE: {kmeans.inertia_:.6f}")
        ndsm_center = kmeans.cluster_centers_[0]
        dsm_center = kmeans.cluster_centers_[1]
        print(f"DSM Center: pca1={dsm_center[0]:.6f}; pca2={dsm_center[1]:.6f}")
        print(f"NDSM Center: pca1={ndsm_center[0]:.6f}; pca2={ndsm_center[1]:.6f}")

        pred_labs = []
        true_labs = []

        predictions = list(kmeans.labels_)
        for i in range(len(predictions)):
            if predictions[i] == 1:
                pred_labs.append("DSM")
            else:
                pred_labs.append("NDSM")
            if labels[i] == 1:
                true_labs.append("DSM")
            else:
                true_labs.append("NDSM")

        fpr = 0
        fnr = 0
        mcr = 0
        ccr = 0
        for i in range(len(predictions)):
            if predictions[i] != labels[i]:
                mcr += 1
                if predictions[i] == 1:
                    fpr += 1
                elif predictions[i] == 0:
                    fnr += 1
            else:
                ccr += 1

        fpr = fpr / (len(predictions) / 2) * 100
        fnr = fnr / (len(predictions) / 2) * 100
        mcr = mcr / len(predictions) * 100
        ccr = ccr / len(predictions) * 100

        print(f"False Positive Rate: {fpr:.6f}")
        print(f"False Negative Rate: {fnr:.6f}")
        print(f"Mis-classification Rate: {mcr:.6f}")
        print(f"Classification Rate: {ccr:.6f}")
        score = silhouette_score(scaled_features, kmeans.labels_)
        print(f"Silhouette Score: {score:.6f}")

        pca_features["Predicted Label"] = list(pred_labs)
        pca_features["True Label"] = list(true_labs)
        pca_features["Small Molecule ID"] = list(df["Small Molecule ID"])
        pca_features["Small Molecule Name"] = list(df["Small Molecule Name"])
        print(pca_features)



        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        #fig, ax1 = plt.subplots(figsize=(3.2, 3.2))
        fig, ax1 = plt.subplots(figsize=(6.86, 2.4))
        col_pall_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)

        sns.scatterplot(
            x=pca_features.loc[:, "Principle Component 1"],
            y=pca_features.loc[:, "Principle Component 2"],
            hue=pca_features.loc[:, "Predicted Label"],
            style=pca_features.loc[:, "True Label"],
            palette=col_pall_sm,
            s=80,
        )
        plt.xlim([-4, 4])
        plt.ylim([-2, 4])
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)
        ax1.legend(loc='upper right', ncol=2, frameon=False, fontsize=8)
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        plt.tight_layout()
        # leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
        # leg.get_frame().set_alpha(0)
        plt.savefig("{}/FIGURES/KMeans.png".format(self.path), format="png", dpi=400)

        df_annotate = self.df.set_index("Small Molecule ID").iloc[1:,:]
        plt.rc('font', size=6)  # controls default text sizes
        for i, txt in enumerate(df_annotate.index):
            plt.annotate(txt, (pca_features.iloc[i, 0] - 0.4, pca_features.iloc[i, 1] + 0.05))
        plt.savefig("{}/FIGURES/KMeans_Annotated.png".format(self.path), format="png", dpi=400)

        dsm_list = []
        ndsm_list = []

        pca_sorted = pca_features.sort_values(by="Predicted Label")

        dsm_list = list(pca_sorted[pca_sorted["Predicted Label"] == "DSM"]["Small Molecule Name"])
        with open('{}/dsm_list.txt'.format(self.path), 'w') as f:
            for item in dsm_list:
                f.write(f"{item}\n")

        ndsm_list = list(pca_sorted[pca_sorted["Predicted Label"] == "NDSM"]["Small Molecule Name"])
        with open('{}/ndsm_list.txt'.format(self.path), 'w') as f:
            for item in ndsm_list:
                f.write(f"{item}\n")







    def k_means_parameters(self):
        parameters = pd.read_csv("parameters.csv").loc[:, ["Biomolecule", "E", "S", "V", "R", "U", "M"]]

        ave_parameters = parameters.groupby("Biomolecule",observed=False).mean()

        df_annotate = self.df.set_index("Small Molecule ID")

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
                   "dsm_lipoic": "D3",
                   "dsm_dihydrolipoic": "D4",
                   "dsm_anisomycin": "D5",
                   "dsm_pararosaniline": "D6",
                   "dsm_pyrivinium": "D7",
                   "dsm_quinicrine": "D8",
                   "dsm_mitoxantrone": "D9",
                   "dsm_daunorubicin": "D10",
                   "DSM": "DSM",
                   "NDSM": "NDSM"
                   }

        ids = []
        dis = []
        for index, row in ave_parameters.iterrows():
            if "dsm" in index:
                ids.append(sm_dict[index])
                if "ndsm" in index:
                    dis.append(0)
                else:
                    dis.append(1)

        ave_parameters["Molecule ID"] = ids

        ave_parameters["Category"] = dis

        print(ave_parameters)

        mean_parameters = ave_parameters.loc[:, ["E", "S", "V", "U", "R", "M", "Category"]].groupby("Category",observed=False).mean()

        mean_parameters["Molecule ID"] = ["NDSM", "DSM"]

        mean_parameters["Category"] = [0, 1]

        print(mean_parameters)

        variables = ["E",
                     "S",
                     "V",
                     "U",
                     "R"]

        col_pall1 = sns.color_palette("rocket", n_colors=3)
        col_pall2 = sns.color_palette("Blues", n_colors=1)
        col_pall = [list(col_pall1[0]), list(col_pall1[2])]

        for var in range(len(variables)):
            sns.set_theme(style="ticks")
            sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
            plt.rc('axes', titlesize=10)  # fontsize of the axes title
            plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
            plt.rc('legend', fontsize=8)  # legend fontsize
            plt.rc('font', size=10)  # controls default text sizes
            plt.rc('axes', linewidth=2)

            fig, ax1 = plt.subplots(figsize=(3.6, 3.6))
            plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
            sns.barplot(ax=ax1, data=mean_parameters, x="Molecule ID", y=variables[var], hue="Molecule ID",
                        palette=col_pall,
                        dodge=False,
                        width=0.5, saturation=100)
            ax1.get_legend().remove()
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            plt.savefig("{}/FIGURES/{}.png".format(self.path, variables[var]), format="png", dpi=400)

        features = ave_parameters.loc[:, ["E",
                                          "S",
                                          "U",
                                          "R"]]

        features = np.array(features)

        labels = list((ave_parameters.loc[:, "Category"]))

        kmeans = KMeans(
            init="random",
            n_clusters=2,
            n_init=20,
            max_iter=400,
            random_state=None)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        pca = PCA(n_components=2)
        pca_features = pd.DataFrame(pca.fit_transform(scaled_features),
                                    columns=["Principle Component 1", "Principle Component 2"])

        kmeans.fit(pca_features)

        print(f"Lowest SSE: {kmeans.inertia_:.6f}")
        ndsm_center = kmeans.cluster_centers_[0]
        dsm_center = kmeans.cluster_centers_[1]
        print(f"DSM Center: pca1={dsm_center[0]:.6f}; pca2={dsm_center[1]:.6f}")
        print(f"NDSM Center: pca1={ndsm_center[0]:.6f}; pca2={ndsm_center[1]:.6f}")

        print(list(kmeans.labels_))
        print(labels)
        pred_labs = []
        true_labs = []

        predictions = list(kmeans.labels_)
        for i in range(len(predictions)):
            if predictions[i] == 1:
                pred_labs.append("DSM")
            else:
                pred_labs.append("NDSM")
            if labels[i] == 1:
                true_labs.append("DSM")
            else:
                true_labs.append("NDSM")

        fpr = 0
        fnr = 0
        mcr = 0
        ccr = 0
        for i in range(len(predictions)):
            if predictions[i] != labels[i]:
                mcr += 1
                if predictions[i] == 1:
                    fpr += 1
                elif predictions[i] == 0:
                    fnr += 1
            else:
                ccr += 1

        fpr = fpr / (len(predictions) / 2) * 100
        fnr = fnr / (len(predictions) / 2) * 100
        mcr = mcr / len(predictions) * 100
        ccr = ccr / len(predictions) * 100

        print(f"False Positive Rate: {fpr:.6f}")
        print(f"False Negative Rate: {fnr:.6f}")
        print(f"Mis-classification Rate: {mcr:.6f}")
        print(f"Classification Rate: {ccr:.6f}")

        pca_features["Predicted Cluster"] = list(pred_labs)
        pca_features["True Label"] = list(true_labs)
        pca_features["Small Molecule ID"] = list()
        print(pca_features)

        score = silhouette_score(scaled_features, kmeans.labels_)
        print(f"Silhouette Score: {score:.6f}")

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        fig, ax1 = plt.subplots(figsize=(3.6, 3.6))
        col_pall1 = sns.color_palette("rocket", n_colors=3)

        sns.scatterplot(
            x=pca_features.loc[:, "Principle Component 1"],
            y=pca_features.loc[:, "Principle Component 2"],
            hue=pca_features.loc[:, "Predicted Cluster"],
            style=pca_features.loc[:, "True Label"],
            palette=col_pall1,
            s=80,
        )

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)
        ax1.legend(loc='upper right', frameon=False, fontsize=6)
        # leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
        # leg.get_frame().set_alpha(0)
        for i, txt in enumerate(df_annotate.index):
            if "ND" in txt:
                txt = "N" + str(txt[-1])
                if "10" in txt:
                    txt = "N10"
            print(txt)
            plt.annotate(txt, (pca_features.iloc[i, 0] - 0.4, pca_features.iloc[i, 1] + 0.05))
        # plt.savefig("FIGURES/KMeans_Annotated.png", format="png", dpi=400)
        plt.savefig("{}/FIGURES/PARAM_KMeans.png".format(self.path), format="png", dpi=400)




if __name__ == '__main__':

    path = sys.argv[1]

    feature_list_total = ["$P_{SG}$",
                "$R_{cond}$ $(\AA)$",
                "$W_{interface}$ $(\AA)$",
                "$\gamma_{1}$ $(mN/m)$",
                "$\gamma_{2}$ $(mN/m)$",
                "$\gamma_ave$ $(mN/m)$",
                "$\Delta G_{trans}$ $(kJ/mol)$",
                "$P_{SM}$",
                "$\phi_{D}$",
                "$N_{D}$",
                "$R_{g}$",
                "$\phi_{R}$",
                "$D$ $\mu m^{2} / s$",
                "$tau$ $ns$",
                "$\l_{Cond}$ A",
                "$\eta_{D}$ Pa s",
                "$\eta_{GK}$ Pa s"]

    # Optimal

    feature_list_optimal = ["$R_{cond}$ $(\AA)$",
                            "$P_{SM}$",
                            "$\phi_{D}$",
                            "$D$ $\mu m^{2} / s$",
                            "$\eta_{D}$ Pa s"]

    classification = SummaryPlots(path)

    classification.clean_df()
    #classification.plot_full()
    #classification.plot_mean()
    classification.plot_violin()
    classification.k_means_simulation(feature_list_optimal)
    #classification.k_means_parameters()

