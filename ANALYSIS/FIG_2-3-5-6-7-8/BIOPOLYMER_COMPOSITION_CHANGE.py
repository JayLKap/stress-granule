import sys

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from RDP_PLOT import RDP
from RDP_NORMALIZE import RDP_NORMALIZE
from scipy.interpolate import UnivariateSpline


def comp_ave(path, category, sm_list):
    df_temp = pd.DataFrame()
    df = pd.DataFrame()
    for sm in sm_list:
        df_sm = pd.read_csv("{}/ANALYSIS_{}_AVE/BioNumDF_{}.csv".format(path, category, sm))
        biopolymer_col = df_sm["Biopolymer"]
        mean_sm = df_sm["Mean"]
        sem_sm = df_sm["SEM"]

        df_temp["Biopolymer"] = biopolymer_col
        df_temp["Mean_{}".format(sm)] = mean_sm
        df_temp["SEM_{}".format(sm)] = sem_sm


    row_avg = df_temp.iloc[:,1:].mean(axis=1)
    row_sem = df_temp.iloc[:,1:].sem(axis=1)

    df_temp["Mean_{}".format(category)] = row_avg
    df_temp["SEM_{}".format(category)] = row_sem

    df_temp.to_csv(
        "{}/ANALYSIS_{}_AGG/BioNumDF_FULL_{}.csv".format(path, category, category),
        index=False)

    df["Biopolymer"] = biopolymer_col
    df["Mean_{}".format(category)] = row_avg
    df["SEM_{}".format(category)] = row_sem

    df.to_csv(
        "{}/ANALYSIS_{}_AGG/BioNumDF_{}.csv".format(path, category, category),
        index=False)


def plot_violin(df):
    df_long = pd.melt(df, id_vars='Compound Class', var_name='Protein', value_name='Value')
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

    fig, ax1 = plt.subplots(figsize=(3.2, 3.2))
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    sns.violinplot(ax=ax1, data=df_long, x = 'Protein', y = 'Value', hue="Compound Class", palette=col_pal_sm, width=1, scale='width', saturation=100,
                   linewidth=1, edgecolor="k", inner="box",
                   cut=0, dodge=True)
    ax1.get_legend().remove()
    ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                    length=4,
                    width=2)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    plt.xticks(rotation=45)
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.savefig("{}/FIGURES/{}_VP.png".format(path, "Biopolymer_Composition"), format="png", dpi=400)




path = sys.argv[1]
dt = int(sys.argv[2])
tmin = int(sys.argv[3])
tmax = int(sys.argv[4])

if "THEORETICAL" not in path:
    dsm_list = ["dsm_anisomycin", "dsm_daunorubicin", "dsm_dihydrolipoicacid", "dsm_hydroxyquinoline", "dsm_lipoamide",
                "dsm_lipoicacid", "dsm_mitoxantrone", "dsm_pararosaniline", "dsm_pyrivinium", "dsm_quinicrine"]

    ndsm_list = ["ndsm_dmso", "ndsm_valeric", "ndsm_ethylenediamine", "ndsm_propanedithiol",
                 "ndsm_hexanediol", "ndsm_diethylaminopentane", "ndsm_aminoacridine", "ndsm_anthraquinone",
                 "ndsm_acetylenapthacene", "ndsm_anacardic"]

if "THEORETICAL" in path:
    dsm_list = []
    ndsm_list = []
    list_path = path.replace("_THEORETICAL", "")
    with open('{}/dsm_list.txt'.format(list_path), 'r') as f:
        for i in f.readlines():
            dsm_list.append(i.strip())

    with open('{}/ndsm_list.txt'.format(list_path), 'r') as f:
        for i in f.readlines():
            ndsm_list.append(i.strip())

category = "DSM"
comp_ave(path, category, dsm_list)

category = "NDSM"
comp_ave(path, category, ndsm_list)

df_sg = pd.read_csv("{}/ANALYSIS_SG_AVE/BioNumDF_sg_X.csv".format(path))
df_dsm = pd.read_csv("{}/ANALYSIS_DSM_AGG/BioNumDF_FULL_DSM.csv".format(path))
df_ndsm = pd.read_csv("{}/ANALYSIS_NDSM_AGG/BioNumDF_FULL_NDSM.csv".format(path))

print(df_sg)
print(df_dsm)
print(df_ndsm)


df = pd.DataFrame(columns=["Compound Class","SG","TDP43","FUS","TIA1","G3BP1","PABP1","TTP","RNA"])

div = np.array([134,16,16,16,32,16,16,21])
div = np.array(df_sg["Mean"])

#row_sg = np.divide(np.array(df_sg["Mean"]), div)
#df.loc[len(df.index)] = ["X"] + list(row_sg)

for sm in dsm_list:
    row_dsm = np.divide(np.array(df_dsm["Mean_{}".format(sm)]), div)
    df.loc[len(df.index)] = ["DSM"] + list(row_dsm)

for sm in ndsm_list:
    row_ndsm = np.divide(np.array(df_ndsm["Mean_{}".format(sm)]), div)
    df.loc[len(df.index)] = ["NDSM"] + list(row_ndsm)

print(df)
plot_violin(df)