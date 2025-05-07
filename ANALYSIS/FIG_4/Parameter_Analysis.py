import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

col_pal_1 = sns.dark_palette("indigo", n_colors=10)
col_pal_2 = sns.light_palette("darkgreen", n_colors=10)

df = df_params[["Biomolecule", "E", "S", "V", "U", "R"]].groupby(by=["Biomolecule"]).mean()
print(df)

fig, axs = plt.subplots(figsize=(3.4, 2.4), tight_layout=True)

axs.tick_params(left=True, bottom=True)

axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)

r = np.linspace(4, 22, 1000)

for mol in range(20):

    eps = df.iloc[mol, 0]
    sig = df.iloc[mol, 1]
    v = df.iloc[mol, 2]
    mu = df.iloc[mol, 3]
    rc = df.iloc[mol, 4]

    alpha = 2 * v * np.power(rc / sig, 2 * mu) * np.power(
        ((1 + 2 * v) / (2 * v * (np.power(rc / sig, 2 * mu) - 1))),
        2 * v + 1)

    phi = eps * alpha * (np.power(sig / r, 2 * mu) - 1) * np.power(
        (np.power(rc / r, 2 * mu) - 1), 2 * v)

    if mol < 10:
        name = "D" + str(mol + 1)
        plt.plot(r, phi, label=name, color=col_pal_2[mol], linewidth=1.5)

    else:
        name = "ND" + str(mol - 9)
        plt.plot(r, phi, label=name, color=col_pal_1[mol - 10], linewidth=1.5)

leg = plt.figlegend(loc='upper right', ncol=2, bbox_to_anchor=(0.9, 0.9))
leg.get_frame().set_alpha(0)
plt.ylim(-0.4, 1.0)
plt.xlim(4, 16)
plt.xticks(np.arange(5, 16, 2.5))

plt.savefig("MIX_PARAMS.png", format="png", dpi=400)





col_pall = sns.color_palette("magma", n_colors=12)
col_pal_2 = col_pall[0:6]
col_pal_1 = col_pall[6:12]


df_mean = df_params[["Type", "E", "S", "V", "U", "R"]].groupby(by=["Type"]).mean()
print(df_mean)

fig, axs = plt.subplots(figsize=(3.4, 2.4), tight_layout=True)

axs.tick_params(left=True, bottom=True)

axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)

r = np.linspace(4, 22, 1000)

for mol in range(2):

    eps = df_mean.iloc[mol, 0]
    sig = df_mean.iloc[mol, 1]
    v = df_mean.iloc[mol, 2]
    mu = df_mean.iloc[mol, 3]
    rc = df_mean.iloc[mol, 4]

    alpha = 2 * v * np.power(rc / sig, 2 * mu) * np.power(
        ((1 + 2 * v) / (2 * v * (np.power(rc / sig, 2 * mu) - 1))),
        2 * v + 1)

    phi = eps * alpha * (np.power(sig / r, 2 * mu) - 1) * np.power(
        (np.power(rc / r, 2 * mu) - 1), 2 * v)

    if mol < 1:
        name = "D"
        plt.plot(r, phi, label=name, color=col_pal_2[0], linewidth=1.5)

    else:
        name = "ND"
        plt.plot(r, phi, label=name, color=col_pal_1[0], linewidth=1.5)

leg = plt.figlegend(loc='upper right', ncol=2, bbox_to_anchor=(0.9, 0.9))
leg.get_frame().set_alpha(0)
plt.ylim(-0.4, 1.0)
plt.xlim(4, 16)
plt.xticks(np.arange(5, 16, 2.5))

plt.savefig("MIX_PARAMS_MEAN.png", format="png", dpi=400)




col_pal_1 = sns.dark_palette("indigo", n_colors=5)
col_pal_2 = sns.light_palette("darkgreen", n_colors=5)

col_pall = sns.color_palette("magma", n_colors=12)
col_pal_2 = col_pall[0:6]
col_pal_1 = col_pall[6:12]

df = df_params[["Biomolecule", "E", "S", "V", "U", "R"]].groupby(by=["Biomolecule"]).mean()
print(df)

fig, axs = plt.subplots(figsize=(3.4, 2.4), tight_layout=True)

axs.tick_params(left=True, bottom=True)

axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)

r = np.linspace(4, 22, 1000)

for mol in range(20):

    eps = df.iloc[mol, 0]
    sig = df.iloc[mol, 1]
    v = df.iloc[mol, 2]
    mu = df.iloc[mol, 3]
    rc = df.iloc[mol, 4]

    alpha = 2 * v * np.power(rc / sig, 2 * mu) * np.power(
        ((1 + 2 * v) / (2 * v * (np.power(rc / sig, 2 * mu) - 1))),
        2 * v + 1)

    phi = eps * alpha * (np.power(sig / r, 2 * mu) - 1) * np.power(
        (np.power(rc / r, 2 * mu) - 1), 2 * v)

    if 0 <= mol < 5:
        name = "D" + str(mol + 1)
        plt.plot(r, phi, label=name, color=col_pal_2[mol], linewidth=1.5)

    elif 10 <= mol < 15:
        name = "ND" + str(mol - 9)
        plt.plot(r, phi, label=name, color=col_pal_1[mol - 10], linewidth=1.5)

leg = plt.figlegend(loc='upper right', ncol=2, bbox_to_anchor=(0.85, 0.9))
leg.get_frame().set_alpha(0)
plt.ylim(-0.4, 0.3)
plt.xlim(5, 12.5)
plt.xticks(np.arange(5, 13, 2.5))

plt.savefig("MIX_PARAMS_SUMMARY_16.png", format="png", dpi=400)


col_pall = sns.color_palette("magma", n_colors=12)
col_pal_2 = col_pall[0:6]
col_pal_1 = col_pall[6:12]

df = df_params[["Biomolecule", "E", "S", "V", "U", "R"]].groupby(by=["Biomolecule"]).mean()
print(df)

fig, axs = plt.subplots(figsize=(3.4, 2.6), tight_layout=True)

axs.tick_params(left=True, bottom=True)

axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)

r = np.linspace(4, 22, 1000)

for mol in range(20):

    eps = df.iloc[mol, 0]
    sig = df.iloc[mol, 1]
    v = df.iloc[mol, 2]
    mu = df.iloc[mol, 3]
    rc = df.iloc[mol, 4]

    alpha = 2 * v * np.power(rc / sig, 2 * mu) * np.power(
        ((1 + 2 * v) / (2 * v * (np.power(rc / sig, 2 * mu) - 1))),
        2 * v + 1)

    phi = eps * alpha * (np.power(sig / r, 2 * mu) - 1) * np.power(
        (np.power(rc / r, 2 * mu) - 1), 2 * v)

    if 5 <= mol < 10:
        name = "D" + str(mol + 1)
        plt.plot(r, phi, label=name, color=col_pal_2[mol-5], linewidth=1.5)

    elif 15 <= mol < 20:
        name = "ND" + str(mol - 9)
        plt.plot(r, phi, label=name, color=col_pal_1[mol - 15], linewidth=1.5)

leg = plt.figlegend(loc='upper right', ncol=2, bbox_to_anchor=(0.85, 0.9))
leg.get_frame().set_alpha(0)
plt.ylim(-0.4, 0.3)
plt.xlim(5, 12.5)
plt.xticks(np.arange(5, 13, 2.5))

plt.savefig("MIX_PARAMS_SUMMARY_510.png", format="png", dpi=400)










col_pall = sns.color_palette("magma", n_colors=20)

df = df_params[["Biomolecule", "E", "S", "V", "U", "R"]].groupby(by=["Biomolecule"]).mean()
print(df)

#fig, axs = plt.subplots(figsize=(3.4, 2.4), tight_layout=True)
fig, axs = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=False, tight_layout=True)
axs[0].tick_params(left=True, bottom=True)
axs[1].tick_params(left=True, bottom=True)

axs[0].tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
axs[1].tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)


r = np.linspace(4, 22, 1000)

for mol in range(20):

    eps = df.iloc[mol, 0]
    sig = df.iloc[mol, 1]
    v = df.iloc[mol, 2]
    mu = df.iloc[mol, 3]
    rc = df.iloc[mol, 4]

    alpha = 2 * v * np.power(rc / sig, 2 * mu) * np.power(
        ((1 + 2 * v) / (2 * v * (np.power(rc / sig, 2 * mu) - 1))),
        2 * v + 1)

    phi = eps * alpha * (np.power(sig / r, 2 * mu) - 1) * np.power(
        (np.power(rc / r, 2 * mu) - 1), 2 * v)

    if 0 <= mol < 10:
        name = "D" + str(mol + 1)
        axs[0].plot(r, phi, label=name, color=col_pall[mol], linewidth=1.5)

    elif 10 <= mol < 20:
        name = "ND" + str(mol - 9)
        axs[1].plot(r, phi, label=name, color=col_pall[mol], linewidth=1.5)

#leg = plt.figlegend(loc='upper right', ncol=2, bbox_to_anchor=(0.85, 0.9))
#leg.get_frame().set_alpha(0)
#leg1 = axs[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(0.01, 1),columnspacing=0.3)
#leg1.get_frame().set_alpha(0)
#
#leg2 = axs[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(0.01, 1),columnspacing=0.3)
#leg2.get_frame().set_alpha(0)

axs[0].set_xlim(4, 12)
axs[0].set_ylim(-0.4, 0.2)
#axs[0].set_xticks([])
axs[0].set_xticklabels([])
axs[0].set_yticks(np.arange(-0.4, 0.2, 0.2))

axs[1].set_xlim(4, 12)
axs[1].set_ylim(-0.4, 0.2)
axs[1].set_xticks(np.arange(4, 13, 2))
axs[1].set_yticks(np.arange(-0.4, 0.2, 0.2))

plt.subplots_adjust(hspace=-1, top=4, bottom=0)

#plt.yticks(np.arange(-0.4, 0.2, 0.2))
#plt.xticks(np.arange(0, 13, 2.5))

plt.savefig("MIX_PARAMS_SUMMARY.png", format="png", dpi=400)