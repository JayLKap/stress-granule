import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")
sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=16)  # fontsize of the axes title
plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.rc('legend', fontsize=16)  # legend fontsize
plt.rc('font', size=16)  # controls default text sizes

df = pd.read_csv("R2_Dataset.csv")

fig, axs = plt.subplots(figsize=(6, 6), dpi=400, tight_layout=True)
sns.lineplot(ax=axs, data=df, y="R^2", x="x", hue="Parameter", linewidth=4, marker="o", palette="rocket")
plt.xlabel("MORDRED Input Parameter Number")
plt.ylabel("$R^{2}$")
axs.spines[['right', 'top']].set_visible(False)
axs.tick_params(left=True, bottom=True)
axs.legend().remove()
#plt.legend(labels=["$\epsilon$", "$\sigma$", "$\mu$", "$r_{c}$"])

leg = plt.figlegend(loc='center right', ncol=1, bbox_to_anchor=(0.7, 0., 0.2, 0.98))
leg.get_frame().set_alpha(0)

plt.savefig("MORDRED_RFR.png", format="png", dpi=400)

plt.show()

