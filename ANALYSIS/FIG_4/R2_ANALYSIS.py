# %%
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

df = pd.read_csv("R2_Dataset.csv")

afont = {'fontname': 'Arial'}

sns.set_theme(style="ticks")
sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=10)  # fontsize of the axes title
plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
plt.rc('legend', fontsize=10)  # legend fontsize
plt.rc('font', size=10)  # controls default text sizes
plt.rc('axes', linewidth=2)

fig, axs = plt.subplots(figsize=(3.2, 2.6), tight_layout=True)
plt.ylim(0, 1)
plt.xlim(0, 16)
plt.xticks(np.arange(0, 17, 2))
sns.lineplot(ax=axs, data=df, x="x", y="R^2", linewidth=2, hue="Parameter", legend=False, palette="rocket", zorder=1)
sns.scatterplot(ax=axs, data=df, x="x", y="R^2", hue="Parameter", palette="rocket", legend=False, s=20, edgecolor="k",
                linewidth=1, zorder=2, style='Parameter', markers={'$\epsilon$': 'o', '$\sigma$': 's', '$\mu$': '^', '$r_{c}$': 'd'})
axs.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4, width=2)
plt.xlabel(None)
plt.ylabel(None)


unique_params = df["Parameter"].unique()
colors = sns.color_palette("rocket", n_colors=len(unique_params))
markers = {'$\epsilon$': 'o', '$\sigma$': 's', '$\mu$': '^', '$r_{c}$': 'd'}

handles = []
labels = []


for param, color in zip(unique_params, colors):
    # Line portion
    line_handle = Line2D([], [], color=color, linewidth=2)
    # Marker portion
    marker_style = markers.get(param, 'o')
    marker_handle = Line2D([], [], marker=marker_style, markerfacecolor=color,
                           markeredgecolor='k', linewidth=0)
    # Combine line and marker into one legend entry
    handles.append((line_handle, marker_handle))
    labels.append(param)

# Create custom legend using HandlerTuple
legend = axs.legend(
    handles,
    labels,
    loc="lower right",
    bbox_to_anchor=(0.94, 0.07),
    ncol=1,
    frameon=False,
    handler_map={tuple: HandlerTuple(ndivide=None)}
)

# Remove duplicate legend entries for 'hue'
#unique_labels = []
#unique_handles = []
#for handle, label in zip(handles, labels):
#    if label not in unique_labels:
#        unique_labels.append(label)
#        unique_handles.append(handle)

# Set the customized legend
#plt.legend(unique_handles, unique_labels)
#sns.move_legend(axs, "lower right", bbox_to_anchor=(0.94, 0.07), ncol=1, title=None, frameon=False)
# Create custom legend handles (line + marker combination)





plt.savefig("R2_Plot.png", format="png", dpi=400)