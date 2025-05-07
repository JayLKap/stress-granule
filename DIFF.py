import numpy as np
import scipy.integrate as spi
from scipy import interpolate
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import numpy as np
import scipy.special
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math
from scipy.stats import linregress
import seaborn as sns
from scipy import integrate
import os
import fnmatch
from sklearn.utils import resample
from tqdm import tqdm

class DIFF:
    def __init__(self, path, msd_array, time_array, rg_array):
        self.kb = 1.3806 * 10 ** (-23)
        self.msd_array = msd_array
        self.rg_array = rg_array
        self.time_array = time_array
        self.path = path

    def boot_strap(self, sm, iterations, T, dt, slope_iterations, tolerance, boot_r2, boot_tolerance, seg):
        msd_plot = []
        rg_array =[]
        alpha_arr = []
        c_arr = []
        time_array = []
        m_arr = []
        b_arr = []
        index_start_arr = []
        index_end_arr = []
        tau_start_arr = []
        tau_end_arr = []
        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        for iter in tqdm(range(iterations), desc="Running Diffusion Bootstrap", unit="iterations"):

            msd_array = np.mean(np.array(resample(self.msd_array, replace=True)), axis=0)
            msd_arr = []
            time_arr = np.arange(1*10**-9,seg*10**-9,1*10**-9)
            for i in range(0, len(msd_array), seg):
                msd_arr.append(np.array(msd_array[i:i+seg])-msd_array[i])
            for i in range(len(msd_arr)):
                msd_arr[i] = msd_arr[i][1:]
            diff = np.abs(len(msd_arr[0]) - len(msd_arr[-1]))
            if diff != 0:
                for i in range((diff)):
                    msd_arr[-1] = np.append(msd_arr[-1], msd_arr[-1][-1])
            msd_arr = np.mean(np.vstack(msd_arr), axis=0)
            msd_array = msd_arr
            msd_plot.append(msd_array)
            rg_array.append(np.mean(np.array(resample(self.rg_array, replace=True)), axis=0))
            time_array = time_arr
            alpha, c, index_start, index_end = self.calc_diffusivity(msd_array=msd_array, time_array=time_array, dt=dt,
                                                                  iterations=slope_iterations, tolerance=tolerance,
                                                                  boot_r2=boot_r2, boot_tolerance=boot_tolerance)
            m, b = self.calc_tau(msd_array=msd_array[index_end:], time_array=time_array[index_end:],
                                                                iterations=slope_iterations)
            alpha_arr.append(alpha)
            c_arr.append(c)
            index_start_arr.append(index_start)
            index_end_arr.append(index_end)
            m_arr.append(m)
            b_arr.append(b)


        index_start = int(np.mean(index_start_arr))
        index_end = int(np.mean(index_end_arr))

        x = time_array
        y = np.mean(np.array(msd_plot), axis=0)
        y_fit_D = time_array[:index_end]*np.mean(alpha_arr) + np.mean(c_arr)
        y_fit_T = time_array[index_end:]*np.mean(m_arr) + np.mean(b_arr)

        x_log = np.log10(x)
        y_log = np.log10(y)
        y_fit_logD = np.log10(time_array*np.mean(alpha_arr)*6)

        D = (np.mean(alpha_arr) / 6)
        D_sme = (np.std(alpha_arr) / (6*np.sqrt(len(alpha_arr)-1)))

        Rg = np.mean(rg_array) * 10 ** -10
        Rg_sme = (np.std(rg_array) / (6*np.sqrt(len(rg_array)-1))) * 10 ** -10

        l = np.sqrt(np.mean(y[index_end:]))
        l_sme = (np.std(y[index_end:]) / np.sqrt(len(y) - index_end - 1))

        tau = ((l ** 2) / D)
        tau_sme = np.sqrt(((2 * l / D) ** 2 * l_sme ** 2 + ((-l) ** 2 / D ** 2) ** 2 * D_sme ** 2))

        eta_D = (self.kb * T) / (6 * np.pi * D * Rg)
        eta_D_sme = np.sqrt((-(self.kb * T) / (6 * np.pi * D ** 2 * Rg)) ** 2 * D_sme ** 2 + (
                -(self.kb * T) / (6 * np.pi * D * Rg ** 2)) ** 2 * Rg_sme ** 2)


        D = D * 10 ** 6
        D_sme = D_sme * 10 ** 6

        Rg = Rg * 10 ** 10
        Rg_sme = Rg_sme * 10 ** 10

        l = l * 10 ** 10
        l_sme = l_sme * 10 ** 10

        print("l: " + str(l) + " ± " + str(l_sme) + " A")
        print("tau: " + str(tau) + " ± " + str(tau_sme) + " s")
        print("D: " + str(D) + " ± " + str(D_sme) + " cm^2/s")
        print("eta: " + str(eta_D) + " ± " + str(eta_D_sme) + " Ns/m^2")
        print("Rg: " + str(Rg) + " ± " + str(Rg_sme) + " A")

        fig, ax1 = plt.subplots(figsize=(3.2, 3.2))
        sns.scatterplot(ax=ax1, x=x, y=y, color="k", legend=False, s=60,
                        edgecolor="k", linewidth=2, zorder=1)

        plt.plot(x[index_start:index_end], y[index_start:index_end], color="b")
        ax1.plot(x[:index_end], y_fit_D, color="b", linewidth=4, zorder=2)

        plt.plot(x[index_end:], y[index_end:], color="r")
        ax1.plot(x[index_end:], y_fit_T, color="r", linewidth=4, zorder=2)

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)
        plt.savefig("{}/IMAGES/{}_LOGDIFF.png".format(self.path,sm), format="png", dpi=400)

        return D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme


    def calc_diffusivity(self, msd_array, time_array, dt, iterations, tolerance, boot_r2, boot_tolerance):
        start = 0
        end = dt
        alpha_arr = []
        c_arr = []
        x_log = np.log10(time_array)
        y_log = np.log10(msd_array)
        longest_region = (0, 0)
        current_region = (0, 0)
        max_length = 0
        current_length = 0
        alpha_best = 0
        best_region = longest_region

        while end < len(x_log):
            x_data = x_log[start:end]
            y_data = y_log[start:end]
            linear_model = linregress(x_data, y_data)
            alpha = linear_model.slope

            if abs(alpha - 1) <= tolerance:
                current_length += 1
                if current_length == 1:
                    current_region = (start, end)
                else:
                    current_region = (current_region[0], end)

                if current_length > max_length:
                    max_length = current_length
                    longest_region = current_region
            else:
                current_length = 0
                if longest_region[1]-longest_region[0]>0:
                    x = x_log[longest_region[0]:longest_region[1]]
                    y = y_log[longest_region[0]:longest_region[1]]
                    alpha_longest = linregress(x, y).slope
                    if abs(alpha_longest - 1) <= abs(alpha_best - 1):
                        best_region = longest_region
                        alpha_best = alpha_longest

            start += 1
            end += 1
        best_start = best_region[0]
        best_end = best_region[1]
        x_data = time_array[best_start:best_end]
        y_data = msd_array[best_start:best_end]
        x_log_data = x_log[best_start:best_end]
        y_log_data = y_log[best_start:best_end]

        for iter in range(iterations):
            x, y, log_x, log_y = resample(x_data, y_data, x_log_data, y_log_data, replace=True)
            try:
                linear_model = linregress(log_x, log_y)
                alpha = linear_model.slope
                error = linear_model.rvalue
                if abs(error) > boot_r2 and alpha>=0 and np.abs(alpha-1)<=boot_tolerance:
                    linear_model = linregress(x, y)
                    alpha_arr.append(linear_model.slope)
                    c_arr.append(linear_model.intercept)
            except:
                iter-=1
        return np.mean(alpha_arr), np.mean(c_arr), best_start, best_end

    def calc_tau(self, msd_array, time_array, iterations):
        alpha_arr = []
        c_arr = []
        x_data = time_array
        y_data = msd_array
        x_log_data = np.log10(x_data)
        y_log_data = np.log10(y_data)
        for iter in range(iterations):
            x, y, log_x, log_y = resample(x_data, y_data, x_log_data, y_log_data, replace=False)
            try:
                linear_model = linregress(x, y)
                alpha = linear_model.slope
                alpha_arr.append(alpha)
                c_arr.append(linear_model.intercept)
            except:
                pass

        return np.mean(alpha_arr), np.mean(c_arr)

if __name__ == '__main__':
    def read_files(sm, folder):
        msd_arr = []
        time_arr = []
        rg_arr = []
        for file in tqdm(os.listdir(folder), desc="Collecting Diffusion Files", unit="files"):
            if fnmatch.fnmatch(file, 'G3BP1_Diffusivity_{}*.csv'.format(sm)):
                df_diffusion = pd.read_csv("{}/{}".format(folder,file))
                msd_arr.append(list(df_diffusion["MSD (um)"][1:]))
                time_arr = list(df_diffusion["Time (s)"][1:])
                rg_arr.append(df_diffusion["Rg"].mean())
        msd_array = np.array(msd_arr)
        time_array = np.array(time_arr)
        rg_array = np.array(rg_arr)
        return msd_array, time_array, rg_array

    dt = 10
    seg_length = 200
    T = 300
    iterations = 10
    slope_iterations = 1000
    tolerance = 0.2
    boot_r2 = 0.9
    boot_tolerance = 0.2

    folder = "ANALYSIS_SG_AVE/DIFFUSIVITY"
    sm = "sg_X"
    msd_array, time_array, rg_array = read_files(sm, folder)
    diff = DIFF(msd_array, time_array, rg_array)
    D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme = diff.boot_strap(
        sm=sm, iterations=iterations, T=T, dt=dt, slope_iterations=slope_iterations,
        tolerance=tolerance, boot_r2=boot_r2, boot_tolerance=boot_tolerance, seg = seg_length)
    """
    folder = "ANALYSIS_DSM"
    sm = "dsm_daunorubicin"
    msd_array, time_array, rg_array = read_files(sm, folder)
    diff = DIFF(msd_array, time_array, rg_array)
    D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme = diff.boot_strap(
        sm=sm, iterations=iterations, T=T, dt=dt, slope_iterations=slope_iterations,
        tolerance=tolerance, boot_r2=boot_r2, boot_tolerance=boot_tolerance)

    folder = "ANALYSIS_DSM"
    sm = "dsm_anisomycin"
    msd_array, time_array, rg_array = read_files(sm, folder)
    diff = DIFF(msd_array, time_array, rg_array)
    D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme = diff.boot_strap(
        sm=sm, iterations=iterations, T=T, dt=dt, slope_iterations=slope_iterations,
        tolerance=tolerance, boot_r2=boot_r2, boot_tolerance=boot_tolerance)
    """
    folder = "ANALYSIS_DSM"
    sm = "dsm_daunorubicin"
    path = "CLASS"
    msd_array, time_array, rg_array = read_files(sm, folder)
    diff = DIFF(path, msd_array, time_array, rg_array)
    D, D_sme, eta_D, eta_D_sme, tau, tau_sme, l, l_sme, Rg, Rg_sme = diff.boot_strap(
        sm=sm, iterations=iterations, T=T, dt=dt, slope_iterations=slope_iterations,
        tolerance=tolerance, boot_r2=boot_r2, boot_tolerance=boot_tolerance, seg=seg_length)