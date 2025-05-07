import numpy as np
import scipy.integrate as spi
from scipy import interpolate
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import math
from ACF import ACF
import seaborn as sns
from tqdm import tqdm
import random

class VSC:
    def __init__(self, path, Pxyz):
        self.Pxyz = Pxyz
        self.path = path
        pass

    def run_vsc(self, vol_sys, name, segments, iterations, n_boot, dt_unit, n_point, n_tau):
        acf = ACF()
        dt, acfs_bootstrap = acf.get_boot_data(self.Pxyz, segments, iterations)

        acf_boot = []

        for i in range(len(acfs_bootstrap)):
            acf_boot.append(acfs_bootstrap[i])
        acf_boot = np.array(acf_boot)

        print('\nCalculating the Viscosity from the Green-Kubo Relation')

        eta_raw, eta_fit, eta_theo, acf_fits_total, amp_opts, tau_opts, y_log0s, dt_log, dt, acf, acf_fit = self.get_visco(dt=dt, acfs=acf_boot, vol=vol_sys, dt_unit=dt_unit,
                                                                                                         n_point=n_point, n_tau=n_tau, n_boot=n_boot)

        eta_raw_mean = np.mean(eta_raw[eta_raw>0])
        eta_raw_sem = np.std(eta_raw[eta_raw>0])/np.sqrt(len(eta_raw[eta_raw>0])-1)
        eta_fit_mean = np.mean(eta_fit[eta_fit>0])
        eta_fit_sem = np.std(eta_fit[eta_fit>0]) / np.sqrt(len(eta_fit[eta_fit>0]) - 1)
        eta_theo_mean = np.mean(eta_theo[eta_theo>0])
        eta_theo_sem = np.std(eta_theo[eta_theo>0]) / np.sqrt(len(eta_theo[eta_theo>0]) - 1)

        print("Viscosity GK (Eta_Raw): " + str(eta_raw_mean) + " ± " + str(eta_raw_sem) + " Ns/m^2")
        print("Viscosity GK (Eta_Fit): " + str(eta_fit_mean) + " ± " + str(eta_fit_sem) + " Ns/m^2")
        print("Viscosity GK (Eta_Theo): " + str(eta_theo_mean) + " ± " + str(eta_theo_sem) + " Ns/m^2")

        # Plot the evolution of the viscosity in time
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
        col_pall = sns.color_palette("rocket", n_colors=2)

        fig, ax1 = plt.subplots(figsize=(3.2, 3.2))
        ax1.plot(dt, acf, label='Raw Viscosity', color=col_pall[0], linewidth=4, zorder=1)
        """
        sns.scatterplot(ax=ax1, x=dt, y=acf, color=col_pall[0], legend=False, s=60,
                        edgecolor="k", linewidth=2, zorder=3)
        """
        ax1.plot(dt_log, acf_fit, label='Fitted Viscosity', color=col_pall[1], linewidth=4, zorder=1)
        leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
        leg.get_frame().set_alpha(0)
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)
        plt.savefig("{}/IMAGES/{}_VISC.png".format(self.path,name), format="png", dpi=400)

        return eta_raw_mean, eta_raw_sem, eta_fit_mean, eta_fit_sem, eta_theo_mean, eta_theo_sem, acf_fits_total, amp_opts, tau_opts, y_log0s, dt_log

    def maxwell_model(self, x, *params):
        y = np.zeros_like(x)

        for i in range(0, len(params), 2):
            a = params[i]
            b = params[i + 1]

            y += a * np.exp(- x / b)

        return y

    def get_visco(self, dt, acfs, vol, dt_unit, n_point, n_tau, n_boot):
        k_conv = (dt_unit * vol)
        log_min = -5
        log_max = np.log10(max(dt))
        dt_log = np.logspace(log_min, log_max, n_point)
        eta_raw = np.zeros(n_boot)
        eta_fit = np.zeros(n_boot)
        eta_theo = np.zeros(n_boot)
        y_log0s = np.zeros(n_boot)
        acf_fits_total = []
        amp_opts = []
        tau_opts = []

        for i in tqdm(range(n_boot), desc="Running Viscosity Bootstrap", unit="iterations"):
            n = random.randint(0,len(acfs)-1)
            acf = acfs[n]
            acf_spline = interpolate.InterpolatedUnivariateSpline(dt, acf)
            acf_log = acf_spline(dt_log)
            acf_norm = acf_log / acf_log[0]

            errs = []
            tau_seq = []
            amp_seq = []
            acf_fits = []

            for n_tau_temp in range(1, n_tau+1):
                initial_params = np.concatenate([[max(acf) / (i + 1), 10 ** (-i)] for i in range(n_tau_temp)])
                param_length = len(initial_params)
                #initial_params = np.concatenate([[np.zeros(len(acf))] for i in range(n_tau_temp)])
                #print(initial_params)
                try:
                    params, _ = curve_fit(self.maxwell_model,
                                          dt_log,
                                          acf_norm,
                                          p0=initial_params,
                                          bounds=(0, max(max(acf_log), max(dt_log))),
                                          maxfev=10000)

                    acf_fit = self.maxwell_model(dt_log, *params) * acf_log[0]

                    err = np.mean(np.abs(acf_fit - acf_log))

                    errs.append(err)

                    amp_seq.append(params[::2])

                    tau_seq.append(params[1::2])

                    acf_fits.append(acf_fit)

                except:
                    initial_params = np.zeros(param_length)
                    params, _ = curve_fit(self.maxwell_model,
                                          dt_log,
                                          acf_norm,
                                          p0=initial_params,
                                          bounds=(0, max(max(acf_log), max(dt_log))),
                                          maxfev=20000)

                    acf_fit = self.maxwell_model(dt_log, *params) * acf_log[0]

                    err = np.mean(np.abs(acf_fit - acf_log))

                    errs.append(err)

                    amp_seq.append(params[::2])

                    tau_seq.append(params[1::2])

                    acf_fits.append(acf_fit)

                #    print(f'n_tau = {n_tau_temp} error ...')

                #    continue
            idx = np.argmin(errs)
            amp_opt = amp_seq[idx]
            tau_opt = tau_seq[idx]
            acf_fit = acf_fits[idx]

            acf_fits_total.append(acf_fit)

            amp_opts.append(amp_opt)
            tau_opts.append(tau_opt)

            integral_raw = spi.cumulative_trapezoid(acf, dt) * k_conv  # raw
            integral_fit = spi.cumulative_trapezoid(acf_fit, dt_log) * k_conv  # fit
            integral_theo = np.sum(amp_opt * tau_opt) * acf_log[0] * k_conv  # theoretical

            eta_raw[i] = integral_raw[-1]  # raw viscosity
            eta_fit[i] = integral_fit[-1]  # fitted viscosity
            eta_theo[i] = integral_theo  # theoretical visocisty
            y_log0s[i] = acf_log[0]


        return eta_raw, eta_fit, eta_theo, acf_fits_total, amp_opts, tau_opts, y_log0s, dt_log, dt, acf, acf_fit

if __name__ == '__main__':
    r = 300
    T=300
    sigma = 8
    folder = "ANALYSIS_SG"
    sm = "sg_X_0"
    #folder = "ANALYSIS_DSM_AVE"
    #sm = "dsm_daunorubicin"
    segments = 20
    iterations = 10
    n_boot = 10
    dt_unit = 2000000E-15
    n_point = 1000
    n_tau = 26

    Pxyz, time = [], []
    vol_sys = 4 / 3 * math.pi * (r*10**-10) ** 3
    vol_atom = 4 / 3 * math.pi * (sigma * 10 ** -10) ** 3
    stress_file = "{}/Stress_Tensor_{}.csv".format(folder, sm)
    with open(stress_file, "r") as file:
        print('\nPreparing Pressure Tensor Array')
        for line in file.readlines()[1:]:
            ln = line.split(",")
            time.append(float(ln[0]) * 20)
            Pxy = -float(ln[4])
            Pxz = -float(ln[5])
            Pyz = -float(ln[6])
            Pxyz.append([Pxy, Pxz, Pyz])
    Pxyz = np.array(Pxyz) * 101325
    path = "CLASS"
    vsc = VSC(path, Pxyz)
    eta_raw, eta_raw_sem, eta_fit, eta_fit_sem, eta_theo, eta_theo_sem, acf_fits_total, amp_opts, tau_opts, y_log0s, dt_log = vsc.run_vsc(vol_sys=vol_sys, name=sm, segments=segments,
                                                                                                  iterations=iterations, n_boot=n_boot,
                                                                                                  dt_unit=dt_unit, n_point=n_point,
                                                                                                  n_tau=n_tau)