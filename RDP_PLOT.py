import numpy as np
import scipy.special
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math
from scipy.stats import linregress
import seaborn as sns
from scipy import integrate
import pylab
import os
from IPython.display import display
import fnmatch
from ACF import ACF
from VSC import VSC

class RDP:
    def __init__(self, density_file, pca_file, cluster_file, T, init, label):
        kb = 1.3806 * 10 ** (-23)

        density_profile = pd.read_csv(density_file)
        self.distances = density_profile.iloc[0:, 0].tolist()
        densities = np.array(density_profile.iloc[0:, 1].tolist())
        #max_dens = np.max(densities)
        self.densities = list(densities)
        self.sig = density_profile.iloc[0:, 2].tolist()
        self.errors = list(np.array(density_profile.iloc[0:, 2].tolist()))

        self.fit_x = []
        self.st1_std = 0
        self.st2_std = 0

        pca = pd.read_csv(pca_file)
        self.l1 = pca.iloc[:, 0].tolist()
        self.l2 = pca.iloc[:, 1].tolist()
        self.l3 = pca.iloc[:, 2].tolist()

        cluster = pd.read_csv(cluster_file)
        self.cluster_mass = cluster["Mass of Largest Droplet (mg)"].mean()
        self.outer_mass = cluster["Mass of External Chains"].mean()
        self.radius = cluster["Largest Droplet Radius of Gyration"].mean()
        self.radius_se = float(cluster["RG SEM"].iloc[0])

        self.label = label

        self.fit_rho = []

        self.fit_A = 0.0
        self.se_A = 0

        self.fit_B = 0.0
        self.se_B = 0

        self.fit_R = 0.0
        self.se_R = 0

        self.fit_W = 0.0
        self.se_W = 0

        self.c_dilute_fit = 0.0
        self.c_dilute_calc = 0.0

        self.dG = 0.0
        self.se_dG = 0.0

        self.RG = 0
        self.se_RG = 0
        self.calc_rad()

        self.fit_density(T, init)

        self.fit_x = np.linspace(0, self.distances[-1], 400)
        self.fit_rho = self.ERF(self.fit_x, self.fit_A, self.fit_B, self.fit_R, self.fit_W)

        self.c_dense_fit = abs(self.fit_B + self.fit_A)
        self.c_dense_fit_se = np.sqrt(self.se_A**2 + self.se_B**2)

        self.c_dilute_fit = np.abs(self.fit_B - self.fit_A)
        self.c_dilute_fit_se = np.sqrt(self.se_A**2 + self.se_B**2)

        self.c_dense_calc = 0
        self.c_dense_calc_se = 0
        self.calc_dense()

        self.c_dilute_calc = 0
        self.c_dilute_calc_se = 0
        self.calc_dilute()

        if self.c_dense_fit > 0:
            self.dG = kb * T * np.log(self.c_dilute_fit / self.c_dense_fit) * 6.022 * 10 ** 23
            self.se_dG = np.sqrt((kb * T / self.c_dense_fit * 6.022 * 10 ** 23) ** 2 * self.c_dilute_fit_se ** 2 + (
                    -kb * T / self.c_dilute_fit * 6.022 * 10 ** 23) ** 2 * self.c_dense_fit_se ** 2)

        self.st_1 = 0
        self.st_1_se = 0
        self.st_2 = 0
        self.st_2_se = 0
        self.surface_tension(T)

        print("A: " + str(self.fit_A) + "\tStandard Deviation: " + str(self.se_A))
        print("B: " + str(self.fit_B) + "\tStandard Deviation: " + str(self.se_B))
        print("R: " + str(self.fit_R) + "\tStandard Deviation: " + str(self.se_R))
        print("W: " + str(self.fit_W) + "\tStandard Deviation: " + str(self.se_W))

        print("c_dense_fit: " + str(self.c_dense_fit) + "\tStandard Deviation: " + str(self.c_dense_fit_se))
        print("c_dense_calc: " + str(self.c_dense_calc) + "\tStandard Deviation: " + str(self.c_dense_calc_se))
        print("c_dilute_fit: " + str(self.c_dilute_fit) + "\tStandard Deviation: " + str(self.c_dilute_fit_se))
        print("c_dilute_calc: " + str(self.c_dilute_calc) + "\tStandard Deviation: " + str(self.c_dilute_calc_se))

        print("dG: " + str(self.dG) + "\tStandard Deviation: " + str(self.se_dG))

        print("st_1: " + str(self.st_1) + "\tStandard Deviation: " + str(self.st_1_se))
        print("st_2: " + str(self.st_2) + "\tStandard Deviation: " + str(self.st_2_se))
        print("st: " + str(self.st) + "\tStandard Deviation: " + str(self.st_se))

        print("Radius Gyration: " + str(self.RG) + "\tStandard Deviation: " + str(self.se_RG))

    def ERF(self, r, A, B, R, W):
        y = B - A * scipy.special.erf((r - R) / (np.sqrt(2) * W))
        return y

    def calc_dense(self):
        r = (self.fit_R + self.fit_W) * 10 ** (-8)
        r_std = np.sqrt((10 ** -8) ** 2 * self.se_R ** 2 + (0.5 * 10 ** -8) ** 2 * self.se_W ** 2)
        vol = (4 / 3) * np.pi * (r ** 3)
        vol_std = np.sqrt((4 * np.pi * r ** 2) ** 2 * r_std ** 2)
        self.c_dense_calc = (self.cluster_mass) / vol
        self.c_dense_calc_se = np.sqrt(((-self.cluster_mass) / (vol ** 2)) ** 2 * vol_std ** 2)

    def calc_dilute(self):
        c_cyt = 120
        if self.label == "Protein":
            c_cyt = 108
        elif self.label == "RNA":
            c_cyt = 12
        r = (self.fit_R + self.fit_W) * 10 ** (-8)
        r_std = np.sqrt((10 ** -8) ** 2 * self.se_R ** 2 + (0.5 * 10 ** -8) ** 2 * self.se_W ** 2)

        vol_clus = (4 / 3) * np.pi * (r ** 3)
        vol_sys = np.mean(self.outer_mass + self.cluster_mass) / c_cyt
        vol = np.abs(vol_sys - vol_clus)
        vol_std = np.sqrt(((-4 * np.pi * r ** 2) ** 2 * r_std ** 2))

        self.c_dilute_calc = np.mean(self.outer_mass) / vol
        self.c_dilute_calc_se = np.sqrt((-(self.outer_mass) / (vol ** 2)) ** 2 * vol_std ** 2)

    def calc_rad(self):
        self.RG = self.radius
        self.se_RG = self.radius_se

    def fit_density(self, T, init):
        x_data = [x * 1 for x in self.distances]
        y_data = [x * 1 for x in self.densities]
        sig = [x * 1 for x in self.sig]
        bnds = [[0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]]

        try:
            atmpt = True
            for i in sig:
                if i == 0:
                    atmpt = False
            if atmpt:
                parameters, covariance = curve_fit(f=self.ERF, xdata=x_data, ydata=y_data, p0=init, sigma=sig,
                                                   bounds=bnds, maxfev=40000)

            else:
                parameters, covariance = curve_fit(f=self.ERF, xdata=x_data, ydata=y_data, p0=init, bounds=bnds, maxfev=40000)

            parameters, covariance = curve_fit(f=self.ERF, xdata=x_data, ydata=y_data, p0=init)

            self.fit_A = parameters[0]
            self.fit_B = parameters[1]
            self.fit_R = parameters[2]
            self.fit_W = parameters[3]

            std = np.sqrt((np.diag(covariance)))
            self.se_A = std[0]
            self.se_B = std[1]
            self.se_R = std[2]
            self.se_W = std[3]

        except:
            print("Failed")

    def surface_tension(self, T):
        kb = 1.3806 * 10 ** (-23)

        R = self.fit_R * 10 ** (-10)
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3

        a = []
        b = []
        c = []
        a_arr = []
        b_arr = []
        c_arr = []

        for i in range(len(l1)):
            L1 = l1[i] * 10 ** (-10)
            L2 = l2[i] * 10 ** (-10)
            L3 = l3[i] * 10 ** (-10)
            a.append((R * L1 ** (1 / 3)) / ((L2 * L3) ** (1 / 6)))
            b.append((R * L2 ** (1 / 3)) / ((L1 * L3) ** (1 / 6)))
            c.append((R * L3 ** (1 / 3)) / ((L1 * L2) ** (1 / 6)))
            a_arr.append((L1 ** (1 / 3)) / ((L2 * L3) ** (1 / 6)) - 1)
            b_arr.append((L2 ** (1 / 3)) / ((L1 * L3) ** (1 / 6)) - 1)
            c_arr.append((L3 ** (1 / 3)) / ((L2 * L3) ** (1 / 6)) - 1)

        a_arr = np.array(a_arr)
        b_arr = np.array(b_arr)
        c_arr = np.array(c_arr)

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        da = a - R
        db = b - R
        dc = c - R

        ensemble1 = np.mean(np.square(da + db)) + np.mean(np.square(da + dc)) + np.mean(np.square(db + dc))
        ensemble1_std = np.std(np.square(da + db)) + np.mean(np.square(da + dc)) + np.mean(np.square(db + dc))/np.sqrt(len(da)-1)

        ensemble2 = np.mean(np.square(da - db)) + np.mean(np.square(da - dc)) + np.mean(np.square(db - dc))
        ensemble2_std = np.std(np.square(da - db)) + np.mean(np.square(da - dc)) + np.mean(np.square(db - dc))/np.sqrt(len(da)-1)

        self.st_1 = (15 * kb * T) / (16 * np.pi * ensemble1)
        self.st_2 = (45 * kb * T) / (16 * np.pi * ensemble2)
        self.st = (self.st_1+self.st_2)/2

        self.st_1_se = ((-15 * kb * T) / (16 * np.pi * ensemble1 ** 2)) ** 2 * ensemble1_std ** 2
        self.st_2_se = ((-45 * kb * T) / (16 * np.pi * ensemble2 ** 2)) ** 2 * ensemble2_std ** 2
        self.st_se = ((self.st_1_se**2+self.st_2_se**2)/2)**0.5