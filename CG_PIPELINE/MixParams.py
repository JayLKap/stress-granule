# Calculate Molecular Descriptors using MORDRED
# https://www.rdkit.org/
# https://github.com/rdkit/rdkit
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# https://pandas.pydata.org
import pandas as pd

# https://numpy.org/doc/stable/release.html
import numpy as np
import seaborn as sns


class Mixer:

    def __init__(self, csv_file):
        self.parameters_file = "WF_Parameters.txt"
        self.csv_file = csv_file
        self.homotypic_actual_parameters = {}
        self.heterotypic_actual_parameters = {}
        self.pred_parameters = {}
        self.actual_parameters = {}
        self.sm_homotypic_parameters = {}
        self.sm_parameters = {}
        self.homotypic = {}
        self.parse_file()
        self.parse_csv()
        self.all_params = {}

    def parse_file(self):
        with open(self.parameters_file, 'r') as params:
            lines = params.readlines()
            for line in lines:
                line_split = line.split()
                atom1 = int(line_split[1])
                atom2 = int(line_split[2])
                eps = float(line_split[4])
                sig = float(line_split[5])
                v = int(line_split[6])
                mu = int(line_split[7])
                rc = float(line_split[8])

                if atom1 in self.actual_parameters.keys():
                    self.actual_parameters[atom1][atom2] = {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}
                else:
                    self.actual_parameters[atom1] = {atom2: {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}}

                if atom1 == atom2:
                    self.homotypic_actual_parameters[atom1] = {
                        atom2: {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}}
                else:
                    if atom1 in self.heterotypic_actual_parameters.keys():
                        self.heterotypic_actual_parameters[atom1][atom2] = {"eps": eps, "sig": sig, "v": v, "mu": mu,
                                                                            "rc": rc}
                    else:
                        self.heterotypic_actual_parameters[atom1] = {
                            atom2: {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}}

    def parse_csv(self):
        with open(self.csv_file, 'r') as params:
            lines = params.readlines()[25:]
            for line in lines:
                data = line.split(",")
                atom = int(data[1])
                eps = float(data[2])
                sig = float(data[3])
                v = int(data[4])
                mu = int(data[5])
                rc = float(data[6])
                self.sm_homotypic_parameters[atom] = {atom: {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}}

        for key in self.homotypic_actual_parameters:
            self.homotypic[key] = (self.homotypic_actual_parameters[key][key])

        for key in self.sm_homotypic_parameters:
            self.homotypic[key] = (self.sm_homotypic_parameters[key][key])


    def geometric_mix(self, par1, par2):
        mix = np.sqrt(par1 * par2)
        return mix

    def arithmetic_mix(self, par1, par2):
        mix = (par1 + par2) / 2
        return mix

    def wh_mix_s(self, s1, s2):
        mix = np.power((np.power(s1, 6) + np.power(s2, 6)) / 2, 1 / 6)
        return mix

    def wh_mix_e(self, e1, e2, s1, s2):
        mix = 2 * np.sqrt(e1 * e2) * ((np.power(s1, 3) * np.power(s2, 3)) / (np.power(s1, 6) + np.power(s2, 6)))
        return mix

    def fh_mix_e(self, e1, e2):
        mix = (2 * e1 * e2) / (e1 + e2)
        return mix

    def fh_mix_s(self, s1, s2):
        mix = self.arithmetic_mix(s1, s2)
        return mix

    def k_mix_s(self, e1, e2, s1, s2):
        mix = np.power(
            np.power((np.power((e1 * np.power(s1, 12)), 1 / 13)
                      + np.power((e2 * np.power(s2, 12)), 1 / 13)) / 2, 13)
            , 1 / 6)
        return mix

    def k_mix_e(self, e1, e2, s1, s2):
        mix = (e1*np.power(s1, 6)*e2*np.power(s2, 6))/np.power(self.k_mix_s(e1, e2, s1, s2), 6)
        return mix

    def calc_rmse(self, par_test, par_pred):
        rmse = np.sqrt(mean_squared_error(par_test, par_pred))
        return rmse

    def get_test_params(self, param):
        params = []
        for key1 in self.actual_parameters.keys():
            for key2 in self.actual_parameters[key1].keys():
                params.append(self.actual_parameters[key1][key2][param])
        return params

    def calc_pred_params(self, param):
        mix_params = {"eps": {"LB": [], "WH": [], "FH": [], "K": []}, "sig": {"LB": [], "WH": [], "FH": [], "K": []}, "v": {"G": [], "A": []}, "mu": {"G": [], "A": []}, "rc": {"G": [], "A": []}}

        for key1 in self.actual_parameters.keys():
            for key2 in self.actual_parameters[key1].keys():
                param1 = self.homotypic_actual_parameters[key1][key1][param]
                param2 = self.homotypic_actual_parameters[key2][key2][param]

                if param == "eps":
                    param3 = self.homotypic_actual_parameters[key1][key1]["sig"]
                    param4 = self.homotypic_actual_parameters[key2][key2]["sig"]

                    mix_params["eps"]["LB"].append(self.geometric_mix(param1, param2))
                    mix_params["eps"]["WH"].append(self.wh_mix_e(param1, param2, param3, param4))
                    mix_params["eps"]["FH"].append(self.fh_mix_e(param1, param2))
                    mix_params["eps"]["K"].append(self.k_mix_e(param1, param2, param3, param4))

                elif param == "sig":
                    param3 = self.homotypic_actual_parameters[key1][key1]["eps"]
                    param4 = self.homotypic_actual_parameters[key2][key2]["eps"]

                    mix_params["sig"]["LB"].append(self.arithmetic_mix(param1, param2))
                    mix_params["sig"]["WH"].append(self.wh_mix_s(param1, param2))
                    mix_params["sig"]["FH"].append(self.arithmetic_mix(param1, param2))
                    mix_params["sig"]["K"].append(self.k_mix_e(param3, param4, param1, param2))

                elif param == "v":
                    mix_params["v"]["G"].append(self.geometric_mix(param1, param2))
                    mix_params["v"]["A"].append(self.arithmetic_mix(param1, param2))

                elif param == "mu":
                    mix_params["mu"]["G"].append(self.geometric_mix(param1, param2))
                    mix_params["mu"]["A"].append(self.arithmetic_mix(param1, param2))

                elif param == "rc":
                    mix_params["rc"]["G"].append(self.geometric_mix(param1, param2))
                    mix_params["rc"]["A"].append(self.arithmetic_mix(param1, param2))

        return mix_params

    def compare_mixing_models(self):
        mixing_parameters = ["eps", "sig", "v", "mu", "rc"]
        optimal_model = {}
        predict = {"eps": {}, "sig": {}, "mu": {}, "v": {}, "rc": {}}
        prediction = {}

        for parameter in mixing_parameters:
            test = self.get_test_params(parameter)
            prediction = self.calc_pred_params(parameter)
            predict[parameter] = self.calc_pred_params(parameter)
            scores = {}
            for mod in prediction[parameter].keys():
                pred = prediction[parameter][mod]
                score = self.calc_rmse(test, pred)
                scores[mod] = score
            optimal_model[parameter] = min(scores, key=scores.get)
            print(scores)

        print(optimal_model)
        return optimal_model


    def calc_sm_mixing(self):
        optimal_model = self.compare_mixing_models()
        mixing_parameters = ["eps", "sig", "v", "mu", "rc"]
        parameter_dict = {}
        for key1 in self.homotypic.keys():
            for key2 in self.homotypic.keys():
                for param in mixing_parameters:

                    value1 = self.homotypic[key1][param]
                    value2 = self.homotypic[key2][param]

                    if param == "eps":
                        value3 = self.homotypic[key1]["sig"]
                        value4 = self.homotypic[key2]["sig"]
                        if optimal_model[param] == "LB":
                            parameter_dict[param] = self.geometric_mix(value1, value2)
                        elif optimal_model[param] == "WH":
                            parameter_dict[param] = self.wh_mix_e(value1, value2, value3, value4)
                        elif optimal_model[param] == "FH":
                            parameter_dict[param] = self.fh_mix_e(value1, value2)
                        elif optimal_model[param] == "K":
                            parameter_dict[param] = self.k_mix_e(value1, value2, value3, value4)

                    elif param == "sig":
                        value3 = self.homotypic[key1]["eps"]
                        value4 = self.homotypic[key2]["eps"]
                        if optimal_model[param] == "LB":
                            parameter_dict[param] = self.arithmetic_mix(value1, value2)
                        elif optimal_model[param] == "WH":
                            parameter_dict[param] = self.wh_mix_s(value1, value2)
                        elif optimal_model[param] == "FH":
                            parameter_dict[param] = self.arithmetic_mix(value1, value2)
                        elif optimal_model[param] == "K":
                            parameter_dict[param] = self.k_mix_s(value3, value4, value1, value2)

                    elif param == "rc":
                        if optimal_model[param] == "G":
                            parameter_dict[param] = (self.geometric_mix(value1, value2))
                        elif optimal_model[param] == "A":
                            parameter_dict[param] = (self.arithmetic_mix(value1, value2))

                    else:
                        if optimal_model[param] == "G":
                            parameter_dict[param] = int(np.round(self.geometric_mix(value1, value2)))
                        elif optimal_model[param] == "A":
                            parameter_dict[param] = int(np.round(self.arithmetic_mix(value1, value2)))

                if int(key2) >= int(key1):
                    if key1 in range(1, 25) and key2 in range(1, 25):
                        if key1 in self.sm_parameters.keys():
                            self.sm_parameters[key1][key2] = self.actual_parameters[key1][key2]
                        else:
                            self.sm_parameters[key1] = {key2: self.actual_parameters[key1][key2]}


                    else:
                        if key1 in self.sm_parameters.keys():
                            self.sm_parameters[key1][key2] = {"eps": parameter_dict["eps"], "sig": parameter_dict["sig"], "v": parameter_dict["v"], "mu": parameter_dict["mu"], "rc": parameter_dict["rc"]}
                        else:
                            self.sm_parameters[key1] = {key2: {"eps": parameter_dict["eps"], "sig": parameter_dict["sig"], "v": parameter_dict["v"], "mu": parameter_dict["mu"], "rc": parameter_dict["rc"]}}




if __name__ == '__main__':
    parameter_file = "parameters.csv"
    mixer = Mixer(parameter_file)

    count = 0

    for key1 in mixer.homotypic_actual_parameters.keys():
        for key2 in mixer.homotypic_actual_parameters[key1].keys():
            #print(str(key1) + "\t" + str(key2) + "\t" + str(mixer.homotypic_actual_parameters[key1][key2]))
            c = 0

    for key1 in mixer.heterotypic_actual_parameters.keys():
        for key2 in mixer.heterotypic_actual_parameters[key1].keys():
            #print(str(key1) + "\t" + str(key2) + "\t" + str(mixer.heterotypic_actual_parameters[key1][key2]))
            c = 0

    for key1 in mixer.actual_parameters.keys():
        for key2 in mixer.actual_parameters[key1].keys():
            #print(str(key1) + "\t" + str(key2) + "\t" + str(mixer.actual_parameters[key1][key2]))
            c = 0
    for key1 in mixer.sm_homotypic_parameters.keys():
        for key2 in mixer.sm_homotypic_parameters[key1].keys():
            #print(str(key1) + "\t" + str(key2) + "\t" + str(mixer.sm_homotypic_parameters[key1][key2]))
            c = 0

    #print(mixer.get_test_params("eps"))
    #print(len(mixer.get_test_params("eps")))

    # print(mixer.calc_pred_params("eps")["LB"])
    for key1 in mixer.sm_homotypic_parameters.keys():
        for key2 in mixer.sm_homotypic_parameters[key1].keys():
            if key1 <= 24 and key2 <= 24:
                #print(len(mixer.calc_pred_params("eps", key1, key2)["LB"]))
                #print(len(mixer.calc_pred_params("eps", key1, key2)["WH"]))
                #print(len(mixer.calc_pred_params("eps", key1, key2)["FH"]))
                #print(len(mixer.calc_pred_params("eps", key1, key2)["K"]))
                c = 0

    #print(mixer.compare_mixing_models())



    for key1 in mixer.homotypic.keys():
        #print(str(key1) + "\t" + str(mixer.homotypic[key1]))
        c = 0

    mixer.calc_sm_mixing()

    count = 0
    for key1 in mixer.sm_parameters.keys():
        for key2 in mixer.sm_parameters[key1].keys():
            count += 1
            #print(str(key1) + "\t" + str(key2) + "\t" + str(mixer.sm_parameters[key1][key2]))





