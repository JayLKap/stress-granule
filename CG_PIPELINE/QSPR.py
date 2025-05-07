# Calculate Molecular Descriptors using MORDRED
# https://www.rdkit.org/
# https://github.com/rdkit/rdkit
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import itertools
import seaborn as sns

os.system("pip install rdkit")
os.system("pip install mordred")

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# https://pandas.pydata.org
import pandas as pd

# https://numpy.org/doc/stable/release.html
import numpy as np

# https://github.com/mordred-descriptor/mordred
from mordred import Calculator, descriptors


class Regressor:

    def __init__(self, molecules_csv, parameters_csv):
        self.molecule_dataset = pd.read_csv(molecules_csv)
        self.mpipi_dataset = pd.read_csv(parameters_csv)
        self.models = {}
        self.descriptors = pd.DataFrame
        self.parameters = pd.DataFrame

    # Create a calculator object with the desired descriptors
    def mordred_descriptors(self, data):
        calc = Calculator(descriptors, ignore_3D=False)
        mols = [Chem.MolFromSmiles(smi) for smi in data]
        print(mols)
        df = calc.pandas(mols)
        return df

    def parse_molecule_files(self):
        mol_name = self.molecule_dataset['Molecules']
        mol_mass = self.molecule_dataset['Mass']
        descript = self.mordred_descriptors(self.molecule_dataset['SMILES']).select_dtypes(['number'])

        # Remove descriptors with low variance
        description = descript.loc[:, descript.var() > 0.01]
        self.descriptors = description
        return mol_name, mol_mass

    # Train the Model
    def train_model(self):
        descriptors = self.descriptors
        parameters = self.mpipi_dataset.loc[:, ['E', 'S', 'U', 'R']]

        # split data into input and output variables
        X_all = descriptors.iloc[:24]
        X_all_copy = X_all
        mol_names = X_all.iloc[:, 0]
        col_dict = {}

        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
        plt.rc('legend', fontsize=20)  # legend fontsize
        plt.rc('font', size=16)  # controls default text sizes
        # Figure Gen

        df = pd.DataFrame(columns=["Parameter", "x", "R^2", "MAE", "Mordred Descriptor"])

        lower = 0.8
        upper = 0.05

        for column in parameters.columns:
            X_all = descriptors.iloc[:24]
            print(column)
            X = pd.DataFrame({"Molecule": list(mol_names)})
            X = X.iloc[:, 1:]
            y = parameters[column]
            mae = 0.0
            mae_best = 0.0
            mae_arr = []
            mae_diff_arr = []
            mae_dict = {}
            col_list = []
            score = 0.0
            score_best = 0.0
            r2_score_arr = []
            r2_diff_arr = []
            r2_dict = {}
            model = RandomForestRegressor()
            test = True
            i = 0
            r2_diff = 1
            x_arr = []
            par_arr = []

            col_id = X_all.var().idxmax()
            max_col = X_all.pop(col_id)
            col_list.append(col_id)
            X[col_id] = max_col

            if column == "E":
                param = "$\epsilon$"
            elif column == "S":
                param = "$\sigma$"
            elif column == "U":
                param = "$\mu$"
            else:
                param = "$r_{c}$"

            while len(col_list) <= len(X_all.columns) and test:
                if ((score_best >= 1) and (r2_diff <= 0.0)):
                    test = False

                else:
                    col_id = X_all.var().idxmax()

                    max_col = X_all.pop(col_id)

                    if score >= score_best and score > 0 and np.abs(score_best-score) <= 0.25:
                        print(col_id)
                        i += 1
                        x_arr.append(i)
                        col_list.append(col_id)
                        X[col_id] = max_col
                        par_arr.append(column)

                        mae_diff = np.abs(mae_best - mae)
                        r2_diff = np.abs(score_best - score)

                        r2_score_arr.append(score_best)
                        r2_diff_arr.append(r2_diff)

                        mae_arr.append(mae_best)
                        mae_diff_arr.append(mae_diff)

                        score_best = score
                        mae_best = mae

                        print(i)
                        print(score_best)
                        print(r2_diff)
                        print(mae_best)

                        df = df.append({"Parameter": param,
                                        "x": i,
                                        "R^2": score_best,
                                        "MAE": mae_best,
                                        "Mordred Descriptor": col_id
                                        }, ignore_index=True)

                    # split data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    X_train.head(40)

                    # fit the model to the training data
                    model.fit(X_train, y_train)

                    # make predictions on the test data
                    y_pred = model.predict(X_test)

                    # evaluate the model using mean absolute error
                    mae = mean_absolute_error(y_test, y_pred)

                    score = r2_score(y_test, y_pred)

            self.models[column] = model


            plt.plot(r2_score_arr)
            plt.show()

            r2_dict[column] = score_best
            mae_dict[column] = mae_best
            col_dict[column] = col_list
            X_all = X_all_copy

            print(len(X_all))

            print("Parameter "+column+": "+str(i) + " Mordred Descriptors with r2 = " + str(score_best) + " and MAE = "+ str(mae_best))

        print(df)
        fig, axs = plt.subplots(figsize=(6, 6), tight_layout=True)
        sns.lineplot(ax=axs, data = df, x="x", y="R^2", linewidth=4, hue="Parameter", palette="rocket", marker='o')
        plt.xlabel('MORDRED Parameter Number')
        plt.ylabel('$R^{2}$')
        plt.show()
        df.to_csv("R2_Dataset.csv", sep=",")
        plt.savefig("R2_Plot.png", format="png", dpi=400)
        return col_dict

    # Step 6
    def model_predict(self):
        mol_names, mol_mass = self.parse_molecule_files()
        columns = self.train_model()
        value = 0.0

        index = 0

        for row in range(24, len(self.molecule_dataset)):
            name = mol_names[row]
            number = row + 1
            m = mol_mass[row]
            sm_row = [name, number, 0.0, 0.0, 1, 0, 0.0, 0.0, m]
            self.mpipi_dataset.loc[len(self.mpipi_dataset.index)] = sm_row
            index += 1

        for col in columns.keys():
            sm_descriptors = self.descriptors.copy().iloc[24:, :]
            sm_descriptors = sm_descriptors.loc[:, columns[col]]
            predictions = self.models[col].predict(sm_descriptors)

            for row in range(24, len(self.molecule_dataset)):
                value = predictions[row - 24]
                if col == "V" or col == "U":
                    value = int(np.round(value))
                self.mpipi_dataset.loc[row, col] = value


if __name__ == '__main__':
    molecule_csv = "molecules.csv"
    parameters_csv = "MPiPi_Parameters.csv"
    MPiPi_Model = Regressor(molecule_csv, parameters_csv)
    MPiPi_Model.model_predict()
    pd.set_option('display.max_columns', None)
    print(MPiPi_Model.molecule_dataset)
    print(MPiPi_Model.mpipi_dataset)
    MPiPi_Model.mpipi_dataset.to_csv("single_parameters.csv", index=False)
