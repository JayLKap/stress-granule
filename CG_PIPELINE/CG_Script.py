import os
import pandas as pd
import ConvertCG as Convert
import QSPR as QSPR
import shutil
import GenBonds as Bondify


# Coarse Grain Molecules

class CG_Script:

    def __init__(self):
        dir_pdb_og_files = "GBCG/SM_PDB_FILES/"

        bead_path = "Bead_PDB_Files/"
        if not os.path.exists(bead_path):
            os.makedirs(bead_path)

        smi_path = "Bead_SMI_Files/"
        if not os.path.exists(bead_path):
            os.makedirs(smi_path)

        fim = open("masses.txt", "w")
        fim.close()

        fib = open("bonds.txt", "w")

        for pdb in sorted(os.listdir(dir_pdb_og_files)):
            self.gen_beads(pdb, dir_pdb_og_files, fib)

        self.gen_smiles()

        self.add_smiles("MPiPi_Molecules.csv")

        self.run_qspr("molecules.csv", "MPiPi_Parameters.csv")

    # Coarse Grain using GBCG
    def gen_beads(self, pdb, dir_pdb_og_files, fib):
        mol_name = pdb.split(".")[0]
        mol_folder = "GBCG/Molecule_Files/{}".format(mol_name) + "/"
        bead_folder = "Bead_PDB_Files/"

        if not os.path.exists(mol_folder):
            os.makedirs(mol_folder)

        pdb_file = dir_pdb_og_files + pdb

        os_command = "python3 GBCG/GB_mapping_spectral_pdb.py -pdb {} -niter 4 -weights mass -max_size {}".format(
            pdb_file, 205)
        os.system(os_command)
        map_file = "map_files/iter.4.map"
        cg_pdb_file = "pdb_files/mol_0.4.pdb"

        mass_file = "masses.txt"
        self.get_mass(map_file, mol_name, mass_file)

        converter = Convert.ConvertCG(map_file, pdb_file)
        converter.convert_cg(mol_folder)
        converter.convert_cg(bead_folder)
        shutil.copyfile(map_file, mol_folder + "cg_{}.map".format(mol_name))
        shutil.copyfile(cg_pdb_file, mol_folder + "cg_{}.pdb".format(mol_name))
        shutil.copyfile(cg_pdb_file, "CG_PDB_Files/{}.pdb".format(mol_name))
        shutil.copyfile(pdb_file, mol_folder + "{}.pdb".format(mol_name))

        bond_pdb_file = "{}cg_{}.pdb".format(mol_folder, mol_name)
        bonding = Bondify.GenBonds(bond_pdb_file, mol_folder)
        bonding.write_file()

        for i in bonding.write_master_file():
            fib.write(i + "\n")

    # Use Original PDB and Map File to Create Fragment PDB Files
    def gen_smiles(self):
        bead_directory = "Bead_PDB_Files/"
        smi_directory = "Bead_SMI_Files/"

        for file in os.listdir(bead_directory):
            mol_name = file.split(".")[0]
            print(mol_name)
            command = "obabel {}{}.pdb -O {}{}.smi".format(bead_directory, mol_name, smi_directory, mol_name)
            print(command)
            os.system(command)

    def get_mass(self, map_file, mol_name, mass_file):
        map = open(map_file, 'r')
        bead_lines = map.readlines()
        fid = open(mass_file, "a")
        for line in bead_lines:
            bead_name = mol_name + "_b" + line.split()[0]
            bead_mass = line.split()[2]
            fid.write(bead_name + " " + bead_mass + "\n")

    def add_smiles(self, molecule_csv):
        fim = open("masses.txt", "r")

        mol_file = pd.read_csv(molecule_csv)

        for line in fim.readlines():
            mol_name = line.split()[0]
            mass = line.split()[1]

            smile_file = "Bead_SMI_Files/" + mol_name + ".smi"

            fis = open(smile_file, "r")

            smile_code = fis.readline().split()[0]

            smile_code = smile_code.replace("[NH]", "[N]")
            smile_code = smile_code.replace("[NH2]", "[NH]")

            row = [mol_name, smile_code, mass]

            mol_file.loc[len(mol_file)] = row

            fim.close()

        mol_file.drop_duplicates(subset=['SMILES'], keep='first')

        if os.path.exists("molecules.csv"):
            os.remove("molecules.csv")
        file = open("molecules.csv", "w")

        mol_file.to_csv(file, index=False)
        file.close()

    # Run QSPR to output parameters

    def run_qspr(self, molecule_csv, parameters_csv):
        MPiPi_Model = QSPR.Regressor(molecule_csv, parameters_csv)
        MPiPi_Model.model_predict()
        MPiPi_Model.mpipi_dataset.drop_duplicates()

        if os.path.exists("parameters.csv"):
            os.remove("parameters.csv")
        file = open("parameters.csv", "w")
        MPiPi_Model.mpipi_dataset.to_csv(file, index=False)
        file.close()



if __name__ == '__main__':
    script = CG_Script()
