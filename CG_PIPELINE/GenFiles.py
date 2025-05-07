import MixParams
import numpy as np
from MixParams import Mixer
import sys


class Settings:

    def __init__(self, parameters_csv, sys_data, bonds):

        self.particles_num = {}
        self.particles_name = {}
        self.parse_file(parameters_csv)

        self.mixer = MixParams.Mixer(parameters_csv)
        self.mixer.calc_sm_mixing()
        self.parameters = self.mixer.sm_parameters

        self.pairwise_lines = []
        self.coulombic_lines = []
        self.get_pairwise()

        self.charge_lines = []
        self.get_charges()

        self.bonds = {}
        self.bond_num = 0
        self.parse_bonds(bonds)

        self.bond_lines = []
        self.bond_map = {}
        self.bond_lines = self.gen_bonds()

        self.gen_sys_data(sys_data)
        self.gen_sys_settings()

        #self.write_mol_file(sm)



    def parse_file(self, parameters_csv):
        with open(parameters_csv, 'r') as params:
            lines = params.readlines()[1:]
            for line in lines:
                line_split = line.split(",")
                name = line_split[0]
                particle_number = int(line_split[1])
                eps = float(line_split[2])
                sig = float(line_split[3])
                v = int(line_split[4])
                mu = int(line_split[5])
                rc = float(line_split[6])
                q = float(line_split[7])
                m = float(line_split[8])
                self.particles_num[particle_number] = {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc, "q": q, "m": m}
                self.particles_name[name] = {"num": particle_number, "eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc, "q": q, "m": m}

    def get_mass_lines(self):
        mass_lines = []
        for key in sorted(self.particles_num.keys()):
            mass_lines.append("{:<3d}{:<8f}{:<4s}\n".format(key, self.particles_num[key]["m"], " # "+str(key)))
        return mass_lines

    def gen_sys_data(self, sys_data):
        fid = open("sys.data", "w")
        atom_dict = {}
        with open(sys_data, 'r') as params:
            lines = params.readlines()
            misc = lines[0:3]
            fid.writelines(misc)
            fid.write("{} atom types\n".format(len(self.particles_num.keys())))
            misc = lines[4:5]
            fid.writelines(misc)
            fid.writelines("{} bond types\n".format(self.bond_num))
            misc = lines[6:lines.index("Masses\n")+1]
            fid.write("\n-0.9000000000000000e+03   1.5000000000000000e+03    xlo xhi\n")
            fid.write("-0.9000000000000000e+03   1.5000000000000000e+03    ylo yhi\n")
            fid.write("-0.9000000000000000e+03   1.5000000000000000e+03    zlo zhi\n")
            fid.write("\nMasses\n\n")
            masses = self.get_mass_lines()
            fid.writelines(masses)
            atoms = lines[lines.index("Atoms\n")+2: lines.index("Bonds\n")-1]
            bonds = lines[lines.index("Bonds\n")+2:]

        fid.write("\nAtoms\n\n")

        for i in atoms:
            fid.write(i)
            atom_num = int(i.split()[0])
            atom_type = int(i.split()[2])
            atom_dict[atom_num] = atom_type

        fid.write("\nBonds\n\n")
        for i in bonds:
            atm1 = int(i.split()[2])
            atm2 = int(i.split()[3])
            atm1_type = int(atom_dict[atm1])
            b_type = 0
            if atm1_type <= 20:
                b_type = 1
            elif atm1_type in range(21, 25):
                b_type = 2
            fid.write("{:<6d}{:<2d}{:<6d}{:<6d}\n".format(int(i.split()[0]), b_type, atm1, atm2))
        fid.close()

    def gen_sys_settings(self):
        fid = open("sys.settings", "w")
        fid.write("# MPiPi Settings File\n\n# Charges\n")
        fid.writelines(self.charge_lines)
        fid.write("\n# Bonds\n")
        fid.writelines(self.bond_lines)
        fid.write("\n# Pair Style\n")
        fid.write("pair_style  hybrid/overlay wf/cut 25.0 coul/debye 0.126 0.0")
        fid.write("\n\n# Pairwise Interactions\n")
        fid.writelines(self.pairwise_lines)
        fid.write("\n# Coulombic Interactions\n")
        fid.writelines(self.coulombic_lines)

    def get_charges(self):
        for key1 in self.particles_num.keys():
            charge1 = np.round(self.particles_num[key1]["q"], 3)
            if charge1 != 0:
                self.charge_lines.append("set type {:<3d} charge {:>3f}\n".format(key1, charge1))
                for key2 in self.parameters[int(key1)].keys():
                    charge2 = np.round(self.particles_num[key2]["q"], 3)
                    if charge2 != 0:
                        self.coulombic_lines.append("pair_coeff {:<3d}{:<3d}coul/debye 35.0\n".format(int(key1), int(key2)))

    def get_pairwise(self):
        for key1 in self.parameters.keys():
            for key2 in self.parameters[key1].keys():
                eps = self.parameters[key1][key2]["eps"]
                sig = self.parameters[key1][key2]["sig"]
                v = self.parameters[key1][key2]["v"]
                mu = self.parameters[key1][key2]["mu"]
                rc = self.parameters[key1][key2]["rc"]

                self.pairwise_lines.append(
                    "pair_coeff {:<3d}{:<3d}wf/cut {:>8f} {:>8f} {:<2d}{:<3d}{:<8f}\n".format(int(key1), int(key2),
                                                                                                eps, sig, v, mu, rc))

    def parse_bonds(self, bonds):
        for key1 in range(1, 21):
            for key2 in range(1, 21):
                if key2 >= key1:
                    if key1 in self.bonds:
                        self.bonds[key1][key2] = {"l": 3.810, "k": 9.600}
                    else:
                        self.bonds[key1] = {key2: {"l": 3.810, "k": 9.600}}

        for key1 in range(21, 25):
            for key2 in range(21, 25):
                if key2 >= key1:
                    if key1 in self.bonds:
                        self.bonds[key1][key2] = {"l": 5.000, "k": 9.600}
                    else:
                        self.bonds[key1] = {key2: {"l": 5.000, "k": 9.600}}

        self.bond_num = 2

        with open(bonds, 'r') as bond:
            lines = bond.readlines()
            for line in lines:
                line_split = line.split()
                key1 = self.particles_name[line_split[0]]["num"]
                key2 = self.particles_name[line_split[1]]["num"]
                length = np.round(float(line_split[2]), 3)
                self.bond_num += 1
                if key1 in self.bonds:
                    self.bonds[key1][key2] = {"l": length, "k": 9.600}
                else:
                    self.bonds[key1] = {key2: {"l": length, "k": 9.600}}

    def gen_bonds(self):
        bond_array = []
        bond_array.append("bond_coeff {:<3d} {:<5f} {:<5f}\n".format(1, 9.600, 3.810))
        bond_array.append("bond_coeff {:<3d} {:<5f} {:<5f}\n".format(2, 9.600, 5.000))
        count = 3
        for key1 in self.bonds.keys():
            for key2 in self.bonds[key1].keys():
                if key1 >= 25 and key2 >= 25:
                    length = self.bonds[key1][key2]["l"]
                    bond_array.append("bond_coeff {:<3d} {:<5f} {:<5f}\n".format(count, 9.600, length))
                    if key1 in self.bond_map:
                        self.bond_map[key1][key2] = count
                    else:
                        self.bond_map[key1] = {key2: count}
                    count += 1
        return bond_array

    def parse_sm_pdb(self, sm):
        mol_name = sm.split("/")[1].split(".")[0]
        beads = {}
        bonds = {}
        bond_num = 1
        with open(sm, 'r') as mol:
            lines = mol.readlines()
            for line in lines[2:]:
                element = line.split()
                if element[0] == 'HETATM':
                    bead_name = str(mol_name)+"_b"+str(element[1])
                    beads[int(element[1])] = {"type": int(self.particles_name[bead_name]["num"]), "x": float(element[5]), "y": float(element[6]), "z": float(element[7])}
                elif element[0] == 'CONECT':
                    atm1 = element[1]
                    for atm2 in element[2:]:
                        if atm2 > atm1:
                            key1 = beads[int(atm1)]["type"]
                            key2 = beads[int(atm2)]["type"]
                            btype = self.bond_map[key1][key2]
                            bonds[bond_num] = {"btype": btype, "atm1": int(atm1), "atm2": int(atm2)}
                            bond_num += 1
        return mol_name, beads, bonds

    def write_mol_file(self, sm):
        mol_name = self.parse_sm_pdb(sm)[0]
        beads = self.parse_sm_pdb(sm)[1]
        bonds = self.parse_sm_pdb(sm)[2]
        coord_lines = []
        type_lines = []

        for key in beads.keys():
            coord_lines.append("{:<3d} {:>9f} {:>9f} {:<9f}\n".format(key, beads[key]["x"], beads[key]["y"], beads[key]["z"]))
            type_lines.append("{:<3d} {:>3d}\n".format(key, (int(beads[key]["type"])-1)))

        fid = open("CG_MOL_Files/{}.mol".format(mol_name.split(".")[0]), "w")
        fid.write("# LAMMPS Molecule Input File\n\n")
        fid.write("{:<3d} atoms\n".format(len(beads)))
        fid.write("{:<3d} bonds\n".format(len(bonds)))

        fid.write("\nCoords\n\n")
        fid.writelines(coord_lines)

        fid.write("\nTypes\n\n")
        fid.writelines(type_lines)

        if len(bonds.keys()) > 0:
            fid.write("\nBonds\n\n")
            for key in bonds.keys():
                fid.write("{:<3d} {:<3d} {:<3d} {:<3d}\n".format(key, bonds[key]["btype"], bonds[key]["atm1"], bonds[key]["atm2"]))






if __name__ == '__main__':
    parameter_csv = sys.argv[1]
    sys_data_file = sys.argv[2]
    bond_file = sys.argv[3]
    sm_name = sys.argv[4]

    genset = Settings(parameter_csv, sys_data_file, bond_file, sm_name)



