import sys
import numpy as np

class EditData:

    def __init__(self, parameters_csv, wf_parameters, database_path, bonds):
        self.particles = {}
        self.parse_file(parameters_csv)

        self.database = database_path

        self.bonds = {}
        self.parse_bonds(bonds)

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
                self.particles[name] = {"num": particle_number, "eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc, "q": q, "m": m}

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


        with open(bonds, 'r') as bond:
            lines = bond.readlines()
            for line in lines:
                line_split = line.split()
                key1 = self.particles[line_split[0]]["num"]
                key2 = self.particles[line_split[1]]["num"]
                length = np.round(float(line_split[2]), 3)
                if key1 in self.bonds:
                    self.bonds[key1][key2] = {"l": length, "k": 9.600}
                else:
                    self.bonds[key1] = {key2: {"l": length, "k": 9.600}}

    def gen_bonds(self):
        bond_array = []
        bond_array.append("bond_coeff {:<3d} {:<5f} {:<5f}\n".format(1, 9.600, 3.810))
        bond_array.append("bond_coeff {:<3d} {:<5f} {:<5f}\n".format(2, 9.600, 5.000))
        count = 3

        for key1 in self.bonds:
            for key2 in self.bonds[key1].keys():
                if key1 >= 25:
                    bond_array.append("bond_coeff {:<3d} {:<5f} {:<5f}\n".format(count, self.bonds[key1][key2]["k"], self.bonds[key1][key2]["l"]))
                    count+=1

        return bond_array




    def write_database(self):
        fid = open("GENDATA/mpipi_database_new.py", "w")
        fid.write("# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        fid.write("# ~~~~~~~~~~~~~~~~~~~~~~ATOM TYPES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        fid.write("# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        fid.write("# Atom type sub dictionaries\n")
        fid.write("# Dictionary keys:\n")
        fid.write("# 	 * 'm' - the mass (in Daltons)\n")
        fid.write("# 	 * 'q' - the default charge (in electrons)\n")
        fid.write("# 	 * 'eps' - the WF self-interaction parameter\n")
        fid.write("# 	 * 'sig' - the WF self-interaction parameter\n")
        fid.write("# 	 * 'v' - the WF self-interaction parameter\n")
        fid.write("# 	 * 'mu' - the WF self-interaction parameter\n")
        fid.write("# 	 * 'rc' - the WF self-interaction parameter\n")
        fid.write("# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%)\n")

        for key in self.particles:
            line = str(
                "atm_types['"+str(self.particles[key]["num"])+"'] = {'pot': 'WF', 'm': "+str(self.particles[key]["m"])
                +", 'q': "+str(self.particles[key]["q"])+", 'eps': "+str(self.particles[key]["eps"])+", 'sig': "
                +str(self.particles[key]["sig"])+", 'v': "+str(self.particles[key]["v"])+", 'mu': "
                +str(self.particles[key]["mu"])+", 'rc': "+str(self.particles[key]["rc"])+"}\n")
            fid.write(line)

        fid.write("\n# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        fid.write("# ~~~~~~~~~~~~~~~~~~~~~~BOND TYPES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        fid.write("# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        fid.write("# Bond type sub dictionaries\n")
        fid.write("# Dictionary keys depend on the interaction type, set by key 'type'\n")
        fid.write("# Coded interaction types include: 'harmonic', 'quartic'\n")
        fid.write("# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

        for key1 in self.bonds:
            for key2 in self.bonds[key1].keys():
                fid.write("bon_types[('"+str(key1)+"', '"+str(key2)+"')] = {'type': 'harmonic', 'kb': "+str(self.bonds[key1][key2]["k"])+", 'b0': "+str(self.bonds[key1][key2]["l"])+"}\n")


if __name__ == '__main__':

    parameters_csv = sys.argv[1]
    wf_parameters = sys.argv[2]
    database_path = sys.argv[3]
    bonds = sys.argv[4]

    database = EditData(parameters_csv, wf_parameters, database_path, bonds)

    count = 0
    for key1 in database.bonds.keys():
        for key2 in database.bonds[key1].keys():
            print(str(key1)+"\t"+str(key2)+"\t")
            print(database.bonds[key1][key2])
            count += 1

    database.write_database()
