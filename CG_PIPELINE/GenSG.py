from scipy.optimize import linprog
import sys
import numpy as np
import os


class GenSG:

    def __init__(self, nParticles, partition, mol_list):
        self.nParticles = nParticles
        self.partition = partition
        self.molecules = {}
        self.read_mol_list(mol_list)
        self.rho = 1 * 10 ** (-24) * 6.022 * 10 ** 23  # Da/A

    def read_mol_list(self, mol_l):
        lines = open(mol_l, "r").readlines()
        for line in lines:
            name = line.split("\n")[0]
            self.molecules[name] = self.calc_mass("GENDATA/" + name + ".pdb")

    def calc_mass(self, mol_pdb):
        particles = 0
        mass = 0
        masses = {
            1: 131.18220,
            2: 57.04220,
            3: 128.18260,
            4: 101.08630,
            5: 156.17880,
            6: 71.07034,
            7: 115.08440,
            8: 129.08250,
            9: 163.17780,
            10: 99.05650,
            11: 113.18460,
            12: 128.08260,
            13: 186.17470,
            14: 147.18000,
            15: 87.06820,
            16: 137.08140,
            17: 114.08450,
            18: 97.10680,
            19: 103.08450,
            20: 113.18460,
            21: 329.20000,
            22: 305.20000,
            23: 345.20000,
            24: 306.20000
        }

        with open(mol_pdb, "r") as pdb:
            atms = pdb.readlines()[2:]
            for atm in atms:
                atm_split = atm.split()
                if atm_split[0] == "ATOM":
                    mass += masses[int(atm_split[2])]
                    particles += 1
        return [mass, particles]

    def gen_percent_system(self, p):
        p = p / 100
        r = 1 - p
        proteinM = []
        protein_n = []
        for i in self.molecules.keys():
            proteinM.append(self.molecules[i][0])
            protein_n.append(self.molecules[i][1])

        rnaM = self.molecules["RNA"][0]
        rna_n = self.molecules["RNA"][1]

        obj = [-1, -1, -1, -1, -1, -1, -1]

        lhs_ineq = [
            [protein_n[0], protein_n[1], protein_n[2], protein_n[3], protein_n[4], protein_n[5], rna_n],
            [proteinM[0] * (1 - p), proteinM[1] * (1 - p), proteinM[2] * (1 - p), proteinM[3] * (1 - p),
             proteinM[4] * (1 - p), proteinM[5] * (1 - p), -p * rnaM],
            [-proteinM[0] * r, -proteinM[1] * r, -proteinM[2] * r, -proteinM[3] * r, -proteinM[4] * r, -proteinM[5] * r,
             rnaM * (1 - r)]
        ]

        rhs_ineq = [
            self.nParticles,
            0,
            0
        ]

        lhs_eq = [
            [1, -2, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 1, -1, 0]
        ]

        rhs_eq = [
            0,
            0,
            0,
            0,
            0,
        ]

        bnd = [
            (0, float("inf")),
            (0, float("inf")),
            (0, float("inf")),
            (0, float("inf")),
            (0, float("inf")),
            (0, float("inf")),
            (0, float("inf"))
        ]

        opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
                      method="interior-point")

        print(opt)

        lines = []
        i = 0

        for key in self.molecules.keys():
            num = int(np.round(opt.x[i]))
            if num > 1:
                lines.append(key + ".pdb " + str(num) + " no")
            i += 1

        return lines

    def gen_protein_system(self, p, protein):
        p = p / 100
        r = 1 - p
        proteinM = self.molecules[protein][0]
        protein_n = self.molecules[protein][1]
        if p == 1:
            rnaM = 0
            rna_n = 0

        print(protein_n)
        print(proteinM)
        print(rna_n)
        print(rnaM)


        obj = [-1, -1]

        lhs_ineq = [
            [protein_n, rna_n],
            [proteinM * (1 - p), -p * rnaM],
            [-proteinM * r, rnaM * (1 - r)]
        ]

        rhs_ineq = [
            self.nParticles,
            0,
            0
        ]

        bnd = [
            (0, 120),
            (0, 120),
        ]

        opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd,
                      method="interior-point")

        print(opt)

        print(str(opt.x[0]*protein_n+opt.x[1]*rna_n))
        print(str(opt.x[0] * proteinM))
        print(str(opt.x[1] * rnaM))

        lines = []
        i = 0

        num_p = int(np.round(opt.x[0]))
        num_rna = int(np.round(opt.x[1]))
        lines.append(protein + ".pdb " + str(num_p) + " no")
        #lines.append("RNA.pdb " + str(num_rna) + " no")

        x = 0
        fid = open("GENDATA/molSG_PURE_" + protein + ".in", "w")
        mass = 0

        for line in lines:
            fid.write(line + "\n")
            num = int(line.split()[1])
            mass += num * self.molecules[protein][0]

        x = np.cbrt((mass * 1 / 0.3 * 1 / 0.5) / self.rho)

        print(x)

        return lines

    def write_percent_in_files(self):
        percent = 0
        x = {}
        while percent <= 100:
            if not os.path.exists("SG_Systems/System_{}/".format(percent)):
                os.mkdir("SG_Systems/System_{}/".format(percent))
            mass = 0
            lines = self.gen_percent_system(percent)
            fid = open("GENDATA/molSG_" + str(percent) + ".in", "w")
            for line in lines:
                fid.write(line + "\n")
                mol = line.split()[0].split(".")[0]
                num = int(line.split()[1])
                mass += num * self.molecules[mol][0]

            x[percent] = np.cbrt((mass*1/0.3*1/0.5)/self.rho)

            search = "PERCENT"
            replace = str(percent)

            with open("lammps_mpipi_script.in", 'r') as file:
                data = file.read()
                data = data.replace(search, replace)

            with open("SG_Systems/System_{}/lammps_mpipi_script{}.in".format(replace, replace), 'w') as file:
                file.write(data)

            with open("mpipi.slurm", 'r') as file:
                data = file.read()
                data = data.replace(search, replace)

            with open("SG_Systems/System_{}/mpipi.slurm".format(replace, replace), 'w') as file:
                file.write(data)

            percent += self.partition
        return x

    def write_protein_in_files(self):
        list = ["G3BP1", "PABP1", "TIA1", "TTP", "TDP43", "FUS"]
        x = {}
        for key in list:
            if not os.path.exists("SG_Systems/System_PURE_{}/".format(key)):
                os.mkdir("SG_Systems/System_PURE_{}/".format(key))
            mass = 0
            lines = self.gen_protein_system(100, key)
            fid = open("GENDATA/molSG_PURE_" + key + ".in", "w")
            for line in lines:
                fid.write(line + "\n")
                mol = line.split()[0].split(".")[0]
                num = int(line.split()[1])
                mass += num * self.molecules[mol][0]

            x[key] = np.cbrt((mass*1/0.3*1/0.5)/self.rho)

            search = "PERCENT"
            replace = str(key)

            with open("lammps_mpipi_script.in", 'r') as file:
                data = file.read()
                data = data.replace(search, replace)

            with open("SG_Systems/System_PURE_{}/lammps_mpipi_script{}.in".format(replace, replace), 'w') as file:
                file.write(data)

            with open("mpipi.slurm", 'r') as file:
                data = file.read()
                data = data.replace(search, replace)

            with open("SG_Systems/System_PURE_{}/mpipi.slurm".format(replace, replace), 'w') as file:
                file.write(data)

        return x

if __name__ == '__main__':
    nParticles = int(sys.argv[1])
    partition = int(sys.argv[2])
    mol_list = sys.argv[3]
    protein = sys.argv[4]

    generator = GenSG(nParticles, partition, mol_list)
    generator.gen_protein_system(100, protein)

    #print(generator.gen_percent_system())
    #dim = generator.write_in_files()
    #print(dim)

    dim = generator.write_protein_in_files()
    print(dim)
