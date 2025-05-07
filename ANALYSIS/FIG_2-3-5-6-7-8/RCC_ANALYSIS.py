import MDAnalysis
import sys
import csv
import numpy
import numpy as np
from MDAnalysis.analysis.distances import distance_array
import pandas as pd
import sklearn.decomposition
from MDAnalysis.analysis import contacts
import random
import MDAnalysis.analysis.msd as msd
import warnings
from tqdm import tqdm

class RDP:

    def __init__(self, gro_file, dcd_file, stress_file, cluster_file, msd_file, system_name, tmin, tmax, dt,
                 cutoff, dims, bin_size):

        # Define Saving Directory and File
        self.folder = "ANALYSIS/"
        self.system_name = system_name
        self.stress_file = stress_file
        self.cluster_file = cluster_file
        self.msd_file = msd_file

        # Define MDAnalysis Universe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Define MDAnalysis Universe")
            self.u = MDAnalysis.Universe(gro_file, dcd_file)
            print("Trajectory Frames: {}".format(len(self.u.trajectory)))
            self.u.add_TopologyAttr("mass")
            self.define_masses()

        # Generate Component Resnames
        self.proteins = self.get_proteins()
        self.rnas = self.get_rna()
        self.sms = self.get_sm()

        # Generate System Properties
        self.num_total_chains = self.get_total_chain_num()
        self.mass_total_chains = self.get_total_chain_mass()

        # Time Intervals
        self.tmin = int(tmin / 200000 * 1000000 / 5)
        print("Start Frame: {}".format(self.tmin))
        self.beg = 0
        self.tmax = int(tmax / 200000 * 1000000 / 5)
        print("End Frame: {}".format(self.tmax))
        self.end = 2000
        self.dt = int(dt / 200000 * 1000000 / 5)
        print("Step Frames: {}".format(self.dt))
        self.cutoff = cutoff
        self.box_dimensions = dims
        self.bin_size = bin_size
        self.system_num = str(tmin)

        # Instantiate Droplet Properties
        self.cluster_num = {}
        self.cluster_rg = {}
        self.num_inner_chains = {}
        self.mass_inner_chains = {}
        self.num_outer_chains = {}
        self.mass_outer_chains = {}
        print("Complete")

        # Generate Frame Cluster Groups
        print("Generate Frame Cluster Groups")
        self.cluster_group = {}
        self.cluster_array = {}
        self.create_clusters()
        print("Complete")

        # Obtain Maximal Continuous Cluster 
        print("Obtain Maximal Continuous Cluster")
        self.max_selection = ""
        self.max_continuous_cluster = self.gen_maximal_continuous_cluster()
        print("Complete")

        # Generate Density Profiles and Perform PCA
        try:
            print("Generate Density Profiles and Perform PCA")
            print("SG Density Profile")
            self.calc_rdp("SG")
            self.calc_pca("SG")
            if len(self.u.select_atoms("resname Protein*")) > 0:
                print("Protein Analysis")
                self.calc_rdp("Protein")
                self.calc_pca("Protein")
            if len(self.u.select_atoms("resname RNA")) > 0:
                print("RNA Analysis")
                self.calc_rdp("RNA")
                self.calc_pca("RNA")
            if len(self.u.select_atoms("resname SM")) > 0:
                print("SM Analysis")
                self.calc_rdp_sm()
            print("Complete")
        except:
            print("Failed RDP and Density")

        # Generate Residue Contact Array
        try:
            print("Generate Residue Contact Array")
            radius = 16
            self.gen_residue_contacts(radius)
            print("Complete")
        except:
            print("Failed Residue Contact")

        # Generate Acid Contact Array
        try:
            print("Generate Acid Contact Array")
            radius = 16
            self.gen_acid_contacts(radius)
            print("Complete")
        except:
            print("Failed Acid Contact")

        # Generate SM Residue Contact Array
        if len(self.u.select_atoms("resname SM")) > 0:
            try:
                print("Generate SM Residue Contact Array")
                radius = 16
                self.gen_sm_residue_contacts(radius)
                print("Complete")
            except:
                print("Failed SM Residue Contact")

        # Generate SM Acid Contact Array
        if len(self.u.select_atoms("resname SM")) > 0:
            try:
                print("Generate SM Acid Contact Array")
                radius2 = 16
                self.gen_sm_acid_contacts(radius2)
                print("Complete")
            except:
                print("Failed SM Acid Contact")

        # Generate Self-Diffusivity
        try:
            print("Calculate Diffusion Coefficient")
            self.calc_diffusivities(select="ProteinG3BP1")
            self.calc_diffusivities(select="ProteinTDP43")
            self.calc_diffusivities(select="ProteinPABP1")
            self.calc_diffusivities(select="ProteinFUS")
            self.calc_diffusivities(select="ProteinTTP")
            self.calc_diffusivities(select="ProteinTIA1")
            self.calc_diffusivities(select="Protein*")
            self.calc_diffusivities(select="RNA")

            print("Complete")
        except:
            print("Failed Diffusion")

        # Generate Stress Tensor
        try:
            print("Generate Stress Tensor")
            if self.tmin == 0:
                self.parse_stress_file(stress_file)
                print("Complete")
        except:
            print("Failed Stress Tensor")

    # Assign Bead Masses
    def define_masses(self):
        masses = {
            1: 131.182200,  # 1
            2: 57.042200, # 2
            3: 128.182600,  # 3
            4: 101.086300, # 4
            5: 156.178800,  # 5
            6: 71.070340,  # 6
            7: 115.084400,  # 7
            8: 129.082500,  # 8
            9: 163.177800,  # 9
            10: 99.056500,  # 10
            11: 113.184600,  # 11
            12: 128.082600,  # 12
            13: 186.174700,  # 13
            14: 147.180000,  # 14
            15: 87.068200, # 15
            16: 137.081400,  # 16
            17: 114.084500,  # 17
            18: 97.106800, # 18
            19: 103.084500,  # 19
            20: 113.184600,  # 20
            21: 329.200000,  # 21
            22: 305.200000,  # 22
            23: 345.200000,  # 23
            24: 306.200000,  # 24
            25: 156.140680,  # 25
            26: 180.203840,  # 26
            27: 194.236100,  # 27
            28: 113.223490,  # 28
            29: 110.199580,  # 29
            30: 125.104050,  # 30
            31: 136.107080,  # 31
            32: 104.108280,  # 32
            33: 158.287740,  # 33
            34: 78.129220,  # 34
            35: 104.152440,  # 35
            36: 118.176380,  # 36
            37: 108.216760,  # 37
            38: 102.133500,  # 38
            39: 158.177540,  # 39
            40: 107.132190,  # 40
            41: 134.134620,  # 41
            42: 81.050770,  # 42
            43: 139.130990,  # 43
            44: 173.212450,  # 44
            45: 101.125530,  # 45
            46: 107.208790,  # 46
            47: 145.160890,  # 47
            48: 100.140800,  # 48
            49: 105.192850,  # 49
            50: 101.125530,  # 50
            51: 105.192850,  # 51
            52: 188.139480,  # 52
            53: 141.193410,  # 53
            54: 115.155470,  # 54
            55: 195.244070,  # 55
            56: 92.120520,  # 56
            57: 91.112550,  # 57
            58: 117.170730,  # 58
            59: 174.245980,  # 59
            60: 112.538610,  # 60
            61: 106.124220,  # 61
            62: 181.301770,  # 62
        }
        for atom_index in range(len(self.u.atoms)):
            self.u.atoms[atom_index].mass = masses[int(self.u.atoms[atom_index].name)]





    # Assign Residue Names
    def get_proteins(self):
        for i in range(1, 21):
            prot_list = self.u.select_atoms("name " + str(i)).residues
        for res in prot_list.residues:
            if len(res.atoms) == 466:
                res.resname = "ProteinG3BP1"
            elif len(res.atoms) == 636:
                res.resname = "ProteinPABP1"
            elif len(res.atoms) == 414:
                res.resname = "ProteinTDP43"
            elif len(res.atoms) == 386:
                res.resname = "ProteinTIA1"
            elif len(res.atoms) == 326:
                res.resname = "ProteinTTP"
            elif len(res.atoms) == 526:
                res.resname = "ProteinFUS"
        return prot_list

    def get_rna(self):
        for i in range(21, 25):
            rna_list = self.u.select_atoms("name " + str(i)).residues
        for res in rna_list.residues:
            res.resname = "RNA"
        return rna_list

    def get_sm(self):
        sm_list = []
        for i in range(25, 63):
            if str(i) in self.u.atoms.names:
                sm_list = self.u.select_atoms("name " + str(i)).residues
            if len(sm_list) > 0:
                for res in sm_list.residues:
                    res.resname = "SM"
        return sm_list

    def get_sg(self):
        sg_res = len(self.rnas + self.proteins)
        sg_list = []
        for i in range(sg_res):
            sg_list.append(self.u.select_atoms("resid " + str(i)))
        return sg_list





    # Get Residue Lists
    def get_protein_pure(self):
        protein_res = len(self.proteins)
        protein_list = []
        for i in range(protein_res):
            protein_list.append(self.u.select_atoms("resid " + str(i)))
        return protein_list

    def get_rna_pure(self):
        rna_res = len(self.rnas)
        rna_list = []
        for i in range(rna_res):
            rna_list.append(self.u.select_atoms("resid " + str(i)))
        return rna_list

    def get_sm_pure(self):
        sm_res = len(self.sm)
        sm_list = []
        for i in range(sm_res):
            sm_list.append(self.u.select_atoms("resid " + str(i)))
        return sm_list





    # Calculate Time Independent System Properties
    def get_total_chain_num(self):
        ret_dict = {"Protein": len(self.u.select_atoms("resname Protein*").residues),
                    "RNA": len(self.u.select_atoms("resname RNA").residues),
                    "SG": len(self.u.select_atoms("resname RNA or resname Protein*").residues),
                    "SM": len(self.u.select_atoms("resname SM").residues)}
        return ret_dict

    def get_total_chain_mass(self):
        ret_dict = {"Protein": sum(self.u.select_atoms("resname Protein*").masses) * 1.66 * 10 ** (-21),
                    "RNA": sum(self.u.select_atoms("resname RNA").masses) * 1.66 * 10 ** (-21),
                    "SG": sum(self.u.select_atoms("resname RNA or resname Protein*").masses) * 1.66 * 10 ** (-21),
                    "SM": sum(self.u.select_atoms("resname SM").masses) * 1.66 * 10 ** (-21)}
        return ret_dict





    # Calculate Time Dependent Cluster Properties
    def get_ave_length(self, cluster):
        ave_length = int(np.round(len(cluster.atoms) / len(cluster.residues)))
        return ave_length

    def get_ave_mass(self, cluster):
        ave_mass = sum(cluster.masses) / len(cluster.residues) * 1.66 * 10 ** (-27) * 6.022 * 10 ** (23) * 10 ** 3
        return ave_mass

    def get_inner_chain_num(self, cluster):
        ret_dict = {"Protein": len(cluster.select_atoms("resname Protein*").residues),
                    "RNA": len(cluster.select_atoms("resname RNA").residues),
                    "SG": len(cluster.select_atoms("resname RNA or resname Protein*").residues),
                    "SM": len(cluster.select_atoms("resname SM").residues)}
        return ret_dict

    def get_inner_chain_mass(self, cluster):
        ret_dict = {"Protein": sum(cluster.select_atoms("resname Protein*").masses) * 1.66 * 10 ** (-21),
                    "RNA": sum(cluster.select_atoms("resname RNA").masses) * 1.66 * 10 ** (-21),
                    "SG": sum(cluster.select_atoms("resname RNA or resname Protein*").masses) * 1.66 * 10 ** (-21),
                    "SM": sum(cluster.select_atoms("resname SM").masses) * 1.66 * 10 ** (-21)}
        return ret_dict

    def get_outer_chain_num(self, cluster):
        ret_dict = {"Protein": self.num_total_chains["Protein"] - self.get_inner_chain_num(cluster)["Protein"],
                    "RNA": self.num_total_chains["RNA"] - self.get_inner_chain_num(cluster)["RNA"],
                    "SG": self.num_total_chains["SG"] - self.get_inner_chain_num(cluster)["SG"],
                    "SM": self.num_total_chains["SM"] - self.get_inner_chain_num(cluster)["SM"]}
        return ret_dict

    def get_outer_chain_mass(self, cluster):
        ret_dict = {"Protein": self.mass_total_chains["Protein"] - self.get_inner_chain_mass(cluster)["Protein"],
                    "RNA": self.mass_total_chains["RNA"] - self.get_inner_chain_mass(cluster)["RNA"],
                    "SG": self.mass_total_chains["SG"] - self.get_inner_chain_mass(cluster)["SG"],
                    "SM": self.mass_total_chains["SM"] - self.get_inner_chain_mass(cluster)["SM"]}
        return ret_dict





    # Generate Time Dependent Clusters
    def create_clusters(self):
        nres = len(self.u.select_atoms("resname Protein* or resname RNA").residues)
        res_list = self.get_sg()
        res = np.arange(-0.5, nres + 0.5, 1)

        for t in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            boxa = t.dimensions
            bond = np.zeros((nres, nres))
            cluster = np.zeros(nres)
            c = self.gen_cluster(nres, bond, cluster, res, res_list, boxa)
            cluster = c[0]
            clus = c[1]
            self.cluster_group[t.frame] = cluster
            self.cluster_array[t.frame] = clus

    def calc_cluster(self, cluster, n, proximity):
        for i in range(n):  # start with each particle in its own cluster
            cluster[i] = i
        nchange = 1
        while nchange > 0:
            nchange = 0  # change cluster assignment until convergence
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if proximity[i, j] == 1:
                        if cluster[i] != cluster[j]:
                            nchange += 1
                            ii = min(cluster[i], cluster[j])  # reduce cluster assignment to lowest cluster index
                            cluster[i] = ii
                            cluster[j] = ii
        return (cluster)

    def gen_cluster(self, nres, bond, cluster, res, res_list, boxa):
        for i in range(nres - 1):  # construct the "proximity" matrix.
            for j in range(i + 1, nres):
                d = distance_array(res_list[i].positions, res_list[j].positions, box=boxa)
                test = np.less(d, [self.cutoff])
                if np.sum(test) >= 1:
                    bond[i, j] = 1
                    bond[j, i] = 1

        clus = self.calc_cluster(cluster, nres, bond)

        hist = np.histogram(clus, bins=res)
        index_bc = np.argmax(hist[0])  # get the index of the biggest cluster.
        cluster = res_list[index_bc]

        for i in range(nres):  # look for the proteins in the biggest cluster
            if clus[i] == index_bc:
                if i != index_bc:
                    cluster = cluster.union(res_list[i])

        return cluster, clus





    # Determine Maximal Continuous Cluster
    def gen_maximal_continuous_cluster(self):
        select = ""
        res_id_array = self.cluster_group[self.tmin].resids
        for time in self.cluster_group.keys():
            current_resids = self.cluster_group[time].resids
            res_id_array = set(res_id_array).intersection(current_resids)

        res_ids = list(res_id_array)

        if len(res_ids) > 2:
            for element in res_ids[0:-2]:
                select += "resid {} or ".format(element)
            select += "resid {}".format(res_ids[-1])

        elif len(res_ids) == 2:
                select += "resid {} or ".format(res_ids[0])
                select += "resid {}".format(res_ids[1])

        elif len(res_ids) < 2:
            n = len(self.u.residues.resids)
            res_id_random = random.randint(1, n)
            select += "resid {}".format(res_id_random)

        self.max_selection = select
        fid = open("{}Max_Continuous_Cluster_{}_{}.txt".format(self.folder, self.system_name, self.system_num), "w")
        fid.write(select)

        return self.u.select_atoms(select)




    # Generate Density Profile of Cluster
    def calc_rdp(self, res_type):
        if res_type == "Protein":
            file_name = "Protein_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname Protein*"
        elif res_type == "RNA":
            file_name = "RNA_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname RNA"
        else:
            file_name = "SG_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname Protein* or resname RNA"
        r_temp = []
        dims = 0

        while dims < self.box_dimensions / 2:
            r_temp.append(dims)
            dims += self.bin_size
        r = np.array(r_temp)
        nr = len(r)

        density = []
        vol = np.zeros(nr - 1)
        for i in range(nr - 1):
            vol[i] = (4 / 3) * np.pi * (r[i + 1] ** 3 - r[i] ** 3)

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            frame = ts.frame
            timestep = frame * 200000 / 1000000
            boxa = ts.dimensions
            cluster_i = self.cluster_group[frame]
            cluster_array_i = self.cluster_array[frame]

            if len(cluster_i.residues) >= 40:
                avg_length = self.get_ave_length(cluster_i.select_atoms(select))
                avg_mass = self.get_ave_mass(cluster_i.select_atoms(select))
                R = distance_array(
                    np.float32(np.array([cluster_i.center_of_mass(), cluster_i.center_of_mass()])),
                    cluster_i.select_atoms(select).positions, box=boxa)
                h = np.histogram(R[0], bins=r)
                density.append((h[0] / vol) * (avg_mass / (6.022 * avg_length)) * 10 ** 4)

                self.cluster_num[timestep] = len(np.unique(cluster_array_i))
                self.cluster_rg[timestep] = cluster_i.select_atoms(select).radius_of_gyration()
                self.num_inner_chains[timestep] = self.get_inner_chain_num(cluster_i)[res_type]
                self.mass_inner_chains[timestep] = self.get_inner_chain_mass(cluster_i)[res_type]
                self.num_outer_chains[timestep] = self.get_outer_chain_num(cluster_i)[res_type]
                self.mass_outer_chains[timestep] = self.get_outer_chain_mass(cluster_i)[res_type]

            else:
                h = np.histogram(1000, bins=r)
                density.append(h[0])

        density_avg = np.zeros(nr - 1)
        density_std = np.zeros(nr - 1)  # standard deviation
        density_se = np.zeros(nr - 1)  # standard error
        le = len(density)

        if le > 0:
            row_list = [
                ['Timestep', 'Total Mass (mg)', 'Total Chain Number', 'Largest Droplet Radius of Gyration',
                 'Number of Droplets', 'Chains in Largest Droplet', 'Mass of Largest Droplet (mg)',
                 'Number of External Chains', 'Mass of External Chains']]

            with open("{}Cluster_{}".format(self.folder, file_name), "w") as file:
                writer = csv.writer(file)
                for key in self.cluster_num.keys():
                    row_list.append(
                        [key, self.mass_total_chains[res_type], self.num_total_chains[res_type], self.cluster_rg[key],
                         self.cluster_num[key], self.num_inner_chains[key], self.mass_inner_chains[key],
                         self.num_outer_chains[key], self.mass_outer_chains[key]])
                writer.writerows(row_list)

            for i in range(nr - 1):
                s = 0
                sig = 0
                for ts in range(le):
                    s += density[ts][i]
                density_avg[i] = s / le
                for ts in range(le):
                    sig += (density[ts][i] - density_avg[i]) ** 2
                if le > 1:
                    density_std[i] = (sig / (le - 1)) ** (0.5)
                    density_se[i] = density_std[i] / (le ** 0.5)
                else:
                    density_std[i] = 0
                    density_se[i] = 0
            rplot = 0.5 * (r[1:] + r[:-1])

            row_list = [['Distance from center of mass (A)', 'Protein density (mg/mL)', 'Standard deviation',
                         'Standard mean error']]

            for i in range(len(density_avg)):
                el = [rplot[i], density_avg[i], density_std[i], density_se[i]]
                row_list.append(el)
            with open("{}DensityProfile_{}".format(self.folder, file_name), "w") as file:
                writer = csv.writer(file)
                writer.writerows(row_list)




    # PCA Analysis for Surface Fluctuations
    def calc_pca(self, res_type):
        Time = np.arange(self.tmin, self.tmax, self.dt)
        nt = int(np.round(self.tmax - self.tmin) / self.dt)
        lambda1 = np.zeros(nt)  # principal value  (related to the length of the principal axes of the ellipsoid)
        lambda2 = np.zeros(nt)
        lambda3 = np.zeros(nt)
        ax1, ax2, ax3 = np.zeros((nt, 3)), np.zeros((nt, 3)), np.zeros((nt, 3))  # principal axis
        ts = 0
        for t in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            clus_atoms = self.cluster_group[t.frame]
            if res_type == "SG":
                test_atoms = clus_atoms
                file_name = "{}PCA_SG_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
            elif res_type == "Protein":
                test_atoms = clus_atoms.select_atoms("resname Protein*")
                file_name = "{}/PCA_Protein_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
            elif res_type == "RNA":
                test_atoms = clus_atoms.select_atoms("resname RNA")
                file_name = "{}/PCA_RNA_{}_{}.csv".format(self.folder, self.system_name, self.system_num)

            pos = test_atoms.positions - clus_atoms.center_of_mass()
            X = np.tensordot(pos, [1, 0, 0], axes=1)
            Y = np.tensordot(pos, [0, 1, 0], axes=1)
            Z = np.tensordot(pos, [0, 0, 1], axes=1)
            data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
            features = data.columns[0:3]
            x = data.loc[:, features].values
            pca_dat = sklearn.decomposition.PCA(n_components=3)
            principalComponents_dat = pca_dat.fit_transform(x)
            lambda1[ts] = pca_dat.explained_variance_[0]
            lambda2[ts] = pca_dat.explained_variance_[1]
            lambda3[ts] = pca_dat.explained_variance_[2]
            ax1[ts] = pca_dat.components_[0]
            ax2[ts] = pca_dat.components_[1]
            ax3[ts] = pca_dat.components_[2]
            ts += 1

        row_list = [
            ['Time (us)', 'l1', 'l2', 'l3', 'ax1x', 'ax1y', 'ax1z', 'ax2x', 'ax2y', 'ax2z''ax3x', 'ax3y', 'ax3z']]
        for i in range(nt):
            el = [Time[i], lambda1[i], lambda2[i], lambda3[i], ax1[i][0], ax1[i][1], ax1[i][2], ax2[i][0], ax2[i][1],
                  ax2[i][2], ax3[i][0], ax3[i][1], ax3[i][2]]
            row_list.append(el)
        with open(file_name, 'w',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)




    # Generate Small Molecule Density Profile
    def calc_rdp_sm(self):
        r_temp = []
        dims = 0

        while dims < 2400 / 2:
            r_temp.append(dims)
            dims += self.bin_size
        r = np.array(r_temp)
        nr = len(r)

        density = []
        vol = np.zeros(nr - 1)
        for i in range(nr - 1):
            vol[i] = (4 / 3) * np.pi * (r[i + 1] ** 3 - r[i] ** 3)

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            frame = ts.frame
            boxa = ts.dimensions
            cluster_i = self.cluster_group[frame]
            sm_atoms = self.u.select_atoms("resname SM")

            avg_mass = sum(sm_atoms.masses) / len(sm_atoms) * 1.66 * 10 ** (-27) * 6.022 * 10 ** (23) * 10 ** 3
            avg_length = len(sm_atoms) / len(sm_atoms.residues.resids)

            R = distance_array(
                np.float16(np.array([cluster_i.center_of_mass(), cluster_i.center_of_mass()])),
                sm_atoms.positions, box=boxa)
            h = np.histogram(R[0], bins=r)
            density.append((h[0] / vol) * (avg_mass / (6.022 * avg_length)) * 10 ** 4)

        density_avg = np.zeros(nr - 1)
        density_std = np.zeros(nr - 1)  # standard deviation
        density_se = np.zeros(nr - 1)  # standard error
        le = len(density)

        if le > 0:
            for i in range(nr - 1):
                s = 0
                sig = 0
                for ts in range(le):
                    s += density[ts][i]
                density_avg[i] = s / le
                for ts in range(le):
                    sig += (density[ts][i] - density_avg[i]) ** 2
                if le > 1:
                    density_std[i] = (sig / (le - 1)) ** (0.5)
                    density_se[i] = density_std[i] / (le ** 0.5)
                else:
                    density_std[i] = 0
                    density_se[i] = 0
            rplot = 0.5 * (r[1:] + r[:-1])

            row_list = [['Distance from center of mass (A)', 'SM density (mg/mL)', 'Standard deviation',
                         'Standard mean error']]

            for i in range(len(density_avg)):
                el = [rplot[i], density_avg[i], density_std[i], density_se[i]]
                row_list.append(el)
            with open("{}DensityProfile_SM_{}_{}.csv".format(self.folder, self.system_name, self.system_num), "w") as file:
                writer = csv.writer(file)
                writer.writerows(row_list)


    # Generate Residue Contacts
    def gen_residue_contacts(self, radius):
        contact_dict = {"ProteinG3BP1": 0,
                        "ProteinPABP1": 1,
                        "ProteinTTP": 2,
                        "ProteinTIA1": 3,
                        "ProteinTDP43": 4,
                        "ProteinFUS": 5,
                        "RNA": 6
                        }
        n = len(contact_dict.keys())
        contact_array = np.zeros((n, n))

        nt = 0
        total_contacts = 0

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            atom_group = self.cluster_group[ts.frame]
            clus_residues = atom_group.residues.resids
            for i in clus_residues:
                for j in clus_residues:
                    if i != j:
                        sel1 = i
                        sel2 = j
                        selection1 = atom_group.select_atoms("resid {}".format(sel1))
                        selection2 = atom_group.select_atoms("resid {}".format(sel2))
                        res_id1 = selection1.residues.resnames[0]
                        res_id2 = selection2.residues.resnames[0]
                        dist = contacts.distance_array(selection1.positions, selection2.positions)
                        test = np.less(dist, [radius])
                        if np.sum(test) >= 1:
                            ind1 = contact_dict[res_id1]
                            ind2 = contact_dict[res_id2]
                            contact_array[ind1][ind2] += 1
                            total_contacts += 1
            nt += 1

        if total_contacts > 0:
            contacts_mean = np.divide(contact_array, total_contacts * nt)
        else:
            contacts_mean = contact_array

        numpy.savetxt("{}Residue_Contacts_Mean_{}_{}.csv".format(self.folder, self.system_name, self.system_num), contacts_mean,
                      delimiter=",")

    def gen_acid_contacts(self, radius):
        num_bonds = pd.read_csv("Bonded_Per_Acid_Num.csv").to_numpy()

        acid_dict = {1: "Met",
                     2: "Gly",
                     3: "Lys",
                     4: "Thr",
                     5: "Arg",
                     6: "Ala",
                     7: "Asp",
                     8: "Glu",
                     9: "Tyr",
                     10: "Val",
                     11: "Leu",
                     12: "Gln",
                     13: "Trp",
                     14: "Phe",
                     15: "Ser",
                     16: "His",
                     17: "Asn",
                     18: "Pro",
                     19: "Cys",
                     20: "Ile",
                     21: "A",
                     22: "C",
                     23: "G",
                     24: "U",
                     }

        n = len(acid_dict.keys())
        clus_atoms = acid_dict.keys()
        contact_array = np.zeros((n, n))
        nt = 0
        total_contacts = 0

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            atom_group = self.cluster_group[ts.frame]
            for i in clus_atoms:
                i = int(i)
                for j in clus_atoms:
                    j = int(j)
                    if j >= i:
                        sel1 = "(type {})".format(i)
                        sel2 = "(type {})".format(j)

                        selection1 = atom_group.select_atoms(sel1)
                        selection2 = atom_group.select_atoms(sel2)

                        dist = contacts.distance_array(selection1.positions, selection2.positions)

                        test = np.less(dist, [radius])

                        average_contacts = np.sum(test)

                        num_acids = (len(selection1) + len(selection2)) / 2

                        num_bonds_acids = num_acids * num_bonds[(i - 1), (j - 1)]

                        if i == j:
                            contact_array[(i - 1), (j - 1)] += (average_contacts * 0.5) - num_bonds_acids
                        else:
                            contact_array[(i - 1), (j - 1)] += average_contacts - num_bonds_acids

                        total_contacts += average_contacts - num_bonds_acids

            nt += 1

        if total_contacts > 0:
            contacts_mean = np.divide(contact_array, total_contacts * nt)
        else:
            contacts_mean = contact_array

        for i, j in np.ndindex(contacts_mean.shape):
            if j < i:
                contacts_mean[i, j] = contacts_mean[j, i]

        np.savetxt("{}Acid_Contacts_Mean_{}_{}.csv".format(self.folder, self.system_name, self.system_num),
                   contacts_mean, delimiter=",")




    def gen_sm_residue_contacts(self, radius):
        contact_dict = {"ProteinG3BP1": 0,
                        "ProteinPABP1": 1,
                        "ProteinTTP": 2,
                        "ProteinTIA1": 3,
                        "ProteinTDP43": 4,
                        "ProteinFUS": 5,
                        "RNA": 6
                        }

        n = len(contact_dict.keys())
        contact_array = np.zeros((n))
        total_contacts = 0
        nt = 0

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            selection1 = self.u.select_atoms("resname SM")
            res_com = selection1.center_of_mass(compound='residues')
            for i in contact_dict.keys():
                selection2 = self.u.select_atoms("resname {}".format(i))
                contact_num = np.sum(np.less(contacts.distance_array(res_com, selection2.positions, box=ts.dimensions), [radius]))
                ind1 = contact_dict[i]
                contact_array[ind1] += contact_num
                total_contacts += contact_num / 2
            nt += 1
        
        if total_contacts > 0:
            contact_array = np.divide(contact_array, total_contacts*nt)

        np.savetxt("{}Residue_SM_Contacts_Mean_{}_{}.csv".format(self.folder, self.system_name, self.system_num), contact_array, delimiter=",")




    def gen_sm_acid_contacts(self, radius):
        acid_dict = {1: "Met",
                     2: "Gly",
                     3: "Lys",
                     4: "Thr",
                     5: "Arg",
                     6: "Ala",
                     7: "Asp",
                     8: "Glu",
                     9: "Tyr",
                     10: "Val",
                     11: "Leu",
                     12: "Gln",
                     13: "Trp",
                     14: "Phe",
                     15: "Ser",
                     16: "His",
                     17: "Asn",
                     18: "Pro",
                     19: "Cys",
                     20: "Ile",
                     21: "A",
                     22: "C",
                     23: "G",
                     24: "U",
                     }

        n = len(acid_dict.keys())
        clus_atoms = acid_dict.keys()
        contact_array = np.zeros(n)
        total_contacts = 0
        nt = 0

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            selection1 = self.u.select_atoms("resname SM")
            res_com = selection1.center_of_mass(compound='residues')
            for i in clus_atoms:
                selection2 = self.u.select_atoms("name {}".format(i))
                dist = contacts.distance_array(res_com, selection2.positions, box=ts.dimensions)
                test = np.less(dist, [radius])
                contact_num = np.sum(test)
                contact_array[i-1] += contact_num
                total_contacts += contact_num/2
            nt += 1

        if total_contacts > 0:
            contacts_mean = np.divide(contact_array, total_contacts*nt)

        np.savetxt("{}Acid_SM_Contacts_Mean_{}_{}.csv".format(self.folder, self.system_name, self.system_num), contacts_mean, delimiter=",")


    # Generate Diffusion Coeffients
    def calc_diffusivities(self, select):
        bio_list = list(set(self.max_continuous_cluster.select_atoms('resname {}'.format(select)).resids))
        print(bio_list)
        for biopolymer in bio_list:
            MSD_SG = msd.EinsteinMSD(self.max_continuous_cluster, select='resid {}'.format(biopolymer),
                                     msd_type='xyz', fft=True)
            MSD_SG.run()
            msd_sg = MSD_SG.results.timeseries
            msd_sg = np.divide(msd_sg, 10**20)
        
            nframes = MSD_SG.n_frames
            timestep = 200000 * 10 ** (-15)  # this needs to be the actual time between frames
            lagtimes = np.arange(nframes) * timestep  # make the lag-time axis
            rg = []
            for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
                res = self.cluster_group[ts.frame].select_atoms("resid {}".format(biopolymer)).residues
                rg.append(res.atoms.radius_of_gyration())
            Rg = np.mean(np.array(rg))
            df = pd.DataFrame({"MSD (um)": msd_sg,
                                "Time (s)": lagtimes,
                               "Rg": Rg,
                                })
            df.to_csv("{}{}_Diffusivity_{}_{}_{}.csv".format(self.folder, select, self.system_name, self.system_num, biopolymer), index=False)

    # Generate Stress Tensors
    def parse_stress_file(self, stress_file):
        write_lines = [["Timestep", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]]
        cluster_list = self.max_continuous_cluster.atoms.indices
        pxx = 0.0
        pyy = 0.0
        pzz = 0.0
        pxy = 0.0
        pxz = 0.0
        pyz = 0.0
        n = 0
        timestep = 0
        with open(stress_file, "r+") as file:
            for line in file:
                if line.strip() == "ITEM: TIMESTEP":
                    write_lines.append([timestep, pxx, pyy, pzz, pxy, pxz, pyz])
                    timestep = int(next(file))
                    pxx = 0.0
                    pyy = 0.0
                    pzz = 0.0
                    pxy = 0.0
                    pxz = 0.0
                    pyz = 0.0
                    n = 0
                elif len(line.strip().split()) == 16:
                    atm = line.split()
                    atom_id = int(atm[0])
                    if atom_id in cluster_list:
                        pxx += float(atm[6])
                        pyy += float(atm[7])
                        pzz += float(atm[8])
                        pxy += float(atm[9])
                        pxz += float(atm[10])
                        pyz += float(atm[11])
                        n += 1
        self.end_time = timestep
        with open("{}Stress_Tensor_{}_{}.csv".format(self.folder, self.system_name, self.system_num), "w") as file:
            writer = csv.writer(file)
            writer.writerows(write_lines)


if __name__ == '__main__':
    cutoff = 20
    dims = 1200
    bin_size = 20

    name = sys.argv[1]
    tmin = int(sys.argv[2])
    tmax = int(sys.argv[3])
    dt = int(sys.argv[4])

    gro_file = "GRO/{}_traj.gro".format(name)
    dcd_file = "TRJ/{}_whole.xtc".format(name)
    #gro_file = "GRO/{}_traj.gro".format(name)
    #dcd_file = "DCD/dcd_Y_hexanediol.dcd"
    stress_file = "STRESS/{}_stress.out.all".format(name)
    cluster_file = "CLUSTER/{}_cluster.out.all".format(name)
    msd_file = "MSD/{}_msd.out.all".format(name)

    rdp = RDP(gro_file=gro_file, dcd_file=dcd_file, stress_file=stress_file, cluster_file=cluster_file,
              msd_file=msd_file, system_name=name, tmin=tmin, tmax=tmax,
              dt=dt, cutoff=cutoff, dims=dims, bin_size=bin_size)
