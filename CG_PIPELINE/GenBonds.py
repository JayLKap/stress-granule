import numpy as np


class GenBonds:

    def __init__(self, pdb_file, path):
        self.pdb_file = pdb_file
        self.atoms = {}
        self.bonds = {}
        self.mol_name = pdb_file.split("/")[2]
        self.path = path

    def parse_file(self):
        het_atoms = {}
        with open(self.pdb_file, 'r') as pdb:
            lines = pdb.readlines()
            for line in lines:
                line_split = line.split()
                if line_split[0] == 'HETATM':
                    atom_id = int(line_split[1])
                    het_atoms[atom_id] = {"type": line_split[2]}
                    het_atoms[atom_id]["crds_x"] = float(line_split[5])
                    het_atoms[atom_id]["crds_y"] = float(line_split[6])
                    het_atoms[atom_id]["crds_z"] = float(line_split[7])
                    het_atoms[atom_id]["type_end"] = line_split[10]

                elif line_split[0] == 'CONECT':
                    atom_id = int(line_split[1])
                    het_atoms[atom_id]["connect"] = line_split[2:]

        self.atoms = het_atoms

    def calc_distance(self, bead1, bead2):
        x1 = bead1["crds_x"]
        y1 = bead1["crds_y"]
        z1 = bead1["crds_z"]
        x2 = bead2["crds_x"]
        y2 = bead2["crds_y"]
        z2 = bead2["crds_z"]
        dist = np.sqrt(np.power((x1-x2), 2)+np.power((y1-y2), 2)+np.power((z1-z2), 2))
        return dist

    def det_bond(self):
        self.parse_file()
        bonds = {"": {"": float}}
        bond_lines = []
        for atom in self.atoms.keys():
            atom1 = self.mol_name+"_b" + str(atom)
            for con_atom in self.atoms[atom]["connect"]:
                atom2 = self.mol_name+"_b" + str(con_atom)
                dist = self.calc_distance(self.atoms[int(atom)], self.atoms[int(con_atom)])
                if atom2 > atom1:
                    bond_lines.append("{:>2s}\t{:>2s}\t{:>8f}".format(atom1, atom2, dist))
                else:
                    bond_lines.append("{:>2s}\t{:>2s}\t{:>8f}".format(atom2, atom1, dist))
        self.bonds = bonds
        bond_lines.sort()
        temp = []
        [temp.append(x) for x in bond_lines if x not in temp]
        return temp

    def write_file(self):
        bond_lines = self.det_bond()
        bond_lines.sort()

        fid = open("{}/bonds.txt".format(self.path), "w")
        fid.write("COMPND    {}\n".format(self.mol_name))
        fid.write("AUTHOR    GENERATED BY GenBonds.py\n")
        for i in bond_lines:
            fid.write(i + "\n")

    def write_master_file(self):
        bond_lines = self.det_bond()
        bond_lines.sort()
        return bond_lines


if __name__ == '__main__':
    import sys
    import itertools
    import os
    import numpy

    pdb_file = sys.argv[1]

    bondify = GenBonds(pdb_file)

    bondify.write_file()
