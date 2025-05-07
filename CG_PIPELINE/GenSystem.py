import os
import sys
import shutil
import GenFiles as GenFiles
import GenSG as GenSG

class GenSystem:

    def __init__(self, mol_file, x_dim, y_dim, z_dim, new_mol, new_system):
        if new_mol == "T":
            import CG_Script as CG_Script
            cg = CG_Script.CG_Script()

        if new_system == "T":
            os.chdir("GENDATA/")
            gendata = (
                "python3 gen_data.py {} mpipi_database_new.py -exclude 'angles dihedrals' -avoid_overlap False -init random -contain 'no' -box \"{} {} {}\"".format(
                    mol_file, x_dim, y_dim, z_dim))
            os.system(gendata)
            os.chdir("..")

        gen = GenFiles.Settings("parameters.csv", "GENDATA/sys.data", "bonds.txt")

        for pdb in sorted(os.listdir("CG_PDB_Files/")):
            gen.write_mol_file("CG_PDB_Files/" + pdb)


if __name__ == '__main__':
    sg_set = (sys.argv[1])

    if sg_set == "T":
        nParticles = int(sys.argv[2])
        partition = int(sys.argv[3])
        mol_list = (sys.argv[4])
        gen_sg = GenSG.GenSG(nParticles, partition, mol_list)
        dim = gen_sg.write_in_files()
        for key in dim.keys():
            mol_file = "molSG_"+str(key)+".in"
            x = dim[key]
            y = dim[key]
            z = dim[key]
            new_mol = "F"
            new_system = "T"
            generator = GenSystem(mol_file, x, y, z, new_mol, new_system)
            shutil.copyfile("sys_start.data", "SG_Systems/System_{}/sys_{}.data".format(key, key))
            shutil.copyfile("sys.settings", "SG_Systems/System_{}/sys.settings".format(key))

    else:
        mol_file = "molList.txt"
        x = 2400
        y = 2400
        z = 2400
        new_mol = "F"
        new_system = "F"
        protein = "G3BP1"
        generator = GenSystem(mol_file, x, y, z, new_mol, new_system)
        shutil.copyfile("sys.data", "SG_Systems/System_PURE_{}/sys_{}.data".format(protein, protein))
        shutil.copyfile("sys.settings", "SG_Systems/System_PURE_{}/sys.settings".format(protein))
