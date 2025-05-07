import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    def gen_biopolymer_ni_nj():
        n = 7
        bio_arr = np.zeros((n,n))
        biopolymers = ["G3BP1", "TDP43", "FUS", "PABP1", "TIA1", "TTP"]

        biopolymer_pos = {
                        "G3BP1": 0,
                        "PABP1": 1,
                        "TTP": 2,
                        "TIA1": 3,
                        "TDP43": 4,
                        "FUS": 5,
                        "RNA": 6
        }
        biopolymer_num = {
            0: 32,
            1: 16,
            2: 16,
            3: 16,
            4: 16,
            5: 16,
            6: 22
        }

        for i in range(n):
            for j in range(n):
                if i == j:
                    bio_arr[i,j] = (biopolymer_num[i]*(biopolymer_num[j]-1))
                else:
                    bio_arr[i, j] = (biopolymer_num[i] * (biopolymer_num[j]))
        """
        for i in range(n):
            for j in range(n):
                bio_arr[i, j] = (biopolymer_num[i]*(biopolymer_num[j]))
        """
        np.savetxt("../BioPolNumJAY.csv", bio_arr, delimiter=",")

        
    def gen_acid_ni_nj():
        n = 24
        bio_arr = np.zeros((n,n))

        nAAcids = [
            3088,
            800,
            2208,
            2304,
            2736,
            4832,
            2144,
            2448,
            3392,
            592,
            6560,
            3696,
            3200,
            2928,
            1520,
            2400,
            1472,
            2080,
            1664,
            512,
            6248,
            3256,
            4774,
            4202
        ]

        for i in range(24):
            for j in range(24):
                if i == j:
                    bio_arr[i,j] = (nAAcids[i]*(nAAcids[j]-1))/2
                else:
                    bio_arr[i, j] = (nAAcids[i]*(nAAcids[j]))
        """
        for i in range(24):
            for j in range(24):
                bio_arr[i, j] = (nAAcids[i]+(nAAcids[j]))/2
       
        """
        np.savetxt("../AcidPolNumJAY.csv", bio_arr, delimiter=",")





    def gen_acid_contacts():
        n = 24
        count = 0
        count_bio = 0
        bond_array = np.zeros((n, n))
        biopolymers = ["G3BP1","TDP43","FUS","PABP1","TIA1","TTP"]

        aa_dict = {'M': 0,
                   'G': 1,
                   'K': 2,
                   'T': 3,
                   'R': 4,
                   'A': 5,
                   'D': 6,
                   'E': 7,
                   'Y': 8,
                   'V': 9,
                   'I': 10,
                   'Q': 11,
                   'W': 12,
                   'F': 13,
                   'S': 14,
                   'H': 15,
                   'N': 16,
                   'P': 17,
                   'C': 18,
                   'L': 19,
                   }

        na_dict = {'A': 20,
                   'C': 21,
                   'T': 22,
                   'G': 23,
                  }

        biopolymer_num = {
            "G3BP1": 32,
            "TDP43": 16,
            "FUS": 16,
            "PABP1": 16,
            "TIA1": 16,
            "TTP": 16,
            "ZACN":22
        }


        for i in biopolymers:
            aa = []
            bio_array = np.zeros((n, n))
            file_path = "{}_seq.txt".format(i)
            with open(file_path, 'r') as file:
                content = file.read()
                for char in content:
                    if char.isalpha():
                        aa.append(char)

            count += len(aa) * biopolymer_num[i]
            print("{}: {}".format(i,len(aa)))
            for j in range(len(aa)):
                for k in range(j+1,j+5):
                    try:
                        if aa_dict[aa[j]] <= aa_dict[aa[k]]:
                            bio_array[aa_dict[aa[j]], aa_dict[aa[k]]] += 1
                        else:
                            bio_array[aa_dict[aa[k]], aa_dict[aa[j]]] += 1
                    except:
                        pass
            bond_array += bio_array * biopolymer_num[i]

        na = []
        bio_array = np.zeros((n, n))
        file_path = "ZACN_seq.txt"
        with open(file_path, 'r') as file:
            content = file.read()
            for char in content:
                if char.isalpha():
                    na.append(char)

        count += len(na) * biopolymer_num["ZACN"]
        print("{}: {}".format("ZACN", len(na)))

        for j in range(len(na)):
            for k in range(j + 1, j + 4):
                try:
                    if na_dict[na[j]] <= na_dict[na[k]]:
                        bio_array[na_dict[na[j]], na_dict[na[k]]] += 1
                    else:
                        bio_array[na_dict[na[k]], na_dict[na[j]]] += 1
                except:
                    pass
        bond_array += bio_array * biopolymer_num["ZACN"]



        for i in range(n):
            for j in range(i + 1, n):
                bond_array[j][i] = bond_array[i][j]

        sns.heatmap(bond_array)
        plt.show()

        print("Biopolymers: {}".format(count))
        print("Acids: {}".format(np.sum(bond_array)))
        np.savetxt("../BondNum.csv", bond_array, delimiter=",")

"""
        a = "MMMMMMAAAAAAPPPPPPTTTTTTUUUUUURRRRRRUUUUUU"
        aa_dict = {
            "M": 0,
            "A": 1,
            "P": 2,
            "T": 3,
            "U": 4,
            "R": 5,
        }
        arr = np.zeros((6, 6))
        for j in range(len(a)):
            for k in range(j + 1, j + 5):
                try:
                    if aa_dict[a[j]] <= aa_dict[a[k]]:
                        arr[aa_dict[a[j]], aa_dict[a[k]]] += 1
                    else:
                        arr[aa_dict[a[k]], aa_dict[a[j]]] += 1
                except:
                    pass
        print(arr)
        print(np.sum(arr))
"""

gen_acid_contacts()
gen_acid_ni_nj()

