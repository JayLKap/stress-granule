import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import sys

if __name__ == '__main__':

    def max_cluster_ave(folder, cat_save, sm, tmin, tmax, dt):
        count = 1
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        residues = []
        for i in range(tmin,tmax,dt):
            with open("{}/Max_Continuous_Cluster_{}_{}.txt".format(category, sm, i), 'r') as file:
             resids = file.readlines()[0].split(" or resid ")
             resids[0] = resids[0].split("resid ")[1]
             resid_list = []
             for i in resids:
                 resid_list.append(int(i))
             residues.append(resid_list)
        res_list = residues[0]
        for i in range(1,len(residues)):
            if len(residues[i])>2:
                res_list = res_list + residues[i]
                count += 1
            else:
                pass
        df = pd.DataFrame(columns=["Step",
                                   "SG",
                                   "TDP43",
                                   "FUS",
                                   "TIA1",
                                   "G3BP1",
                                   "PABP1",
                                   "TTP",
                                   "RNA"])

        step = 1
        for res in residues:
            biopolymer_name = {
                "Step": 0,
                "SG": 0,
                "TDP43": 0,
                "FUS": 0,
                "TIA1": 0,
                "G3BP1": 0,
                "PABP1": 0,
                "TTP": 0,
                "RNA": 0
            }
            biopolymer_name["Step"] = step
            if len(res) > 2:
                for i in res:
                    biopolymer_name["SG"] += 1
                    if i <= 33:
                        biopolymer_name["G3BP1"] += 1
                    elif 33 < i <= 49:
                        biopolymer_name["PABP1"] += 1
                    elif 49 < i <= 65:
                        biopolymer_name["TIA1"] += 1
                    elif 65 < i <= 81:
                        biopolymer_name["TTP"] += 1
                    elif 81 < i <= 97:
                        biopolymer_name["FUS"] += 1
                    elif 97 < i <= 113:
                        biopolymer_name["TDP43"] += 1
                    elif 113 < i <= 135:
                        biopolymer_name["RNA"] += 1
                step += 1

                df = df._append(biopolymer_name, ignore_index=True)

        row_mean = np.array(df.mean())[1:]
        row_sem = np.array(df.sem())[1:]
        df_mean = pd.DataFrame(columns = ["Biopolymer"])
        df_mean["Biopolymer"] = ["SG",
                                 "TDP43",
                                 "FUS",
                                 "G3BP1",
                                 "TIA1",
                                 "PABP1",
                                 "TTP",
                                 "RNA"
                                ]
        df_mean["Mean"] = row_mean
        df_mean["SEM"] = row_sem

        df_mean.to_csv("{}/{}/BioNumDF_{}.csv".format(folder,cat_save,sm), index=False)





        biopolymer_count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0
        }

        biopolymer_name = {
            "G3BP1": 0,
            "PABP1": 0,
            "TTP": 0,
            "TIA1": 0,
            "TDP43": 0,
            "FUS": 0,
            "RNA": 0
        }

        for i in res_list:
            if i <= 33:
                #G3BP1 SIM (0)
                #G3BP1 CON (0)
                biopolymer_count[0] += 1
                biopolymer_name["G3BP1"] += 1
            elif 33 < i <= 49:
                # PABP1 SIM (1)
                # PABP1 CON (1)
                biopolymer_count[1] += 1
                biopolymer_name["PABP1"] += 1
            elif 49 < i <= 65:
                # TIA1 SIM (2)
                # TIA1 CON (3)
                biopolymer_count[3] += 1
                biopolymer_name["TIA1"] += 1
            elif 65 < i <= 81:
                # TTP SIM (3)
                # TTP CON (2)
                biopolymer_count[2] += 1
                biopolymer_name["TTP"] += 1
            elif 81 < i <= 97:
                # FUS SIM (4)
                # FUS CON (5)
                biopolymer_count[5] += 1
                biopolymer_name["FUS"] += 1
            elif 97 < i <= 113:
                # TDP43 SIM (5)
                # TDP43 CON (4)
                biopolymer_count[4] += 1
                biopolymer_name["TDP43"] += 1
            elif 113 < i <= 135:
                # RNA SIM (6)
                # RNA CON (6)
                biopolymer_count[6] += 1
                biopolymer_name["RNA"] += 1

        for i in biopolymer_count.keys():
            biopolymer_count[i] /= count

        print(sm)

        print(df_mean)

        bio_count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0
        }
        print(biopolymer_count)
        return biopolymer_count

    def gen_ave_biopolymer_ni_nj(path, biopolymer_num, sm, category):
        n = 7
        bio_arr = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    bio_arr[i,j] = (biopolymer_num[i]*(biopolymer_num[j]-1))/2
                else:
                    bio_arr[i,j] = (biopolymer_num[i] * (biopolymer_num[j]))
        np.savetxt("{}/{}/BioPolNum_{}.csv".format(path,category,sm), bio_arr, delimiter=",")

    def gen_agg_biopolymer_ni_nj(sm_list, path):
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/BioPolNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 0
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/BioPolNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AGG/BioPolNum_{}.csv".format(path, category, category), index=False, header=False)



    def gen_acid_contacts(biopolymer_num_dict, path, category, sm):
        biopolymer_acids = {
            "G3BP1": 0,
            "PABP1": 0,
            "TTP": 0,
            "TIA1": 0,
            "TDP43": 0,
            "FUS": 0,
            "RNA": 0
        }
        n = 24
        count = 0
        count_bio = 0
        bond_array = np.zeros((n, n))
        biopolymers = ["G3BP1","TDP43","FUS","PABP1","TIA1","TTP"]

        acid_array = np.zeros((n))

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
            "G3BP1": biopolymer_num_dict[0],
            "PABP1": biopolymer_num_dict[1],
            "TTP": biopolymer_num_dict[2],
            "TIA1": biopolymer_num_dict[3],
            "TDP43": biopolymer_num_dict[4],
            "FUS": biopolymer_num_dict[5],
            "RNA": biopolymer_num_dict[6]
        }


        for i in biopolymers:
            aa = []
            bio_array = np.zeros((n, n))
            na_array = np.zeros((n))
            file_path = "CM_NORM/{}_seq.txt".format(i)
            with open(file_path, 'r') as file:
                content = file.read()
                for char in content:
                    if char.isalpha():
                        aa.append(char)

            count += len(aa) * biopolymer_num[i]
            biopolymer_acids[i] = len(aa)


            for j in range(len(aa)):
                for k in range(j+1,j+4):
                    try:
                        if aa_dict[aa[j]] <= aa_dict[aa[k]]:
                            bio_array[aa_dict[aa[j]], aa_dict[aa[k]]] += 1
                        else:
                            bio_array[aa_dict[aa[k]], aa_dict[aa[j]]] += 1
                    except:
                        pass

            for j in range(len(aa)):
                na_array[aa_dict[aa[j]]] += 1

            acid_array += na_array * biopolymer_num[i]
            bond_array += bio_array * biopolymer_num[i]

        na = []
        bio_array = np.zeros((n, n))
        file_path = "CM_NORM/ZACN_seq.txt"
        with open(file_path, 'r') as file:
            content = file.read()
            for char in content:
                if char.isalpha():
                    na.append(char)

        count += len(na) * biopolymer_num["RNA"]
        biopolymer_acids["RNA"] = len(na)
        print(biopolymer_acids)


        for j in range(len(na)):
            for k in range(j + 1, j + 3):
                try:
                    if na_dict[na[j]] <= na_dict[na[k]]:
                        bio_array[na_dict[na[j]], na_dict[na[k]]] += 1
                    else:
                        bio_array[na_dict[na[k]], na_dict[na[j]]] += 1
                except:
                    pass
        bond_array += bio_array * biopolymer_num["RNA"]

        for j in range(len(na)):
            na_array[na_dict[na[j]]] += 1

        acid_array += na_array * biopolymer_num["RNA"]

        for i in range(n):
            for j in range(i + 1, n):
                bond_array[j][i] = bond_array[i][j]

        print("Biopolymers: {}".format(count))
        print("Acids: {}".format(np.sum(bond_array)))
        np.savetxt("{}/{}/BondNum_{}.csv".format(path, category, sm), bond_array, delimiter=",")

        return acid_array, biopolymer_acids


    def gen_acid_ni_nj(path, category, sm, nAAcids):
        n = 24
        bio_arr = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    bio_arr[i,j] = (nAAcids[i]*(nAAcids[j]))
                else:
                    bio_arr[i, j] = (nAAcids[i]*(nAAcids[j]))

        np.savetxt("{}/{}/AcidPolNum_{}.csv".format(path,category,sm), bio_arr, delimiter=",")


    def gen_agg_acid_ni_nj(sm_list, path):
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/AcidPolNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/AcidPolNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AGG/AcidPolNum_{}.csv".format(path, category, category), index=False, header=False)

    def gen_agg_bond_ni_nj(sm_list, path):
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/BondNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/BondNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AGG/BondNum_{}.csv".format(path, category, category), index=False, header=False)

    def gen_agg_bio_count(sm_list, path):
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/BioNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/BioNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AGG/BioNum_{}.csv".format(path, category, category), index=False, header=False)

    def gen_agg_acid_count(sm_list, path):
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/AcidNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/AcidNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AGG/AcidNum_{}.csv".format(path, category, category), index=False, header=False)


    path = sys.argv[1]
    dt = int(sys.argv[2])
    tmin = int(sys.argv[3])
    tmax = int(sys.argv[4])

    sm = "sg_X"
    category = "ANALYSIS_SG_AVE"
    biopolymer_count = max_cluster_ave(path, category, sm, tmin, tmax, dt)

    df = pd.DataFrame(biopolymer_count, index=[0])
    df.to_csv("{}/{}/BioNum_{}.csv".format(path, category, sm), index=False, header=False)
    gen_ave_biopolymer_ni_nj(path, biopolymer_count, sm, category)
    nAAcids, bio_acids = gen_acid_contacts(biopolymer_count, path, category, sm)
    np.savetxt("{}/{}/AcidNum_{}.csv".format(path, category, sm), np.array(nAAcids), delimiter=",")
    gen_acid_ni_nj(path, category, sm, nAAcids)
    print(bio_acids)



    if "THEORETICAL" in path:
        dsm = []
        ndsm = []
        list_path = path.replace("_THEORETICAL", "")
        with open('{}/dsm_list.txt'.format(list_path), 'r') as f:
            for i in f.readlines():
                dsm.append(i.strip())

        with open('{}/ndsm_list.txt'.format(list_path), 'r') as f:
            for i in f.readlines():
                ndsm.append(i.strip())

    else:
        dsm = ["dsm_anisomycin", "dsm_daunorubicin", "dsm_dihydrolipoicacid", "dsm_hydroxyquinoline", "dsm_lipoamide",
           "dsm_lipoicacid", "dsm_mitoxantrone", "dsm_pararosaniline", "dsm_pyrivinium", "dsm_quinicrine"]
        ndsm = ["ndsm_dmso", "ndsm_valeric", "ndsm_ethylenediamine", "ndsm_propanedithiol",
                "ndsm_hexanediol", "ndsm_diethylaminopentane", "ndsm_aminoacridine",
                "ndsm_anthraquinone", "ndsm_acetylenapthacene", "ndsm_anacardic"]

    for i in dsm:
        category = "ANALYSIS_DSM_AVE"
        biopolymer_count = max_cluster_ave(path, category, i, tmin, tmax, dt)
        df = pd.DataFrame(biopolymer_count, index=[0])
        df.to_csv("{}/{}/BioNum_{}.csv".format(path, category, i), index=False, header=False)
        gen_ave_biopolymer_ni_nj(path, biopolymer_count, i, category)
        nAAcids, bio_acids = gen_acid_contacts(biopolymer_count, path, category, i)
        np.savetxt("{}/{}/AcidNum_{}.csv".format(path, category, i), np.array(nAAcids), delimiter=",")
        gen_acid_ni_nj(path, category, i, nAAcids)

    gen_agg_biopolymer_ni_nj(dsm, path)
    gen_agg_acid_ni_nj(dsm, path)
    gen_agg_bond_ni_nj(dsm, path)
    gen_agg_bio_count(dsm, path)
    gen_agg_acid_count(dsm, path)




    for i in ndsm:
        category = "ANALYSIS_NDSM_AVE"
        biopolymer_count = max_cluster_ave(path, category, i, tmin, tmax, dt)
        df = pd.DataFrame(biopolymer_count, index=[0])
        df.to_csv("{}/{}/BioNum_{}.csv".format(path, category, i), index=False, header=False)
        gen_ave_biopolymer_ni_nj(path, biopolymer_count, i, category)
        nAAcids, bio_acids = gen_acid_contacts(biopolymer_count, path, category, i)
        np.savetxt("{}/{}/AcidNum_{}.csv".format(path, category, i), np.array(nAAcids), delimiter=",")
        gen_acid_ni_nj(path, category, i, nAAcids)

    gen_agg_biopolymer_ni_nj(ndsm, path)
    gen_agg_acid_ni_nj(ndsm, path)
    gen_agg_bond_ni_nj(ndsm, path)
    gen_agg_bio_count(ndsm, path)
    gen_agg_acid_count(ndsm, path)


    def generic():
        biopolymer_num = {
            0: 33,
            1: 16,
            2: 16,
            3: 16,
            4: 16,
            5: 16,
            6: 21
        }

        biopolymer_count = {
            "G3BP1": 33,
            "PABP1": 16,
            "TTP": 16,
            "TIA1": 16,
            "TDP43": 16,
            "FUS": 16,
            "RNA": 21
        }

        df = pd.DataFrame(biopolymer_count, index=[0])
        df.to_csv("{}/{}/BioNum_{}.csv".format("CM_NORM", "MAPS", "SYSTEM"), index=False, header=False)

        gen_ave_biopolymer_ni_nj("CM_NORM", biopolymer_num, "SYSTEM", "MAPS")

        nAAcids, bio_acids = gen_acid_contacts(biopolymer_num, "CM_NORM", "MAPS", sm="SYSTEM")

        np.savetxt("{}/{}/AcidNum_{}.csv".format("CM_NORM", "MAPS", "SYSTEM"), np.array(nAAcids), delimiter=",")

        gen_acid_ni_nj("CM_NORM", "MAPS", "SYSTEM", nAAcids)


    generic()










