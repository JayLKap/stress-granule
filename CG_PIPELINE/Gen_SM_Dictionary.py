import pandas as pd
from IPython.display import display


if __name__ == '__main__':
    fid = open("masses.txt", "r")
    data = fid.readlines()
    sm_y_dict = {}
    sm_n_dict = {}
    sm_list = []
    sm_class_list = []
    sm_code_list = []
    y_count = 1
    n_count = 1

    df = pd.read_csv("masses.txt", sep =" ", header = None, names = ["Bead", "MASS"])

    for line in data:
        full_name = line.split(" ")[0]
        name = full_name.split("_")[0] + "_" + full_name.split("_")[1]
        sm_class = full_name.split("_")[0]
        sm_list.append(name)
        sm_class_list.append(sm_class)


    df["SM"] = sm_list

    df["SM_Class"] = sm_class_list



    print(df)

    df2 = df.groupby('SM').sum()

    sm_class = []

    display(df2)


    for index, row in df2.iterrows():
        sm_classification = index.split("_")[0]
        sm_class.append(sm_classification)



    print(len(sm_class))

    df2["SM_Class"] = sm_class

    df2 = df2.sort_values(['SM_Class', 'MASS'])

    for index, row in df2.iterrows():
        sm_classification = index.split("_")[0]
        if sm_classification == "Y":
            sm_code_list.append("D" + str(y_count))
            y_count += 1

        elif sm_classification == "N":
            sm_code_list.append("ND" + str(n_count))
            n_count += 1



    df2["CODE"] = sm_code_list

    df3 = pd.DataFrame()
    print(df2)

    df3["CODE"] = df2["CODE"]

    print(df3)

    sm_dict = df3.to_dict()

    print(sm_dict)






