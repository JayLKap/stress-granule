import numpy as np

def reshape_to_subarrays(arr, p, q):
    n, m = arr.shape

    # Check if p and q are factors of n and m
    if n % p != 0 or m % q != 0:
        raise ValueError("p must be a factor of n and q must be a factor of m")

    # Calculate the number of sub-arrays
    num_subarrays_n = n // p
    num_subarrays_m = m // q

    # Initialize the list to hold the sub-arrays
    subarrays = []

    for i in range(num_subarrays_n):
        for j in range(num_subarrays_m):
            # Extract each sub-array
            sub_array = arr[i * p:(i + 1) * p, j * q:(j + 1) * q]
            subarrays.append(sub_array)
    ave = np.mean(np.array(subarrays), axis=0)
    return ave


# Example usage
arr = np.array([[2, 2, 4, 4],
                [2, 2, 4, 4],
                [8, 8, 12, 12],
                [8, 8, 12, 12]])

arr = np.array([[2, 3, 4, 5],
                [1, 4, 3, 6],
                [7, 8, 11, 12],
                [2, 9, 10, 13]])

p = 2
q = 2

ave = reshape_to_subarrays(arr, p, q)
print(ave)


biopolymer_list = ["ProteinG3BP1", "ProteinTDP43", "ProteinFUS", "ProteinPABP1", "ProteinTIA1", "ProteinTTP","RNA"]
biopolymer_length = [466, 414, 526, 636, 386, 326, 840]


for i in range(len(biopolymer_list)):
    for j in range(i, len(biopolymer_list)):
        for k in range(0,2000,50):
            bio_i = biopolymer_list[i]
            bio_j = biopolymer_list[j]
            file = "ANALYSIS_SG/Domain_Contacts_Total_sg_X_{}_{}_{}.csv.npz".format(bio_i,bio_j,k)
            arr = np.load(file)