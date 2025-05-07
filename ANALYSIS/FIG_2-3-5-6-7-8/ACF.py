import numpy as np

class ACF:
    def __init__(self):
        pass

    def multi_kernel(self, x, t=1):
        """
        Element-wise multiplication.
        """

        if t == 1:
            result = np.multiply(x[:-t], x[t:])

        elif t == 0:
            result = np.multiply(x[:], x[:])

        return result


    def multi_kernel_block(self, x, t=1):
        """
        Moving-window multiplication.
        For example, when dt=10,
        Calculate the mean of first 10, second 10, etc.
        dt=10 will be reflected as the product of the adjacent means.
        This is better than calculating the product of raw t=0 and t=10.
        """

        n_ind = len(x) // t

        indices = np.arange(n_ind) * t + np.arange(t)[:, None]

        subarrays = x[indices]

        mean = np.mean(subarrays, axis=0)

        result = np.multiply(mean[:-1], mean[1:])

        return result


    def get_boot_data(self, p_xyz, segment, bootstrap):
        """
        Generate bootstrapped data from input pressure tensor.

        Args:
            p_xyz (array-like): Off-diagonal pressure tensor.
            segment (int, optional): Number of segments for partitioning the data. Default is 40.
            bootstrap (int, optional): Number of bootstrap iterations. Default is 1000.

        Returns:
            tuple: Tuple containing:
                - numpy.ndarray: Time steps.
                - numpy.ndarray: Bootstrapped autocorrelation functions.
        """

        acfs = []

        n = len(p_xyz)

        w = int(n / segment)  # this is the number of non-overlapping segments

        for i in range(segment):

            xyz = p_xyz[w * i:w * (i + 1)]

            acf = []

            # you can change this for different sampling frequency

            dt = np.arange(0, int(w / 2), 1)

            for j in dt:

                if j <= 1:
                    # calculate autocorrelation function. 0, 1, and 2 correspond to the xy, yz, and xz components.
                    acf_temp = self.multi_kernel(xyz[..., 0], j).mean() + self.multi_kernel(xyz[..., 1], j).mean() + self.multi_kernel(
                        xyz[..., 2], j).mean()
                else:
                    acf_temp = self.multi_kernel_block(xyz[..., 0], j).mean() + self.multi_kernel_block(xyz[..., 1],
                                                                                              j).mean() + self.multi_kernel_block(
                        xyz[..., 2], j).mean()

                acf.append(acf_temp)

            acf = np.array(acf)

            acfs.append(acf)

        acfs = np.array(acfs)

        acfs_pre_bootstrap = np.copy(acfs)

        # perform bootstrapping

        acfs_bootstrap = []

        for k in range(bootstrap):
            boot_idx = np.random.RandomState(k).choice(np.arange(len(acfs_pre_bootstrap)), len(acfs_pre_bootstrap))

            acfs_bootstrap.append(acfs_pre_bootstrap[boot_idx].mean(axis=0))

        acfs_bootstrap = np.array(acfs_bootstrap)

        return dt, acfs_bootstrap