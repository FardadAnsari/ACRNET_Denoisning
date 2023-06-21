import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = ['Cost2100DataLoader']


class Cost2100DataLoader(object):
    r""" PyTorch DataLoader for COST2100 dataset.
    """

    def __init__(self, root, batch_size, num_workers, scenario):
        assert os.path.isdir(root), root
        assert scenario in {"in", "out"}, scenario
        self.batch_size = batch_size
        self.num_workers = num_workers

        dir_test = os.path.join(root, f"DATA_Htest{scenario}.mat")
        dir_raw = os.path.join(root, f"DATA_HtestF{scenario}_all.mat")
        channel, nt, nc, nc_expand = 2, 32, 32, 125

        # Test data loading, including the sparse data and the raw data
        # two dimensional 20k,2048 ------> 20k,2,32,32
        data_test = sio.loadmat(dir_test)['HT']
        data_test = torch.tensor(data_test, dtype=torch.float32).view(
            data_test.shape[0], channel, nt, nc)

        raw_test = sio.loadmat(dir_raw)['HF_all']
        # two dimensional ----> 20k,4k ------> raw_test -----> 20k,32,125,2

        # real part 20k,32,125,1
        real = torch.tensor(np.real(raw_test), dtype=torch.float32)
        # imaginary part 20k,32,125,1
        imag = torch.tensor(np.imag(raw_test), dtype=torch.float32)


        # raw_test ------ > 20k,32,125,2
        raw_test = torch.cat((real.view(raw_test.shape[0], nt, nc_expand, 1),
                              imag.view(raw_test.shape[0], nt, nc_expand, 1)), dim=3)
        #### ------ >
        self.test_dataset = TensorDataset(data_test, raw_test)

    def __call__(self):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=False)

        return test_loader
