import os
import pandas as pd
import xarray as xr
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from matplotlib import pyplot as plt

class sst_prcp_ds(Dataset):
    def __init__(self, data_path = '/Users/cruiseryy/Documents/GitHub/CAM_DEMO/demo_data/', 
                 sst_file = 'HadISST_sst.nc', 
                 prcp_file = 'sta_monthly.csv', 
                 channel = 1,
                 lag = 0,
                 start = 0,
                 end = 240,
                 baseline = (0, 240)):
        
        # this is a bit here because we want do prediction here
        # the predictand prcp is over 1981-01 to 2020-12 (480 samples)
        # the predictor sst must be extended (by a buffer length) when lags are considered 
        # but this buffer length should not affect how the baseline period is defined
        # thus the climatology is computed over an index range of (buffer+b[0], sbuffer+b[1])
        # and to select predictors at a certain lag of tau, we use x[idx-tau+buffer]
        self.buffer = 24

        self.channel = channel
        self.lag = lag

        with xr.open_dataset(data_path + sst_file) as sst0:
    
            sst0 = sst0.sst.sel(latitude=slice(60, -55), time=slice('1979-01-01', '2021-01-01'))
            sst0 = sst0.where(sst0>0, 0)

            sst_clim = sst0.isel(time=slice(self.buffer+baseline[0], self.buffer+baseline[1])).groupby('time.month')
            sst_clim_avg = sst_clim.mean(dim='time')
            sst_clim_std = (sst_clim.std(dim='time') + 1e-6)

            self.sst = (sst0.isel(time=slice(start, end + self.buffer)).groupby('time.month') - sst_clim_avg).groupby('time.month') / sst_clim_std 

            self.sst.coords['longitude'] = (self.sst.coords['longitude'] + 360) % 360 
            self.sst = self.sst.sortby(self.sst.longitude)

        prcp0 = np.mean(np.loadtxt(data_path + prcp_file), axis=1)

        prcp_clim_avg = np.zeros([12, ])
        prcp_clim_std = np.zeros([12, ])
        for i in range(12):
            prcp_clim_avg[i] = np.mean(prcp0[baseline[0]+i:baseline[1]:12])
            prcp_clim_std[i] = np.std(prcp0[baseline[0]+i:baseline[1]:12])

        nn = (end - start) // 12 
        self.prcp = (prcp0[start:end] - np.tile(prcp_clim_avg, [nn, ])) / np.tile(prcp_clim_std, [nn, ]) 

        pause = 1

      
    def __len__(self):
        return len(self.prcp)

    def __getitem__(self, idx):
        sst_stack = np.zeros([self.channel, self.sst.shape[1], self.sst.shape[2]])
        for i in range(self.channel):
            sst_stack[i, :, :] = self.sst.isel(time=idx+self.buffer-self.lag-i).to_numpy()
        xx = torch.from_numpy(sst_stack).type(torch.float32)
        yy = 1 if self.prcp[idx] > 0 else 0
        yy = torch.from_numpy(np.array(yy)).type(torch.float32)
        return xx, yy

if __name__ == '__main__':
    test_data = sst_prcp_ds(channel=3, lag=2)
    xx, yy = test_data.__getitem__(idx=0)
    pause = 1


