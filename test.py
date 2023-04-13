import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

data = xr.open_dataset('./data/HadISST_sst.nc')
sst = data.sst

sst.coords['longitude'] = (sst.coords['longitude'] + 360) % 360 
sst = sst.sortby(sst.longitude)
sst = sst.where(sst>0)
tp = sst.sel(longitude=slice(120, 280), latitude=slice(20, -20), time=slice('1981-01-01', '2023-01-01'))
