import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

# SST data collected from the Met Office Hadley Centre, UK
data = xr.open_dataset('./data/HadISST_sst.nc')
sst = data.sst

# change longitude from [-180, 180] to [0, 360] just to make life easier
sst.coords['longitude'] = (sst.coords['longitude'] + 360) % 360 
sst = sst.sortby(sst.longitude)

# clean up ice values (-1.8) and missing values (-9999)
sst = sst.where(sst>0)

# crop the tropical Pacific [20S, 20N], [120E, 280E] over 1981-2020
tp = sst.sel(longitude=slice(120, 280), latitude=slice(20, -20), time=slice('1981-01-01', '2021-01-01'))

# crop the domain that defines the Nino 3.4 index
nino34 = tp.sel(latitude=slice(5, -5), longitude=slice(190, 240))

# this is how to calculate monthly climatologies and anomalies using xarray (a little bit confusing since not repmat is used)
mm = nino34.groupby('time.month')
nino_anomaly = mm - mm.mean(dim='time')
nino_idx = nino_anomaly.mean(dim=['latitude', 'longitude'])

# # i wroted these to test if i understand the above some lines correctly
# i = j = 0
# mmm = mm.mean(dim='time')
# tmp0 = nino34[:, i, j].to_numpy()
# ss = mmm[:, i, j].to_numpy()
# tmp1 = nino_anomaly[:, 0, 0].to_numpy()
# tmp2 = tmp1 + np.tile(ss, [40, ])
# plt.scatter(tmp0, tmp2)

pause = 1