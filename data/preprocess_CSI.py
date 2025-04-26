#!/usr/bin/env python
# ---------------------------------------------------------------------
#  Compute clear-sky index (CSI = SWR / GHI_clear-sky) for Mindanao
#  Author: <your name>   Date: 2025-04-26
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import pvlib
from joblib import Parallel, delayed
from tqdm import tqdm

# ------------------------------------------------------------------ #
# 1.  Load inputs                                                    #
# ------------------------------------------------------------------ #
print("→ Loading SWR NetCDF …")
ds_swr = xr.open_dataset(
            "data/spatiotemporal_swr_mindanao.nc",
            engine="netcdf4",   # optional
            decode_times=True,  # default
            mask_and_scale=True # default → values already scaled!
         )

swr = ds_swr["SWR"].astype(float)      # READY-TO-USE, no extra factor

#   extract coords
lat = ds_swr["latitude"].values          # (124,)
lon = ds_swr["longitude"].values         # (176,)
times_utc = ds_swr.indexes["time"]

print("→ Loading resampled DEM …")
dem = rxr.open_rasterio("data/mindanao_dem_0p05.tif").squeeze()

dem = dem.fillna(0)          # ← NEW: set missing elevations to 0 m
alt = dem.values             # shape (124, 176)

# sanity check: shapes must match
assert alt.shape == (lat.size, lon.size), "DEM shape mismatch!"

# ------------------------------------------------------------------ #
# 2.  Vectorised clear-sky GHI computation                           #
# ------------------------------------------------------------------ #
n_lat, n_lon, n_t = lat.size, lon.size, times_utc.size
ghi_cs = np.empty((n_lat, n_lon, n_t), dtype=np.float32)

def clearsky_one_pixel(i, j):
    """Return clear-sky GHI (array length n_t) for pixel (i,j)."""
    loc = pvlib.location.Location(float(lat[i]), float(lon[j]),
                                  tz="UTC", altitude=float(alt[i, j]))
    return loc.get_clearsky(times_utc, model="ineichen")["ghi"].values

print("→ Computing clear-sky irradiance with pvlib (Ineichen) …")
results = Parallel(n_jobs=-1)(
    delayed(clearsky_one_pixel)(i, j)
    for i in tqdm(range(n_lat), desc="latitude")
    for j in range(n_lon)
)

# reshape list → 3-D array
ghi_cs = np.stack(results).reshape(n_lat, n_lon, n_t)

# ------------------------------------------------------------------ #
# 3.  Clear-sky index + quality filters                              #
# ------------------------------------------------------------------ #
print("→ Computing clear-sky index …")
csi = swr.transpose("latitude", "longitude", "time").values / ghi_cs

#   mask missing SWR values and very low sun (θz > 85°)
zenith = pvlib.solarposition.get_solarposition(
            times_utc, lat.mean(), lon.mean())["apparent_zenith"].values
csi[..., zenith > 85] = np.nan

#   clip physically reasonable range
np.clip(csi, 0, 1.3, out=csi)

# ------------------------------------------------------------------ #
# 4.  Save to NetCDF                                                 #
# ------------------------------------------------------------------ #
print("→ Writing clear_sky_index_mindanao.nc …")


# (i) clear-sky GHI DataArray
ghi_da = xr.DataArray(
            ghi_cs.astype(np.float32),
            coords=dict(latitude=("latitude", lat),
                        longitude=("longitude", lon),
                        time=("time", times_utc)),
            dims=("latitude", "longitude", "time"),
            name="GHI_cs",
            attrs=dict(long_name="Clear-sky global horizontal irradiance",
                       units="W m-2",
                       model="Ineichen–Perez (pvlib)")
)
ghi_da.to_netcdf("data/clear_sky_ghi_mindanao.nc", format="NETCDF4")

# (ii) clear-sky index DataArray
csi_da = xr.DataArray(
            csi.astype(np.float32),
            coords=dict(latitude=("latitude", lat),
                        longitude=("longitude", lon),
                        time=("time", times_utc)),
            dims=("latitude", "longitude", "time"),
            name="CSI",
            attrs=dict(long_name="Clear-sky index (SWR / GHI_cs)",
                       units="1")
)
csi_da.to_netcdf("data/clear_sky_index_mindanao.nc", format="NETCDF4")
print("✓ Done")
