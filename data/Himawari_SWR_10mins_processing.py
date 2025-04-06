import xarray as xr
import glob
import os

# Directory containing the downloaded Himawari-9 NetCDF files
data_dir = "data/Himawari9_PAR_20250404/"
file_pattern = os.path.join(data_dir, "*.nc")
nc_files = sorted(glob.glob(file_pattern))
print(f"Found {len(nc_files)} files.")

def preprocess(ds):
    """
    Preprocessing function to assign the time coordinate from the start_time variable.
    This will allow xarray to order the datasets by time.
    """
    # Assign the 'start_time' variable as a coordinate named 'time'
    ds = ds.assign_coords(time=ds.start_time)
    return ds

# Open all files as a single multi-file dataset, using the preprocess function.
ds = xr.open_mfdataset(nc_files, combine='by_coords', preprocess=preprocess)
print("Combined dataset:")
print(ds)

# Select only the SWR variable (Shortwave Radiation)
ds_swr = ds[['SWR']]

# Subset the SWR data for Mindanao using the provided bounding box:
#   geospatial_lat_min = 4.47159290313721
#   geospatial_lat_max = 10.6715927124023
#   geospatial_lon_min = 117.864234924316
#   geospatial_lon_max = 126.664237976074
#
# Note: The latitude axis in the file is likely descending.
ds_swr_mindanao = ds_swr.sel(
    latitude=slice(10.6715927124023, 4.47159290313721),
    longitude=slice(117.864234924316, 126.664237976074)
)

# Write the spatiotemporal subset to a new NetCDF file
output_file = "spatiotemporal_swr_mindanao.nc"
ds_swr_mindanao.to_netcdf(output_file)
print(f"Spatiotemporal SWR dataset for Mindanao saved to {output_file}")
