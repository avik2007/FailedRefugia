import numpy as np
import os
import argopy
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gsw
from datetime import datetime   
from argopy import DataFetcher as ArgoDataFetcher

def calculate_thermodynamics(sp, t, p, lon, lat):
    """
    Wrapper for GSW functions to be used with apply_ufunc.
    Inputs are expected to be numpy arrays (handled by xarray wrapper).
    """
    # 1. Absolute Salinity (SA)
    # GSW requires (SP, P, lon, lat)
    # We ensure P is broadcasted correctly by xarray before it gets here
    sa = gsw.SA_from_SP(sp, p, lon, lat)
    
    # 2. Conservative Temperature (CT)
    ct = gsw.CT_from_t(sa, t, p)
    
    # 3. Density (rho)
    rho = gsw.rho(sa, ct, p)
    
    # 4. Specific Heat Capacity (Cp)
    cp = gsw.cp_t_exact(sa, t, p)
    
    # 5. Energy Density (J/m^3) = rho * cp * T (in-situ)
    # We return the energy density directly
    return rho * cp * t

# once we have the ability to calculate thermodynamics for one ocean column, we can incorporate it into 
# a function that makes xarray inputs/outputs
def compute_ohc_layer(ds_input, layer_label):
    # Ensure inputs are present
    # Salinity sensors can be fouled up in Argo sensors, so we have to account for this possibility
    if 'PSAL' not in ds_input or 'TEMP' not in ds_input:
        print(f"⚠️ Missing variables in {layer_label}")
        return None

    # GSW needs Lat/Lon as arguments. 
    # ds_input['LONGITUDE'] and ['LATITUDE'] are likely 1D (by profile).
    # We don't need to manually broadcast them; apply_ufunc handles alignment 
    # if we pass them as xarray objects.
    
    # However, PRESSURE is a vertical coordinate (PRES_INTERPOLATED).
    # We must ensure it's passed correctly.
    
    # THE WRAPPER:
    # We use apply_ufunc to push the xarray objects into the numpy-based GSW function
    energy_density = xr.apply_ufunc(
        calculate_thermodynamics,
        ds_input['PSAL'],
        ds_input['TEMP'],
        ds_input['PRES_INTERPOLATED'],
        ds_input['LONGITUDE'],
        ds_input['LATITUDE'],
        input_core_dims=[[], [], [], [], []], # All inputs map point-to-point (broadcasting happens auto)
        output_core_dims=[[]], # Returns one array of same shape
        vectorize=True,        # Loops if core dims don't match (safe fallback)
        dask='parallelized',   # If you use Dask chunks later, this is ready
        output_dtypes=[float]
    )
    
    # 2. Integrate over depth (J/m^3 -> J/m^2)
    ohc = energy_density.integrate(coord='PRES_INTERPOLATED')
    
    # 3. Metadata
    ohc.name = layer_label
    ohc.attrs['units'] = 'J/m^2'
    ohc.attrs['description'] = 'Integrated Heat Content (TEOS-10: Rho*Cp*T)'
    
    return ohc


# --- TIME SERIES RESAMPLING ---
# we convert the xarray ocean heat content structure into a pandas structure. We then resample to look
# monthly mean trends. This function takes all the floats in the region we picked (no matter what the
# size, and lumps their data together, creating a terribly spatially averaged heat content, subject
# to biases in a region that may have heterogeneous heat content (north south bias, for example))
def make_timeseries(da_ohc):
    if da_ohc is None: return None
    df = da_ohc.to_dataframe().reset_index()
    df = df.set_index('TIME')
    # Resample to Monthly Mean
    return df[da_ohc.name].resample('MS').mean()

# this 
def make_spatially_weighted_timeseries(da_ohc):
    # 1. Convert to DataFrame
    df = da_ohc.to_dataframe().reset_index()
    
    # 2. Create Spatial Bins (e.g., 1-degree bins)
    # We round the Lat/Lon to the nearest integer to create "bins"
    """
     UPDATE THIS TO ALLOW FOR BINS OF ANY SIZE. 1 DEGREE COULD BE TOO SMALL 
    """
    df['lat_bin'] = df['LATITUDE'].round(0)
    df['lon_bin'] = df['LONGITUDE'].round(0)
    
    # 3. Create Time Bins (Month)
    # We use a string format for grouping: "2023-01"
    df['month_year'] = df['TIME'].dt.to_period('M')
    
    # --- THE TRICK ---
    # Step A: Average inside each 1x1 degree bin for each month
    # This collapses the 50 offshore floats into a single "grid value"
    grid_means = df.groupby(['month_year', 'lat_bin', 'lon_bin'])[da_ohc.name].mean().reset_index()
    
    # Step B: Average the grid cells for each month
    # Now every grid cell gets an equal vote, regardless of how many floats were in it
    regional_means = grid_means.groupby('month_year')[da_ohc.name].mean()
    
    # Convert index back to Timestamp for plotting
    regional_means.index = regional_means.index.to_timestamp()
    
    return regional_means

def load_argo_data_advanced(nc_dir, start_date, end_date, lat_bounds, lon_bounds, depth_bounds=[0, 200]):
    """
    Fetches Argo data directly from Erddap (Always), processes it, and saves the dataframe with a descriptive filename.
    
    Args:
        nc_dir (str): Directory where the processed DataFrame will be saved.
        start_date, end_date (str): 'YYYY-MM-DD'.
        lat_bounds, lon_bounds (list): [min, max].
        depth_bounds (list): [min, max] in meters.
        
    Returns:
        pd.DataFrame: Cleaned data [lat, lon, temp, depth, time_days, float_id, date]
    """
    # 1. SETUP
    ref_date = pd.to_datetime(start_date)
    os.makedirs(nc_dir, exist_ok=True)
    
    # GENERATE DESCRIPTIVE FILENAME
    # Format: argo_START_END_latMIN-MAX_lonMIN-MAX_zMIN-MAX.pkl
    fname = (f"argo_{start_date}_to_{end_date}_"
             f"lat{lat_bounds[0]}_{lat_bounds[1]}_"
             f"lon{lon_bounds[0]}_{lon_bounds[1]}_"
             f"z{depth_bounds[0]}_{depth_bounds[1]}.pkl")
             
    save_path = os.path.join(nc_dir, fname)

    # 2. FETCH DATA (Always)
    print(f"\n🌊 INITIATING ARGO FETCH (Source: Erddap)")
    print(f"   Target: {start_date} to {end_date}")
    print(f"   Region: Lat {lat_bounds} | Lon {lon_bounds}")
    print(f"   Depth:  {depth_bounds[0]}m to {depth_bounds[1]}m")
    print(f"   Save Path: {save_path}")
    
    try:
        fetcher = ArgoDataFetcher(src='erddap').region(
            [lon_bounds[0], lon_bounds[1], 
             lat_bounds[0], lat_bounds[1], 
             depth_bounds[0], depth_bounds[1], 
             start_date, end_date]
        )
        ds = fetcher.to_xarray()
        print(f"   ✅ Data received from server.")
    except Exception as e:
        print(f"   ❌ FETCH FAILED: {e}")
        return pd.DataFrame()

    # 3. PROCESS DATA
    print(f"   🔄 Processing profiles...")
    all_rows = []
    
    # Identify variables
    t_var = 'TEMP_ADJUSTED' if 'TEMP_ADJUSTED' in ds else 'TEMP'
    p_var = 'PRES_ADJUSTED' if 'PRES_ADJUSTED' in ds else 'PRES'
    
    if t_var not in ds:
        print("   ❌ No Temperature variable found.")
        return pd.DataFrame()

    # Extract arrays
    lats = ds.LATITUDE.values
    lons = ds.LONGITUDE.values
    times = ds.TIME.values
    temps = ds[t_var].values
    
    has_depth = p_var in ds
    if has_depth: pressures = ds[p_var].values
    
    # --- HANDLING DATA DIMENSIONS ---
    # Case A: Profile Mode (2D)
    if temps.ndim == 2:
        num_profiles = temps.shape[0]
        for p in range(num_profiles):
            lat_p = lats[p]
            lon_p = lons[p]
            time_p = times[p]
            
            if lon_p > 180: lon_p -= 360
            if not (lat_bounds[0] <= lat_p <= lat_bounds[1]): continue
            if not (lon_bounds[0] <= lon_p <= lon_bounds[1]): continue
            if pd.isnull(time_p): continue
            
            # Vertical Search
            profile_temps = temps[p, :]
            
            if has_depth:
                profile_pres = pressures[p, :]
                valid_mask = (
                    (profile_pres >= depth_bounds[0]) & 
                    (profile_pres <= depth_bounds[1]) & 
                    (~np.isnan(profile_temps))
                )
                valid_indices = np.where(valid_mask)[0]
            else:
                valid_indices = np.where(~np.isnan(profile_temps))[0]
            
            if len(valid_indices) == 0: continue 
            
            # Take shallowest valid
            best_idx = valid_indices[0] 
            t_val = profile_temps[best_idx]
            depth_val = profile_pres[best_idx] if has_depth else np.nan
            
            # ID
            try:
                if 'PLATFORM_NUMBER' in ds:
                    raw_id = ds.PLATFORM_NUMBER.values[p]
                    if isinstance(raw_id, bytes): f_id = raw_id.decode('utf-8').strip()
                    else: f_id = str(raw_id).strip()
                else: f_id = "unknown"
            except: f_id = "unknown"

            dt = pd.to_datetime(time_p)
            days_delta = (dt - ref_date).total_seconds() / 86400.0
            
            all_rows.append({
                'lat': float(lat_p),
                'lon': float(lon_p),
                'temp': float(t_val),
                'depth': float(depth_val),
                'time_days': float(days_delta),
                'float_id': f_id,
                'date': dt
            })

    # Case B: Point Mode (1D)
    elif temps.ndim == 1:
        for i in range(len(temps)):
            lat_p = lats[i]
            lon_p = lons[i]
            if lon_p > 180: lon_p -= 360
            
            if not (lat_bounds[0] <= lat_p <= lat_bounds[1]): continue
            if not (lon_bounds[0] <= lon_p <= lon_bounds[1]): continue
            
            t_val = temps[i]
            if np.isnan(t_val): continue
            
            depth_val = np.nan
            if has_depth:
                p_val = pressures[i]
                if np.isnan(p_val) or not (depth_bounds[0] <= p_val <= depth_bounds[1]):
                    continue
                depth_val = p_val
            
            dt = pd.to_datetime(times[i])
            days_delta = (dt - ref_date).total_seconds() / 86400.0
            
            try:
                raw_id = ds.PLATFORM_NUMBER.values[i]
                if isinstance(raw_id, bytes): f_id = raw_id.decode('utf-8').strip()
                else: f_id = str(raw_id).strip()
            except: f_id = "unknown"

            all_rows.append({
                'lat': float(lat_p),
                'lon': float(lon_p),
                'temp': float(t_val),
                'depth': float(depth_val),
                'time_days': float(days_delta),
                'float_id': f_id,
                'date': dt
            })

    # 4. SAVE & RETURN
    df = pd.DataFrame(all_rows)
    print(f"✅ COMPLETE: Loaded {len(df)} valid observations.")
    
    if not df.empty:
        try:
            df.to_pickle(save_path)
            print(f"   💾 DataFrame saved to: {save_path}")
        except Exception as e:
            print(f"   ⚠️ Save failed: {e}")
            
    return df