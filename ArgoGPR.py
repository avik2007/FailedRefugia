import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics.pairwise import haversine_distances


from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import numpy as np
import warnings


"""
The following function provides the option for the global validation of a Gaussian Process Regression.
Use this if you are happy with coming up with one set of correlation lengths (one for lat, one for lon)
for your system. 
"""
def generalized_cross_validation(df, feature_cols=['lat', 'lon'], target_col='temp', 
                                 method='KFold', k_fold_data_percent=10,
                                 auto_tune=True, tune_subsample_frac=0.05, tune_iterations=5,
                                 length_scale_val=1.0, noise_val=0.1):
    """
    Global GP Validation (Smart Hybrid).
    Splits the data into Train/Test sets and fits ONE Gaussian Process to the entire training set.
    
    ---------------------------------------------------------------------------
    HOW TO INTERPRET & TUNE (BASED ON K-FOLD):
    ---------------------------------------------------------------------------
    The critical metric is 'Std Z' (Standard Deviation of Z-scores). 
    Ideal value is 1.0.

    1. If Std Z > 1.1 (The "Arrogant" Model):
       - Diagnosis: Model is Overconfident. It predicts small error bars, but real errors are large.
       - Cause: It thinks distant points are more related than they actually are.
       - ACTION: DECREASE 'length_scale_val' (make it rougher) OR INCREASE 'noise_val'.

    2. If Std Z < 0.9 (The "Paranoid" Model):
       - Diagnosis: Model is Underconfident. It predicts huge error bars, but predictions are actually good.
       - Cause: It ignores useful neighbors nearby, assuming they aren't relevant.
       - ACTION: INCREASE 'length_scale_val' (make it smoother) OR DECREASE 'noise_val'.

    * Note regarding LOFO: LOFO Z-scores naturally fluctuate due to spatial non-stationarity. 
      Do not obsess over tuning LOFO to 1.0; use K-Fold for tuning physics.
    ---------------------------------------------------------------------------
    
    PARAMETERS:
    ---------------------------------------------------------------------------
    1. df (pd.DataFrame): 
       The master table. Must contain feature_cols, target_col, and 'float_id'.

    2. feature_cols (list of str): 
       Dimensions for similarity (e.g., ['lat', 'lon'] for 2D, or ['lat', 'lon', 'time_days'] for 3D).
       
    3. target_col (str): 
       The variable to predict (e.g., 'temp', 'psal').

    4. method (str):
       - 'KFold': Tests interpolation accuracy by holding out random points. Best for tuning.
       - 'LOFO':  Tests scientific rigor by holding out entire instruments.

    5. k_fold_data_percent (float): 
       Percentage of data to hold out for TESTING in each fold (e.g., 10%).

    6. auto_tune (bool): 
       - True:  Runs a pre-step on random subsets to LEARN the best length/noise.
                Ignores the manual knobs below. 
       - False: Uses the manual length_scale_val and noise_val.

    7. tune_subsample_frac (float): 
       Fraction of data to use for EACH auto-tune iteration (0.0 to 1.0).

    8. tune_iterations (int):
       How many times to run the optimizer on random subsets (Default: 5).
       The final parameters are the average of these runs.

    9. length_scale_val (float): 
       MANUAL knob. Used only if auto_tune=False.
       
    10. noise_val (float): 
       MANUAL knob. Used only if auto_tune=False.
    ---------------------------------------------------------------------------
    """
    print(f"\n🚀 STARTING GLOBAL VALIDATION: {method}")

    # --- SAFETY CHECK: DIMENSION MISMATCH ---
    if not auto_tune:
        # Check if length_scale_val is a list/array (Anisotropic)
        if hasattr(length_scale_val, '__len__') and not isinstance(length_scale_val, (str, float, int)):
            if len(length_scale_val) != len(feature_cols):
                raise ValueError(
                    f"\n❌ CONFIG ERROR: You provided {len(feature_cols)} feature columns {feature_cols}, "
                    f"but a length_scale_val list of size {len(length_scale_val)}: {length_scale_val}.\n"
                    f"👉 Please provide exactly one length scale per feature, or a single float for isotropic mode."
                )
    # 1. SETUP & SCALING
    X = df[feature_cols].values
    y = df[target_col].values
    # Robust Grouping: Force to string to prevent splitter errors
    groups = df['float_id'].astype(str).values 
    mean_lat = df['lat'].mean()
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    phys_scale_X = scaler_X.scale_  
    phys_scale_y = scaler_y.scale_[0]

    # ---------------------------------------------------------
    # 2. ROBUST AUTO-TUNE STEP
    # ---------------------------------------------------------
    final_length_scale = length_scale_val
    final_noise = noise_val
    
    if auto_tune:
        N_points = len(X_scaled)

        target_n = int(N_points * tune_subsample_frac)

        if N_points < 100:
            n_sub = N_points
        else:
            n_sub = max(100, min(target_n, 2000))
        
        # Calculate Guardrails (Bounds)
        data_span = np.max(X_scaled, axis=0) - np.min(X_scaled, axis=0)
        max_dist = np.max(data_span) 
        upper_bound = max_dist * 1.5 
        lower_bound = 0.05 
        
        print(f"   🤖 AutoTuning: Running {tune_iterations} iterations on {n_sub} points ({tune_subsample_frac*100:.1f}%) to estimate correlation lengths/times...")
        print(f"      (Constraint: Length Scale capped at {upper_bound:.2f} standard deviations)")
        
        learned_ls = []
        learned_noise = []
        
        for run in range(tune_iterations):
            idx_tune = np.random.choice(N_points, n_sub, replace=False)
            X_tune, y_tune = X_scaled[idx_tune], y_scaled[idx_tune]
            
            k_tune = ConstantKernel(1.0) * \
                     RBF(length_scale=[1.0]*X.shape[1], length_scale_bounds=(lower_bound, upper_bound)) + \
                     WhiteKernel(noise_level=0.1)
            
            gp_tune = GaussianProcessRegressor(kernel=k_tune, n_restarts_optimizer=0)
            gp_tune.fit(X_tune, y_tune)
            
            learned_ls.append(gp_tune.kernel_.k1.k2.length_scale)
            learned_noise.append(gp_tune.kernel_.k2.noise_level)
            
        final_length_scale = np.mean(learned_ls, axis=0) 
        final_noise = np.mean(learned_noise)
        
        phys_ls = final_length_scale * phys_scale_X
        phys_noise_sigma = np.sqrt(final_noise) * phys_scale_y
        
        print(f"      ✅ LEARNED HYPERPARAMETERS (Avg of {tune_iterations} runs):")
        print(f"         Noise (Uncertainty): ±{phys_noise_sigma:.3f} °C")
        print(f"         Correlation Lengths:")
        
        for i, col in enumerate(feature_cols):
            val = phys_ls[i]
            if 'lat' in col.lower():
                km_val = val * 111.0
                print(f"           - {col}: {val:.3f}°  (~{km_val:.0f} km)")
            elif 'lon' in col.lower():
                km_val = val * 111.0 * np.cos(np.radians(mean_lat))
                print(f"           - {col}: {val:.3f}°  (~{km_val:.0f} km at {mean_lat:.1f}N)")
            elif 'time' in col.lower() or 'day' in col.lower():
                print(f"           - {col}: {val:.1f} days")
            else:
                print(f"           - {col}: {val:.3f} (unknown units)")
    else:
        print(f"   🔧 Using Manual Parameters: Length={final_length_scale}, Noise={final_noise}")

    # 3. CHOOSE SPLITTER
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
    #("LOOO" is not the technically accurate name for the method, but I mix it up sometimes with KFolding.)
    elif (method == 'KFold' or method=='LOOO'): 
        n_splits = int(100 / k_fold_data_percent)
        if n_splits < 2: n_splits = 2
        if n_splits >= len(df): n_splits = len(df)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"   ⚡ Strategy: {n_splits}-Fold CV (Testing {k_fold_data_percent}% per fold)")
    else:
        raise ValueError(f"Unknown method '{method}'. Please use 'LOFO' or 'KFold'.")
    
    # 4. RUN LOOP
    y_preds = []
    y_true = []
    y_sigmas = []
    MAX_TRAIN = 2000 
    
    # Explicitly pass groups to avoid LOFO error
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_scaled, groups=groups)):
        
        if len(train_idx) > MAX_TRAIN:
            train_subset = np.random.choice(train_idx, size=MAX_TRAIN, replace=False)
        else:
            train_subset = train_idx
            
        X_train = X_scaled[train_subset]
        y_train = y_scaled[train_subset]
        X_test = X_scaled[test_idx]
        
        k = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            RBF(length_scale=final_length_scale, length_scale_bounds="fixed") + \
            WhiteKernel(noise_level=final_noise, noise_level_bounds="fixed")
        
        gp = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=0.0)
        gp.fit(X_train, y_train)
        
        pred, std = gp.predict(X_test, return_std=True)
        
        y_preds.extend(pred)
        y_sigmas.extend(std)
        y_true.extend(y_scaled[test_idx])
        
        if method == 'KFold':
             if cv.get_n_splits() > 10 and i % (cv.get_n_splits() // 10) == 0:
                 print(f"   Processed fold {i+1}...", end='\r')
        else:
             if i % 5 == 0:
                 print(f"   Processed float {i+1}...", end='\r')

    # 5. SCORING
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    y_sigmas = np.array(y_sigmas)
    
    if len(y_preds) == 0: return np.array([])

    z_scores = (y_true - y_preds) / y_sigmas
    y_true_c = scaler_y.inverse_transform(y_true.reshape(-1,1)).flatten()
    y_pred_c = scaler_y.inverse_transform(y_preds.reshape(-1,1)).flatten()
    
    rmse = np.sqrt(np.mean((y_true_c - y_pred_c)**2))
    
    epsilon = 1e-9
    rel_error_vector = (y_pred_c - y_true_c) / (y_true_c + epsilon)
    rms_rel_error = np.sqrt(np.mean(rel_error_vector**2))
    
    print(f"\n✅ RESULTS ({method}):")
    print(f"   RMSE:                {rmse:.3f} °C")
    print(f"   Rel. Error (RMSRE):  {rms_rel_error:.4f} (dimensionless)")
    print(f"   Mean Z:              {np.mean(z_scores):.3f}")
    print(f"   Std Z:               {np.std(z_scores):.3f} (Ideal: 1.0)")
    
    return {
        "z_scores": z_scores,
        "rmse": rmse,
        "best_length_scale": final_length_scale, # The winner (Auto or Manual)
        "best_noise": final_noise                # The winner (Auto or Manual)
    }





"""
This is almost certaintly the best thing to use, but it is computationally incredibly costly because you study space and
time varying correlation distances. Save this for cloud computing.
"""

def validate_moving_window(df, feature_cols=['lat', 'lon'], target_col='temp', 
                           method='LOFO', k_fold_data_percent=10,
                           radius_km=300, min_neighbors=10, max_samples=1000,
                           auto_tune=True, tune_subsample_frac=0.05, tune_iterations=5,
                           length_scale_val=1.0, noise_val=0.1,
                           optimization_mode='group'): 
    """
    Moving Window (Local GP) Validation with Adaptive Optimization.
    
    PURPOSE:
    Validates the mapping strategy by simulating the final map generation process.
    It iterates through test points, identifies local neighbors within 'radius_km',
    and fits a unique Gaussian Process for that specific window.
    
    OPTIMIZATION STRATEGY ("The Goldilocks Approach"):
    1. Global Init: First, we estimate a baseline length scale from the full dataset.
    2. Local Adaptation: Then, based on 'optimization_mode', we adapt that baseline 
       to the local conditions (e.g., eddies vs gyres).
    
    ---------------------------------------------------------------------------
    PARAMETERS:
    ---------------------------------------------------------------------------
    1. df (pd.DataFrame): 
       The master data table. Must contain:
       - Feature columns (e.g., lat, lon)
       - Target column (e.g., temp)
       - 'float_id': Used for grouping in LOFO validation.

    2. feature_cols (list of str): 
       Dimensions used for similarity.
       - ['lat', 'lon']: Standard 2D mapping.
       - ['lat', 'lon', 'time_days']: 3D Spatio-Temporal mapping.

    3. target_col (str): 
       The variable to predict (e.g., 'temp', 'psal').

    4. method (str):
       - 'LOFO' (Recommended): "Leave-One-Float-Out". Tests scientific robustness 
         by holding out entire instruments. Use with optimization_mode='group'.
       - 'KFold': Random sampling. Tests interpolation accuracy. Use with 
         optimization_mode='point' or 'global'.

    5. k_fold_data_percent (float): 
       Percentage of data to test per fold (if method='KFold').

    6. radius_km (float): 
       The Horizon. 
       - Filters neighbors: Only points within this radius are used for training.
       - Bounds optimizer: Local length scales are forbidden from exceeding 
         2x this radius (prevents "Infinite Length Scale" artifacts).

    7. min_neighbors (int): 
       The Void Threshold. If a test point has fewer than this many neighbors,
       we skip prediction. Prevents unstable models in sparse regions.

    8. max_samples (int): 
       Speed Limit. Stops validation after testing this many total points.
       Essential for 'point' mode which is computationally expensive.

    9. auto_tune (bool): 
       - True: Runs a pre-loop Global Estimation step to find baseline parameters.
       - False: Uses manual 'length_scale_val' and 'noise_val' as the baseline.

    10. tune_subsample_frac (float): 
        Fraction of data to use for Global Estimation (e.g., 0.05 = 5%).

    11. tune_iterations (int):
        Number of random subsets to test during Global Estimation. 
        Averaging these runs provides a stable starting point for local models.

    12. length_scale_val / noise_val (float): 
        Manual knobs used only if auto_tune=False.

    13. optimization_mode (str) - THE CRITICAL PERFORMANCE KNOB:
       - 'group' (Recommended for LOFO): "Per-Float Optimization".
         Calculates local physics ONCE per float (averaging 3 points on the track),
         then locks those parameters to predict the rest of that float.
         * Speed: Fast (~50x faster than point).
         * Accuracy: High (Captures local physics of the water mass).
         
       - 'point' (Scientific Rigor): "Locally Stationary".
         Re-runs the optimizer for every single test point. 
         * Speed: Very Slow.
         * Accuracy: Maximum.
       
       - 'global' (Speed Check): "Globally Stationary".
         Uses the fixed Global Baseline parameters for every window.
         * Speed: Fastest.
         * Accuracy: Lower (Ignores that physics change spatially).
    ---------------------------------------------------------------------------
    """
    print(f"\n🚀 STARTING MOVING WINDOW VALIDATION: {method}")
    print(f"   Config: Radius={radius_km}km | Mode: {optimization_mode.upper()}")
    
    # 1. DATA PREP
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['float_id'].astype(str).values 
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    phys_scale_X = scaler_X.scale_
    phys_scale_y = scaler_y.scale_[0]
    X_rad = np.radians(X[:, :2]) 
    radius_rad = radius_km / 6371.0

    # 2. GLOBAL BASELINE (Initialization)
    start_length_scale = length_scale_val
    start_noise = noise_val
    
    # Calculate Max Distance for Bounds
    data_span = np.max(X_scaled, axis=0) - np.min(X_scaled, axis=0)
    max_dist = np.max(data_span)
    
    if auto_tune:
        print(f"   🤖 Global Estimator: Running {tune_iterations} iterations to find baseline...")
        N_points = len(X_scaled)
        n_sub = int(N_points * tune_subsample_frac)
        n_sub = max(100, min(n_sub, 2000))
        
        learned_ls = []
        learned_noise = []
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for run in range(tune_iterations):
                idx_tune = np.random.choice(N_points, n_sub, replace=False)
                X_tune, y_tune = X_scaled[idx_tune], y_scaled[idx_tune]
                
                k_tune = ConstantKernel(1.0) * \
                         RBF(length_scale=[1.0]*X.shape[1], length_scale_bounds=(0.05, max_dist*1.5)) + \
                         WhiteKernel(noise_level=0.1)
                
                gp_tune = GaussianProcessRegressor(kernel=k_tune, n_restarts_optimizer=0)
                gp_tune.fit(X_tune, y_tune)
                
                learned_ls.append(gp_tune.kernel_.k1.k2.length_scale)
                learned_noise.append(gp_tune.kernel_.k2.noise_level)
            
        start_length_scale = np.mean(learned_ls, axis=0) 
        start_noise = np.mean(learned_noise)
        print(f"      ✅ Baseline Found: Length={start_length_scale}, Noise={start_noise:.4f}")
    
    # 3. SPLITTER LOOP
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
    elif method == 'KFold':
        # 'group' mode requires discrete groups. If KFold (random), force global or point.
        if optimization_mode == 'group':
            print("   ⚠️  NOTE: 'group' mode requires LOFO. Switching to 'global' for KFold.")
            optimization_mode = 'global'
        n_splits = int(100 / k_fold_data_percent)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_preds = []
    y_true = []
    y_sigmas = []
    learned_local_ls = [] 
    
    samples_processed = 0
    ignored_count = 0
    
    # Loop over Folds/Floats
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_scaled, groups=groups)):
        if max_samples and samples_processed >= max_samples: break
        
        # Subsample test set
        points_needed = max_samples - samples_processed if max_samples else len(test_idx)
        n_take = min(len(test_idx), points_needed)
        if n_take < len(test_idx):
             current_test_idx = np.random.choice(test_idx, size=n_take, replace=False)
        else:
             current_test_idx = test_idx

        # -----------------------------------------------------------
        # STRATEGY: GROUP OPTIMIZATION (Hybrid Mode)
        # -----------------------------------------------------------
        current_ls = start_length_scale
        current_noise = start_noise
        
        if optimization_mode == 'group':
            # 1. Pick up to 3 random representative points from this float/group
            sample_size = min(3, len(current_test_idx))
            sample_indices = np.random.choice(current_test_idx, size=sample_size, replace=False)
            
            group_ls_list = []
            group_noise_list = []
            
            # 2. Optimize on just these 3 points to "Learn the Float's Physics"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                
                for s_idx in sample_indices:
                    # Geometry
                    target_pt_rad = X_rad[s_idx].reshape(1, -1)
                    train_subset_rad = X_rad[train_idx]
                    dists = haversine_distances(train_subset_rad, target_pt_rad).flatten()
                    neighbor_mask = dists < radius_rad
                    valid_train_indices = train_idx[neighbor_mask]
                    
                    if len(valid_train_indices) < min_neighbors: continue
                    
                    # Local Bounds logic
                    avg_scale_km = np.mean(phys_scale_X) * 111.0
                    radius_scaled = radius_km / avg_scale_km
                    upper_bound = max(1.0, radius_scaled * 2.0)

                    # Optimize
                    X_loc = X_scaled[valid_train_indices]
                    y_loc = y_scaled[valid_train_indices]
                    k = ConstantKernel(1.0) * \
                        RBF(length_scale=start_length_scale, length_scale_bounds=(0.05, upper_bound)) + \
                        WhiteKernel(noise_level=start_noise)
                    
                    gp_opt = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0)
                    gp_opt.fit(X_loc, y_loc)
                    
                    group_ls_list.append(gp_opt.kernel_.k1.k2.length_scale)
                    group_noise_list.append(gp_opt.kernel_.k2.noise_level)
            
            # 3. Average them to get the parameters for this entire float
            if len(group_ls_list) > 0:
                current_ls = np.mean(group_ls_list, axis=0)
                current_noise = np.mean(group_noise_list)

        # -----------------------------------------------------------
        # PREDICTION LOOP (Window by Window)
        # -----------------------------------------------------------
        for t_idx in current_test_idx:
            samples_processed += 1
            
            # A. Neighbors
            target_pt_rad = X_rad[t_idx].reshape(1, -1)
            target_feature = X_scaled[t_idx].reshape(1, -1)
            train_subset_rad = X_rad[train_idx]
            dists = haversine_distances(train_subset_rad, target_pt_rad).flatten()
            
            neighbor_mask = dists < radius_rad
            valid_train_indices = train_idx[neighbor_mask]
            
            if len(valid_train_indices) < min_neighbors:
                ignored_count += 1
                continue 
            
            X_local = X_scaled[valid_train_indices]
            y_local = y_scaled[valid_train_indices]
            
            # B. Kernel Setup
            if optimization_mode == 'point':
                # Mode A: Re-Optimize every point (Scientific)
                optimizer_setting = 0
                bounds_setting = (0.05, max(1.0, (radius_km/(np.mean(phys_scale_X)*111.0))*2.0))
                # Use Global as start, but allow optimization
                ls_use = start_length_scale
                noise_use = start_noise
                
            else:
                # Mode B ('group') or C ('global'): Use FIXED parameters
                # We use the 'current_ls' we calculated above (either from group avg or global)
                optimizer_setting = None
                bounds_setting = "fixed"
                ls_use = current_ls
                noise_use = current_noise
            
            # C. Fit & Predict
            k = ConstantKernel(1.0, constant_value_bounds="fixed") * \
                RBF(length_scale=ls_use, length_scale_bounds=bounds_setting) + \
                WhiteKernel(noise_level=noise_use, noise_level_bounds=bounds_setting)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gp = GaussianProcessRegressor(kernel=k, optimizer=optimizer_setting, alpha=0.0)
                gp.fit(X_local, y_local)
            
            pred, std = gp.predict(target_feature, return_std=True)
            y_preds.append(pred[0])
            y_sigmas.append(std[0])
            y_true.append(y_scaled[t_idx])
            
            # Track what we used
            if optimization_mode == 'point':
                learned_local_ls.append(np.mean(gp.kernel_.k1.k2.length_scale))
            else:
                learned_local_ls.append(np.mean(current_ls))

        print(f"   Samples processed: {samples_processed}...", end='\r')

    # 4. SCORING
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    y_sigmas = np.array(y_sigmas)
    
    if len(y_preds) == 0: return np.array([])

    z_scores = (y_true - y_preds) / y_sigmas
    y_true_c = scaler_y.inverse_transform(y_true.reshape(-1,1)).flatten()
    y_pred_c = scaler_y.inverse_transform(y_preds.reshape(-1,1)).flatten()
    rmse = np.sqrt(np.mean((y_true_c - y_pred_c)**2))
    rms_rel_error = np.sqrt(np.mean(((y_pred_c - y_true_c) / (y_true_c + 1e-9))**2))
    
    print(f"\n✅ RESULTS ({method}):")
    print(f"   Avg Local LS:        {np.mean(learned_local_ls):.2f} (scaled)")
    print(f"   RMSE (Valid):        {rmse:.3f} °C")
    print(f"   Rel. Error (RMSRE):  {rms_rel_error:.4f}")
    print(f"   Mean Z:              {np.mean(z_scores):.3f}")
    print(f"   Std Z:               {np.std(z_scores):.3f}")
    
    return z_scores



    """
    PHASE 3: PRODUCTION MAPPER (The Generator).
    
    Generates a continuous Gridded Map (NetCDF/Xarray) from sparse Argo data
    using the "Fixed Kernel" parameters tuned in the Validation phase.
    
    -------------------------------------------------------------------------
    STRATEGY: "Integrate First, Map Second"
    -------------------------------------------------------------------------
    Instead of 4D Kriging (Lat, Lon, Depth, Time), we rely on the user passing
    pre-integrated layers (e.g., 'ohc_source'). This allows us to map each 
    physical layer with its own unique correlation length (e.g., Surface = Chaotic, 
    Deep = Smooth).
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Input data. MUST contain:
        - 'lat', 'lon' : Spatial coordinates (degrees).
        - 'time_days'  : Numeric time (e.g., days since start).
        - target_col   : The variable to interpolate (e.g., 'ohc_source').
        
    grid_lat, grid_lon : 1D arrays
        The spatial mesh you want to produce (e.g., np.arange(30, 40, 0.5)).
        
    grid_time : 1D array
        The time steps you want to produce (must match units of 'time_days').
        
    target_col : str
        The specific column in 'df' to map (e.g. 'ohc_response').
        
    radius_km : float
        The "Horizon" of the model. Points further than this are ignored 
        to save compute time. (Standard: ~300km).
        
    final_length_scale : float or array-like
        The physical correlation length (in SIGMAS) you found during validation.
        Can be a scalar (isotropic) or an array matching dimensions (anisotropic).
        
    final_noise : float
        The noise floor (uncertainty) you found during validation.
        (e.g., 0.1).
        
    is_3d : bool
        If True, includes 'time_days' in the distance calculation.
        
    time_buffer: int
        Number of days we are including in each time data point in our trend. Basically a time_buffer number of days
        will be combined to make a monthly map for us to study trends. This buffer smooths out our kriged fields. If
        you set this number too small, any time a float enters the window, there will be a sharp spike in the field
        and in the uncertainty. Too large and you're wasting computational power. We are expecting long correlation times
        for climate scale behavior, so the default is 60 days
    RETURNS:
    --------
    xr.Dataset
        A 3D Xarray (Time, Lat, Lon) containing:
        - 'temp' (Prediction)
        - 'uncertainty' (Standard Deviation / Error Bars)
    """

def produce_kriging_map(df, 
                        grid_lat, grid_lon, grid_time,
                        target_col='temp',
                        radius_km=300, min_neighbors=5,
                        final_length_scale=1.0, final_noise=0.1,
                        is_3d=True, time_buffer = 60):
   
    
    # ---------------------------------------------------------
    # 1. SETUP & SCALING
    # ---------------------------------------------------------
    # We must scale inputs so Lat (deg), Lon (deg), and Time (days) 
    # are treated equally by the isotropic kernel.
    feature_cols = ['lat', 'lon', 'time_days'] if is_3d else ['lat', 'lon']
    
    print(f"\n🗺️ STARTING PRODUCTION MAPPING for '{target_col}'...")
    if np.ndim(final_length_scale) == 0:
        print(f"   Using Kernel: Length={final_length_scale:.3f} (Sigmas), Noise={final_noise:.3f}")
    else:
        print(f"   Using Kernel: Length={final_length_scale} (Sigmas), Noise={final_noise:.3f}")
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit the scaler on ALL data to establish the global coordinate system
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Pre-calculate radians for Lat/Lon to use fast Haversine distance later
    # (Only needed for the spatial dimensions 0 and 1)
    X_rad_space = np.radians(X[:, :2]) 
    radius_rad = radius_km / 6371.0  # Convert km to Earth Radians
    
    # ---------------------------------------------------------
    # 2. DEFINE THE PHYSICS (THE KERNEL)
    # ---------------------------------------------------------
    # We use a FIXED kernel. We do NOT optimize (fit) inside the loop.
    # The parameters (length_scale, noise) are hard-coded from your Validation results.
    # optimizer=None ensures the model doesn't try to "re-learn" physics at every pixel.
    k = ConstantKernel(1.0, "fixed") * \
        RBF(length_scale=final_length_scale, length_scale_bounds="fixed") + \
        WhiteKernel(noise_level=final_noise, noise_level_bounds="fixed")
    
    # ---------------------------------------------------------
    # 3. PREPARE THE OUTPUT GRID
    # ---------------------------------------------------------
    shape = (len(grid_time), len(grid_lat), len(grid_lon))
    map_mean = np.full(shape, np.nan) # The Prediction Map
    map_std = np.full(shape, np.nan)  # The Uncertainty Map
    
    total_slices = len(grid_time)
    
    # ---------------------------------------------------------
    # 4. THE MAIN LOOP (Time -> Space)
    # ---------------------------------------------------------
    for t_i, t_val in enumerate(grid_time):
        
        # --- OPTIMIZATION A: TEMPORAL FILTER ---
        # Instead of searching the entire dataset for every pixel, 
        # we first grab only floats that exist near this time step.
        # This reduces the matrix size from N=10,000 to N=200, making it fast.
        
        if is_3d:
            # We filter for data within +/- 60 days (default) (2 months) of the target date.
            # Why 60? It's a safe buffer. The kernel (length_scale) will naturally 
            # downweight points far away in time, but we hard-cut them here for speed.
            time_col_idx = 2
            # Note: We must look at RAW time (X[:,2]) not scaled X_scaled yet
            time_mask = np.abs(X[:, time_col_idx] - t_val) < time_buffer
            
            # If no data exists in this window, skip the whole month (leave as NaNs)
            if np.sum(time_mask) < min_neighbors:
                print(f"   Skipping Slice {t_i+1}/{total_slices} (Not enough data)...", end='\r')
                continue
                
            X_subset = X_scaled[time_mask]
            y_subset = y_scaled[time_mask]
            X_rad_subset = X_rad_space[time_mask]
        else:
            # 2D Mode: Use all data (Climatology)
            X_subset = X_scaled
            y_subset = y_scaled
            X_rad_subset = X_rad_space

        # Loop through Space (Lat/Lon)
        for lat_i, lat_val in enumerate(grid_lat):
            for lon_i, lon_val in enumerate(grid_lon):
                
                # --- OPTIMIZATION B: SPATIAL FILTER ---
                # Use Haversine distance to find neighbors within radius_km
                target_rad = np.radians([[lat_val, lon_val]])
                dists = haversine_distances(X_rad_subset, target_rad).flatten()
                
                # Create the local neighborhood mask
                mask = dists < radius_rad
                
                # If this pixel is in a "Void" (no nearby floats), leave it as NaN.
                # This explicitly identifies the "Observationally Opaque" regions.
                if np.sum(mask) < min_neighbors: continue
                
                # --- THE KRIGING STEP ---
                # 1. Prepare Target Point (Lat, Lon, Time) scaled to global norms
                coords = [lat_val, lon_val, t_val] if is_3d else [lat_val, lon_val]
                target_scaled = scaler_X.transform([coords])
                
                # 2. Instantiate GP with FIXED Physics
                gp = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=0.0)
                
                # 3. Fit ONLY to the local neighborhood (Subset of Subset)
                gp.fit(X_subset[mask], y_subset[mask])
                
                # 4. Predict
                pred, std = gp.predict(target_scaled, return_std=True)
                
                # 5. Inverse Transform (Back to real units: Joules or Deg C)
                # Note: We must handle scaler shapes carefully
                map_mean[t_i, lat_i, lon_i] = scaler_y.inverse_transform([[pred[0]]])[0][0]
                
                # Uncertainty scales linearly, so we multiply by the scaler's scale factor
                map_std[t_i, lat_i, lon_i] = std[0] * scaler_y.scale_[0]
        
        # Progress Bar
        print(f"   Mapped Slice {t_i+1}/{total_slices} (Time={t_val:.0f})...", end='\r')

    print("\n✅ Mapping Complete.")
    
    # ---------------------------------------------------------
    # 5. EXPORT TO XARRAY
    # ---------------------------------------------------------
    # We package the 3D numpy arrays into a labelled Data Cube
    return xr.Dataset(
        data_vars={
            target_col: (["time", "lat", "lon"], map_mean),
            f"{target_col}_uncert": (["time", "lat", "lon"], map_std)
        },
        coords={
            "time": grid_time,
            "lat": grid_lat,
            "lon": grid_lon
        },
        attrs={
            "description": f"Kriged Map of {target_col}",
            "kernel_length_scale": final_length_scale,
            "kernel_noise": final_noise,
            "units": "Joules/m^2" if "ohc" in target_col else "degC"
        }
    )