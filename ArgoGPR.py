import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics.pairwise import haversine_distances


def generalized_cross_validation(df, feature_cols=['lat', 'lon'], target_col='temp', 
                                 method='KFold', k_fold_data_percent=10,
                                 auto_tune=True, tune_subsample_frac=0.05,
                                 length_scale_val=1.0, noise_val=0.1):
    """
    Global GP Validation (Smart Hybrid).
    1. Learns hyperparameters by averaging 5 optimization runs on small subsets.
    2. Applies those fixed parameters to the full cross-validation (Fast).
    
    ---------------------------------------------------------------------------
    PARAMETERS:
    ---------------------------------------------------------------------------
    1. df (pd.DataFrame): 
       The master table. Must contain feature_cols, target_col, and 'float_id'.

    2. feature_cols (list of str): 
       Dimensions for similarity (e.g., ['lat', 'lon'] or ['lat', 'lon', 'time_days']).
       
    3. target_col (str): 
       Variable to predict (e.g., 'temp').

    4. method (str):
       - 'KFold': Tests interpolation accuracy (Random hold-out).
       - 'LOFO':  Tests scientific rigor (Entire instrument hold-out).

    5. k_fold_data_percent (float): 
       Percentage of data to hold out for TESTING in each fold (e.g., 10%).

    6. auto_tune (bool): 
       - True:  Runs the optimizer on subsets to LEARN the best length/noise.
                Ignores the manual knobs below. 
       - False: Uses the manual length_scale_val and noise_val.

    7. tune_subsample_frac (float): 
       Fraction of data to use for EACH auto-tune iteration (0.0 to 1.0).
       e.g., 0.05 uses 5% of data. The code runs 5 iterations total.

    8. length_scale_val (float): 
       MANUAL knob. Used only if auto_tune=False.
       The "Smoothness" of the function (in standardized units).
       
    9. noise_val (float): 
       MANUAL knob. Used only if auto_tune=False.
       The "Distrust" level.
    ---------------------------------------------------------------------------
    """
    print(f"\n🚀 STARTING GLOBAL VALIDATION: {method}")
    
    # 1. SETUP
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['float_id'].values 
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # ---------------------------------------------------------
    # 2. ROBUST AUTO-TUNE STEP (Average of 5 Runs)
    # ---------------------------------------------------------
    final_length_scale = length_scale_val
    final_noise = noise_val
    
    if auto_tune:
        N_points = len(X_scaled)
        n_sub = int(N_points * tune_subsample_frac)
        # Ensure we have at least 100 points to train, but not too many (<2000) for speed
        n_sub = max(100, min(n_sub, 2000))
        
        print(f"   🤖 Auto-Tuning: Running 5 iterations on {n_sub} points ({tune_subsample_frac*100:.1f}%)...")
        
        learned_ls = []
        learned_noise = []
        
        for run in range(5):
            # Random subset
            idx_tune = np.random.choice(N_points, n_sub, replace=False)
            X_tune, y_tune = X_scaled[idx_tune], y_scaled[idx_tune]
            
            # Optimization
            k_tune = ConstantKernel(1.0) * RBF(length_scale=[1.0]*X.shape[1]) + WhiteKernel(noise_level=0.1)
            gp_tune = GaussianProcessRegressor(kernel=k_tune, n_restarts_optimizer=0) # Speed up by removing restarts
            gp_tune.fit(X_tune, y_tune)
            
            learned_ls.append(gp_tune.kernel_.k1.k2.length_scale)
            learned_noise.append(gp_tune.kernel_.k2.noise_level)
            
        # Average the results
        # Note: If length_scale is a list (anisotropic), we average axis-wise
        final_length_scale = np.mean(learned_ls, axis=0) 
        final_noise = np.mean(learned_noise)
        
        print(f"      ✅ Averaged Length Scale: {final_length_scale}")
        print(f"      ✅ Averaged Noise Level:  {final_noise:.4f}")
    else:
        print(f"   🔧 Using Manual Parameters: Length={final_length_scale}, Noise={final_noise}")

    # 3. CHOOSE SPLITTER
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
    elif method == 'KFold':
        n_splits = int(100 / k_fold_data_percent)
        if n_splits < 2: n_splits = 2
        if n_splits >= len(df): n_splits = len(df)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"   ⚡ Strategy: {n_splits}-Fold CV (Testing {k_fold_data_percent}% per fold)")

    # 4. RUN LOOP (Using FIXED Parameters)
    y_preds = np.zeros_like(y)
    y_sigmas = np.zeros_like(y)
    tested_mask = np.zeros(len(y), dtype=bool)
    MAX_TRAIN = 2000 
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_scaled, groups=groups)):
        
        # Subsample training if massive
        if len(train_idx) > MAX_TRAIN:
            train_subset = np.random.choice(train_idx, size=MAX_TRAIN, replace=False)
        else:
            train_subset = train_idx
            
        X_train = X_scaled[train_subset]
        y_train = y_scaled[train_subset]
        X_test = X_scaled[test_idx]
        
        # --- FIXED PARAMETER KERNEL ---
        k = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            RBF(length_scale=final_length_scale, length_scale_bounds="fixed") + \
            WhiteKernel(noise_level=final_noise, noise_level_bounds="fixed")
        
        gp = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=0.0)
        gp.fit(X_train, y_train)
        
        pred, std = gp.predict(X_test, return_std=True)
        
        y_preds[test_idx] = pred
        y_sigmas[test_idx] = std
        tested_mask[test_idx] = True
        
        if cv.get_n_splits() > 10 and i % (cv.get_n_splits() // 10) == 0:
             print(f"   Processed fold {i+1}...", end='\r')

    # 5. SCORING
    y_true_valid = y_scaled[tested_mask]
    y_pred_valid = y_preds[tested_mask]
    y_sigma_valid = y_sigmas[tested_mask]
    
    if len(y_true_valid) == 0:
        return np.array([])

    z_scores = (y_true_valid - y_pred_valid) / y_sigma_valid
    
    y_true_c = scaler_y.inverse_transform(y_true_valid.reshape(-1,1)).flatten()
    y_pred_c = scaler_y.inverse_transform(y_pred_valid.reshape(-1,1)).flatten()
    rmse = np.sqrt(np.mean((y_true_c - y_pred_c)**2))
    
    print(f"\n✅ RESULTS ({method}):")
    print(f"   RMSE:   {rmse:.3f} °C")
    print(f"   Mean Z: {np.mean(z_scores):.3f}")
    print(f"   Std Z:  {np.std(z_scores):.3f} (Ideal: 1.0)")
    
    return z_scores



def validate_moving_window(df, feature_cols=['lat', 'lon'], target_col='temp', 
                           method='KFold', k_fold_data_percent=10,
                           radius_km=300, min_neighbors=5, max_samples=1000,
                           auto_tune=True, tune_subsample_frac=0.05,
                           length_scale_val=1.0, noise_val=0.1):
    """
    Moving Window (Local GP) Validation (Smart Hybrid). The moving window concept
    assumes that the length scales of the correlation (or time scales ) vary in space.
    So it recalculates these values for every float while performing Kfolding or LOFO.
    
    1. Learns hyperparameters by averaging 5 optimization runs on small subsets.
    2. Applies those fixed parameters to every local window. 
    
    ---------------------------------------------------------------------------
    PARAMETERS:
    ---------------------------------------------------------------------------
    1. df (pd.DataFrame): 
       The master table. Must contain feature_cols, target_col, and 'float_id'.

    2. feature_cols (list): 
       Dimensions for similarity (e.g., ['lat', 'lon']).
       
    3. target_col (str): 
       Variable to predict.

    4. method (str):
       - 'KFold': Random sampling.
       - 'LOFO':  Entire Float hold-out.

    5. k_fold_data_percent (float): 
       Percentage of data to test per fold. (e.g., 10 = 10% test set).

    6. radius_km (float): 
       The "Horizon". Only neighbors within this distance are used for training.

    7. min_neighbors (int): 
       Void check. If fewer than this many neighbors exist, skip prediction.

    8. max_samples (int): 
       Speed Limit. Stops validation after testing this many total points.

    9. auto_tune (bool): 
       - True: Runs the optimizer on subsets to LEARN best length/noise.
       - False: Uses manual knobs.

    10. tune_subsample_frac (float): 
        Fraction of data to use for EACH auto-tune iteration (0.0 to 1.0).
        e.g., 0.05 uses 5% of data. The code runs 5 iterations total.

    11. length_scale_val (float): 
        MANUAL knob. Used only if auto_tune=False.
        
    12. noise_val (float): 
        MANUAL knob. Used only if auto_tune=False.
    ---------------------------------------------------------------------------
    """
    print(f"\n🚀 STARTING MOVING WINDOW VALIDATION: {method}")
    print(f"   Config: Radius={radius_km}km | Max Samples={max_samples}")
    
    # 1. DATA PREP
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['float_id'].values 
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_rad = np.radians(X[:, :2]) 
    EARTH_RADIUS_KM = 6371.0
    radius_rad = radius_km / EARTH_RADIUS_KM

    # ---------------------------------------------------------
    # 2. ROBUST AUTO-TUNE STEP (Average of 5 Runs)
    # ---------------------------------------------------------
    final_length_scale = length_scale_val
    final_noise = noise_val
    
    if auto_tune:
        N_points = len(X_scaled)
        n_sub = int(N_points * tune_subsample_frac)
        n_sub = max(100, min(n_sub, 2000))
        
        print(f"   🤖 Auto-Tuning: Running 5 iterations on {n_sub} points ({tune_subsample_frac*100:.1f}%)...")
        
        learned_ls = []
        learned_noise = []
        
        for run in range(5):
            idx_tune = np.random.choice(N_points, n_sub, replace=False)
            X_tune, y_tune = X_scaled[idx_tune], y_scaled[idx_tune]
            
            k_tune = ConstantKernel(1.0) * RBF(length_scale=[1.0]*X.shape[1]) + WhiteKernel(noise_level=0.1)
            gp_tune = GaussianProcessRegressor(kernel=k_tune, n_restarts_optimizer=0)
            gp_tune.fit(X_tune, y_tune)
            
            learned_ls.append(gp_tune.kernel_.k1.k2.length_scale)
            learned_noise.append(gp_tune.kernel_.k2.noise_level)
            
        final_length_scale = np.mean(learned_ls, axis=0) 
        final_noise = np.mean(learned_noise)
        
        print(f"      ✅ Averaged Length Scale: {final_length_scale}")
        print(f"      ✅ Averaged Noise Level:  {final_noise:.4f}")
    else:
        print(f"   🔧 Using Manual Parameters: Length={final_length_scale}, Noise={final_noise}")

    # 3. CHOOSE SPLITTER & RUN LOOP
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
    elif method == 'KFold':
        n_splits = int(100 / k_fold_data_percent)
        if n_splits < 2: n_splits = 2
        if n_splits >= len(df): n_splits = len(df)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_preds = []
    y_true = []
    y_sigmas = []
    
    samples_processed = 0
    ignored_count = 0
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_scaled, groups=groups)):
        if max_samples and samples_processed >= max_samples: break
        
        # Subsample test set for speed
        points_needed = max_samples - samples_processed if max_samples else len(test_idx)
        n_take = min(len(test_idx), points_needed)
        if n_take < len(test_idx):
             current_test_idx = np.random.choice(test_idx, size=n_take, replace=False)
        else:
             current_test_idx = test_idx

        for t_idx in current_test_idx:
            samples_processed += 1
            
            # Neighbors
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
            
            # --- USE FIXED PARAMETERS ---
            k = ConstantKernel(1.0, constant_value_bounds="fixed") * \
                RBF(length_scale=final_length_scale, length_scale_bounds="fixed") + \
                WhiteKernel(noise_level=final_noise, noise_level_bounds="fixed")
            
            gp = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=0.0)
            gp.fit(X_local, y_local)
            
            pred, std = gp.predict(target_feature, return_std=True)
            
            y_preds.append(pred[0])
            y_sigmas.append(std[0])
            y_true.append(y_scaled[t_idx])

        print(f"   Samples processed: {samples_processed}...", end='\r')

    # 4. SCORING
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    y_sigmas = np.array(y_sigmas)
    
    if len(y_preds) > 0:
        z_scores = (y_true - y_preds) / y_sigmas
        
        y_true_c = scaler_y.inverse_transform(y_true.reshape(-1,1)).flatten()
        y_pred_c = scaler_y.inverse_transform(y_preds.reshape(-1,1)).flatten()
        rmse = np.sqrt(np.mean((y_true_c - y_pred_c)**2))
        
        print(f"\n✅ RESULTS ({method}):")
        print(f"   Points Tested:    {len(y_preds)}")
        print(f"   Voids/Ignored:    {ignored_count}")
        print("-" * 30)
        print(f"   RMSE (Valid):     {rmse:.3f} °C")
        print(f"   Mean Z:           {np.mean(z_scores):.3f}")
        print(f"   Std Z:            {np.std(z_scores):.3f} (Ideal: 1.0)")
        
        return z_scores
    else:
        print("\n❌ NO VALID PREDICTIONS.")
        return np.array([])