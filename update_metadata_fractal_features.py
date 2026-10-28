# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:10:19 2026

@author: lifep
"""

import mne
import numpy as np
import pandas as pd
import neurokit2 as nk


DATA_PATH = 'E:\\COGA_eec\\data\\'
WRITE_PATH = 'E:\\COGA_eec\\eeg_pipe\\'



def extract_cnt_name(csv_name):
    if pd.isna(csv_name):
        return csv_name
    base_name = csv_name.split('_cnt_', 1)[0]
    return ''.join([base_name,'.cnt'])



def extract_fractal_features(df, file_col='eeg_file_name'):
    # Initialize columns for our aggregated metrics
    metrics = {
        'hurst_mean': [],
        'mfdfa_width_mean': [],
        'higuchi_fd_mean': [],
        'hjorth_mob_mean': [],
        'hjorth_comp_mean': []
    }
    
    for _, row in df.iterrows():
        # 1. Load the EEG file (handles .cnt, .set, .vhdr, etc. based on extension)
        site_name = row.site.lower()
        path_file = DATA_PATH + site_name + '\\' + row[file_col] + '.cnt'
        raw = mne.io.read_raw(path_file, data_format='auto', preload=True, verbose=False)
        
        # Optional: Filter the signal to a standard resting-state bandwidth (e.g., 0.5 - 45 Hz)
        # raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
        
        # 2. Crop to the first 5 minutes (300 seconds)
        # If a recording is shorter than 5 minutes, crop to its maximum length
        tmax = min(300.0, raw.times[-1])
        raw.crop(tmin=0, tmax=tmax)
        
        # 3. Create 10-second non-overlapping epochs
        epochs = mne.make_fixed_length_epochs(raw, duration=10.0, preload=True, verbose=False)
        
        # Get data as a NumPy array: shape (n_epochs, n_channels, n_times)
        # Note: We compute non-linear features *per channel* first, then average. 
        # Averaging raw signals across channels before computing fractal dimensions 
        # destroys scale-free dynamics due to phase cancellation.
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        # Temporary arrays to hold epoch-level results
        ep_hurst = np.zeros((n_epochs, n_channels))
        ep_width = np.zeros((n_epochs, n_channels))
        ep_hfd = np.zeros((n_epochs, n_channels))
        ep_hmob = np.zeros((n_epochs, n_channels))
        ep_hcomp = np.zeros((n_epochs, n_channels))
        
        for ep_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                signal = data[ep_idx, ch_idx, :]
                
                # A. Monofractal Hurst Exponent via Detrended Fluctuation Analysis
                # nk.fractal_dfa returns the exponent and an info dictionary
                hurst, _ = nk.fractal_dfa(signal, multifractal=False)
                
                # B. Multifractal Spectrum Width via MF-DFA
                # Sweeping across q-moments to evaluate the scaling exponent spectrum
                _, mf_info = nk.fractal_dfa(signal, multifractal=True, q=np.arange(-5, 6))
                
                # Extract singularity exponents (alpha or h) to find the spectrum width (W)
                # NK2 returns these in the info dict; width = max(h) - min(h)
                exponents = mf_info.get('alpha', mf_info.get('h', [0]))
                if len(exponents) > 1:
                    width = np.max(exponents) - np.min(exponents)
                else:
                    width = np.nan
                    
                # C. Higuchi Fractal Dimension
                hfd, _ = nk.fractal_higuchi(signal)
                
                # D. Hjorth Parameters
                # Note: If your NeuroKit2 version doesn't map these directly, 
                # mobility = std(diff(signal)) / std(signal)
                # complexity = mobility(diff(signal)) / mobility(signal)
                try:
                    hjorth, _ = nk.complexity_hjorth(signal)
                    mob, comp = hjorth['Hjorth_Mobility'], hjorth['Hjorth_Complexity']
                except AttributeError:
                    # Fallback manual calculation if NK2 version differs
                    deriv1 = np.diff(signal)
                    deriv2 = np.diff(deriv1)
                    mob = np.std(deriv1) / np.std(signal)
                    comp = (np.std(deriv2) / np.std(deriv1)) / mob
                
                # Store features for this specific channel and epoch
                ep_hurst[ep_idx, ch_idx] = hurst
                ep_width[ep_idx, ch_idx] = width
                ep_hfd[ep_idx, ch_idx] = hfd
                ep_hmob[ep_idx, ch_idx] = mob
                ep_hcomp[ep_idx, ch_idx] = comp
        
        # Average across all channels and all 10-second epochs to get a single subject-level feature vector
        # (Alternatively, you can extract ROI-specific columns, e.g., 'hurst_frontal', 'hurst_parietal')
        metrics['hurst_mean'].append(np.nanmean(ep_hurst))
        metrics['mfdfa_width_mean'].append(np.nanmean(ep_width))
        metrics['higuchi_fd_mean'].append(np.nanmean(ep_hfd))
        metrics['hjorth_mob_mean'].append(np.nanmean(ep_hmob))
        metrics['hjorth_comp_mean'].append(np.nanmean(ep_hcomp))
        
    # Map the calculated metrics back to the original DataFrame
    for col_name, values in metrics.items():
        df[col_name] = values
        
    return df


# PRELIMINARY PROCESSING OF METADATA
# meta_df = pd.read_pickle(r'C:\Users\lifep\Documents\Data\pacdat_MASTER.pkl')
meta_df = pd.read_pickle(f'{WRITE_PATH}meta_data.pkl')
meta_df = meta_df[pd.notna(meta_df.eeg_file_name)]
meta_df = meta_df.sort_values(['ID', 'age_this_visit'], ascending=[True, True]).reset_index(drop=True)
meta_df['cnt_file_name'] = meta_df['eeg_file_name'].apply(extract_cnt_name)
meta_df['sample_freq'] = meta_df['eeg_file_name'].str.split('_cnt_').str[1]
meta_df['duration_seconds'] = 0
meta_df['right_handed'] = 999
meta_df['data_format'] = ''

ndf = extract_fractal_features(meta_df)
meta_df.to_pickle(WRITE_PATH + 'meta_data.pkl')
