# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:31:23 2026

@author: lifep
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import mne_connectivity
from mne_connectivity import spectral_connectivity_epochs

# 1. Load continuous .cnt file (retaining original sampling rate)
# fif_file = 'E:\\COGA_eec\\eeg_pipe\\eec_3_c1_10016003_32_eeg.fif'
# fif_file = 'E:\\COGA_eec\\eeg_pipe\\eec_3_a1_10016003_32_eeg.fif'
fif_file = 'E:\\COGA_eec\\eeg_pipe\\eec_4_e1_10228024_32_eeg.fif'
# raw = mne.io.read_raw_cnt(cnt_file, preload=True)

if 'raw' not in locals():
    raw = mne.io.read_raw_fif(fif_file, preload=True)
else:
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    
raw.drop_channels(['X', 'Y'])

# Re-reference (average reference helps minimize spurious coherence)
if not raw.info['custom_ref_applied']:
    raw.set_eeg_reference('average')

# 2. Filter to a specific band (e.g., Alpha: 8-12 Hz)
# Phase coherence is highly frequency-dependent; broadband phase is usually uninformative
fmin, fmax = 8.0, 12.0
raw.filter(l_freq=fmin, h_freq=fmax)

# 3. Create fixed-length dummy epochs for resting-state continuous data
# 2-second epochs with 0% overlap is standard for basic resting state
epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)

# fif_path = r"E:\COGA_eec\eeg_pipe\eec_1_a1_10003051-epo.fif"
# fif_path = 'E:\\COGA_eec\\eeg_pipe\\eec_3_c1_10016003_32-epo.fif'
# fif_path = WRITE_PATH + cnt_file_name.replace('.cnt', '-epo.fif')
# epochs = mne.read_epochs(fif_path, preload=True)

# 4. Calculate all-to-all connectivity
# Options: 'wpli2_debiased', 'imcoh', 'plv', 'pli'
method = 'wpli2_debiased' 

con = spectral_connectivity_epochs(
    epochs,
    method=method,
    sfreq=raw.info['sfreq'],
    mode='multitaper',
    fmin=fmin,
    fmax=fmax,
    faverage=True,  # Averages the connectivity values across the 8-12 Hz band
    n_jobs=-1       # Utilize all available cores
)

# 5. Extract the NxN connectivity matrix
# output='dense' automatically reshapes the 1D connection array into a symmetric NxN matrix
con_matrix = con.get_data(output='dense').squeeze()
ch_names = epochs.info['ch_names']

# 6. Generate the heatmap
plt.figure(figsize=(14, 12))
ax = plt.subplot(111)

sns.heatmap(
    con_matrix,
    cmap='viridis',
    xticklabels=ch_names,
    yticklabels=ch_names,
    ax=ax,
    square=True,
    cbar_kws={'label': method.upper()}
)

plt.title(f'Resting State Phase Coherence ({method.upper()}) - {fmin}-{fmax} Hz')
plt.tight_layout()
plt.show()