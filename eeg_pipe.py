# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:32:04 2026

@author: lifep
"""

import numpy as np
import pandas as pd
# import os
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels
from autoreject import AutoReject
import traceback
import matplotlib.pyplot as plt


DATA_PATH = 'E:\\COGA_eec\\data\\'
WRITE_PATH = 'E:\\COGA_eec\\eeg_pipe\\'
notch_freqs = [60.0, 80.0]     # FREQUENCY (Hz) TO REMOVE LINE NOISE AND UNKNOWN NOISE FROM SIGNAL  
lowfrq = 1              # LOW PASS FREQUENCY, RECOMMENDED SETTING TO 1 HZ IF USING mne-icalabel
hifrq = 100             # HIGH PASS FREQUENCY
do_plot_channels = False  # TO GENERATE PLOTS OF THE CLEANED EEG SIGNAL
eye_blink_chans = ['X', 'Y'] # NAMES OF CHANNELS CONTAINING EOG
    
def extract_cnt_name(csv_name):
    """
    Converts 'FZ_eec_1_a1_10003051_cnt_256.csv' 
    to 'eec_1_a1_10003051.cnt'
    """
    if pd.isna(csv_name):
        return csv_name
    
    # Split on the first underscore only, keep the second half
    base_name = csv_name.split('_cnt_', 1)[0]
    # # TO DEAL WITH CNT FILENAMES CONTAINING TRAILING STRINGS NOT IN ORIGINAL FILENAME
    # # THE 8-DIGIT ID SHOULD BE THE LAST PART OF THE FILENAME
    # if len(base_name.split('_'))>4:      
    #     base_name = '_'.join(base_name.split('_')[:4])
    # Add the extension
    return ''.join([base_name,'.cnt'])

def rename_channels_eeglab_standard(raw):
    # Create a dictionary mapping the all-caps names to MNE's expected case
    rename_mapping = {
        'FPZ': 'Fpz',
        'OZ': 'Oz'
    }
    # Neuroscan files often use older 10-20 names (T3/T4) instead of the updated 10-10 names (T7/T8). 
    # If your files use T3/T4, you must rename those as well so MNE can find the lateral anchors.
    if 'T3' in raw.ch_names:
        rename_mapping.update({'T3': 'T7', 'T4': 'T8'})
    elif 'T7' in raw.ch_names or 'T7' in [ch.upper() for ch in raw.ch_names]:
        rename_mapping.update({'T7': 'T7', 'T8': 'T8'})
    # Apply the renaming
    raw.rename_channels(rename_mapping)
    # Plot the figure for your publication
    raw.plot_sensors(kind='topomap', sphere='eeglab')
    

def salvage_cnt_data(file_path, data_dtype='<i4'):
    """
    Reads the raw time-series data from a truncated Neuroscan .cnt file,
    bypassing the broken event table completely.
    """
    with open(file_path, 'rb') as f:
        # 1. Read the number of channels from the SETUP header.
        # In .cnt files, n_channels is stored as a 16-bit integer at byte offset 370.
        f.seek(370)
        # n_channels = np.fromfile(f, dtype='<i2', count=1)[0]
        n_channels = int(np.fromfile(f, dtype='<i2', count=1)[0])
        
        # 2. Calculate the start of the continuous data block.
        # The SETUP header is 900 bytes, followed by 75 bytes per channel for the ELECTLOC block.
        data_offset = 900 + (75 * n_channels)
        
        # 3. Jump directly to the data.
        f.seek(data_offset)
        
        # 4. Read the remainder of the file into a 1D array.
        raw_1d = np.fromfile(f, dtype=data_dtype)
        
        # 5. Handle incomplete samples caused by the unexpected EOF.
        # Data is multiplexed (Sample1_Ch1, Sample1_Ch2... Sample2_Ch1...)
        remainder = raw_1d.size % n_channels
        if remainder != 0:
            print(f"Warning: File truncated mid-sample. Discarding {remainder} trailing values.")
            raw_1d = raw_1d[:-remainder]
            
        # 6. Reshape into (n_samples, n_channels) and transpose to (n_channels, n_samples)
        # which is the standard layout for downstream EEG analysis.
        data_2d = raw_1d.reshape(-1, n_channels).T
        
    print(f"Successfully recovered {data_2d.shape[1]} samples across {n_channels} channels.")
    return data_2d

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the metadata dataframe

meta_df = pd.read_pickle(f"{WRITE_PATH}meta_data.pkl")
# meta_df = pd.read_pickle(r"E:\COGA_eec\pacdat_MASTER_fz.pkl")
# meta_df = pd.read_pickle(r'C:\Users\lifep\Documents\Data\pacdat_MASTER.pkl')

# DO SOME PRELIMINARY PROCESSING OF METADATA TO EXTRACT CORRECT .CNT FILENAMES

# meta_df = meta_df[pd.notna(meta_df.eeg_file_name)]
meta_df = meta_df[meta_df.data_format=='auto']
meta_df = meta_df.sort_values(['ID', 'age_this_visit'], ascending=[True, True]).reset_index(drop=True)


# Create a new column with the clean .cnt filenames
meta_df['cnt_file_name'] = meta_df['eeg_file_name'].apply(extract_cnt_name)
meta_df['sample_freq'] = meta_df['eeg_file_name'].str.split('_cnt_').str[1]

# print(meta_df[['eeg_file_name', 'cnt_file_name']].head())


# OPEN CNT FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
i = 4
cnt_file_name = meta_df.iloc[i].cnt_file_name
sample_freq = meta_df.iloc[i].sample_freq
site_name = meta_df.iloc[i].site.lower()
cnt_file_path = DATA_PATH + site_name + '\\' + cnt_file_name
print(f"Attempting to process: {meta_df.iloc[i].eeg_file_name}")


if sample_freq=='256':
    this_data_format = 'int16'
else:
    this_data_format = 'int32'
print(f"Data format = {this_data_format}, sample rate = {sample_freq}")

try:
    # 1. Probe the header first without loading the heavy data matrix
    raw_header = mne.io.read_raw_cnt(cnt_file_path, data_format=this_data_format, preload=False, verbose=True)
    
    # 2. Extract the embedded hardware information
    actual_sample_rate = raw_header.info['sfreq']
    channel_count = raw_header.info['nchan']
    
    print(f"Success: File contains {channel_count} channels sampled at {actual_sample_rate} Hz.")
    
    # 4. Now load the actual data safely into memory
    raw = raw_header.load_data()
    
except (RuntimeError, MemoryError, ValueError) as e:
    # If the file header is corrupted and triggers a byte/memory error, log it and skip
    print(f"Cannot extract info; header is corrupted for {cnt_file_name}.")
    print(f"Error caught: {e}")
    print(f"Target {salvage_cnt_data} for salvage_cnt_data()")
    
    # Fallback: Try extracting eeg data directly
    # Note: Newer Neuroscan files are 32-bit ('<i4'). If the array returns massive or noisy 
    # voltage values, the original file is likely 16-bit. If so, change data_dtype to '<i2'.
    # recovered_data = salvage_cnt_data(cnt_file_path)
    
    # In your final script, append this filename to a 'failed_files_log.txt' here
    with open(WRITE_PATH + 'errors_cnt_files.txt', 'a') as bf:
        bf.write(str(cnt_file_name) + '\t ' + str(e) + '\n')
        # bf.write(str(cnt_file_name) + '\n')

        
    # continue  <-- uncomment this when you put it in the actual for-loop



# MONTAGE SETUP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WE EXCLUDE THE BLANK CHANNEL AND RELABEL CHANNEL TYPES OF THE TWO EYE CHANNELS TO eog
# ASSUMES THAT ALL CHANNELS ARE LABELED AS EEG WHETHER THEY ARE OR NOT
# X = VEOG, Y = HEOG
for ch in eye_blink_chans:
    if ch in raw.ch_names:
        raw.set_channel_types({ch: 'eog'})
raw.drop_channels(['BLANK'], on_missing='warn')
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage, match_case=False)
# raw.compute_psd().plot()
# rename_channels_eeglab_standard(raw)
# raw.plot_sensors(kind='3d')

# IMPORT AND DOWNSAMPLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# downsampling to reduce processing time of downstream signal filtering
if raw.info['sfreq'] > 256.0:
    raw.resample(256.0)
# NOW WE PERFORM PREPROCESSING STEPS ON THE (COMPLETELY) RAW DATA FROM THE .CNT FILES
# LOW AND HIGH PASS FILTERING THAT SATISFIES ZERO-PHASE DESIGN
raw = raw.filter(lowfrq, hifrq)
# REMOVE 60 HZ LINE NOISE FROM SIGNAL WITH NOTCH FILTER
# raw.notch_filter(60, filter_length='auto', phase='zero', verbose=False)
raw.notch_filter(freqs=notch_freqs, filter_length='auto', phase='zero', verbose=False)

# SIGNAL QUALITY CHECKS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize the NoisyChannels object with your filtered data.
# Setting a random_state ensures the RANSAC spatial correlation step is mathematically reproducible.
nd = NoisyChannels(raw, random_state=42)

# Execute the full suite of PREP bad channel detection algorithms.
# This single command runs flatline detection, amplitude thresholding, and spatial correlation checks.
nd.find_all_bads()

# Retrieve the compiled list of outlier channels.
bad_channels = nd.get_bads()
print(f"\nPyPREP identified {len(bad_channels)} bad channels: {bad_channels}")
print(f"NaN: {nd.bad_by_nan}")
print(f"Flat: {nd.bad_by_flat}")
print(f"Extreme Amplitude Deviation: {nd.bad_by_deviation}")
print(f"High-Frequency Noise: {nd.bad_by_hf_noise}")
print(f"Low Spatial Correlation: {nd.bad_by_correlation}")
print(f"RANSAC (Spatial Predictability) Failure: {nd.bad_by_ransac}")

# Append these channels to MNE's bad channel list.
raw.info['bads'].extend(bad_channels)

# Retrieve a dictionary mapping every PREP evaluation test to the channels that failed it
bads_dict = nd.get_bads(as_dict=True)

# Loop through the dictionary and print only the tests that actually flagged channels
for test_name, failed_channels in bads_dict.items():
    if failed_channels:
        print(f"{test_name}: {failed_channels}")

if do_plot_channels:    
    raw.compute_psd().plot()


# RE-REFERENCING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # WE NEED TO APPLY A COMMON AVERAGE REFERENCE TO USE MNE-ICALabel         
raw = raw.set_eeg_reference("average")


# ARTIFACT REMOVAL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ica = ICA(
    n_components=15,
    max_iter="auto",
    random_state=42,
    method="infomax",
    fit_params=dict(extended=True),
    verbose=False
)
ica.fit(raw, picks='eeg')
eog_indices, eog_scores = ica.find_bads_eog(raw)
if eog_indices:
    print(f"Ground-truth validation flagged eye components: {eog_indices}")
    ica.exclude.extend(eog_indices)
    ica.plot_scores(eog_scores)
    ica.plot_components(picks=eog_indices) #, sphere='eeglab')
    ica.plot_properties(raw, picks=eog_indices)
else:
    print("No strong EOG correlations found.")
    # ica.plot_components()

    
# WE COMBINE THE NON-BRAIN ICs FROM BOTH eog_indices AND exclude_idx TO 
# COVER ALL POSSIBLE NON-BRAIN ARTIFACTS FOR REMOVAL
ic_labels = label_components(raw, ica, method="iclabel")
probabilities = ic_labels['y_pred_proba']
labels = ic_labels["labels"]
# THEN EXCLUDE ANY ICs THAT ARE NOT CLASSIFIED AS 'BRAIN' OR 'OTHER'
exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
# AND FINALLY WE RECONSTRUCT THE SIGNAL USING THE INCLUDED ICs
# COMBINING ALL NON-BRAIN ICs AND REMOVING THEM
ic_to_remove = [*set(exclude_idx)]
ica.exclude = ic_to_remove
raw_clean = ica.apply(raw)


# PSEUDO-EPOCHING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Pseudo-Epoching: Slice the continuous clean data into fixed-length epochs
# We are using 4.0 seconds, but 2 to 4 seconds is standard for resting state
epochs = mne.make_fixed_length_epochs(raw_clean, duration=10, preload=True)
# THEN REMOVING PREDICTED BAD FOR UserWarning: 
# channels are marked as bad. These will be ignored. 
# If you want them to be considered by autoreject please remove them from epochs.info["bads"]
epochs.info["bads"] = []

# 2. Initialize AutoReject
# This automatically computes thresholds to drop bad epochs
ar = AutoReject(random_state=42, picks='eeg')

# 3. Fit the model and apply the rejection
epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

print(f"Dropped {reject_log.bad_epochs.sum()} bad epochs out of {len(epochs)}.")

# 10. Save your cleaned epochs here before the loop restarts
epochs_clean.save(WRITE_PATH + cnt_file_name.replace('.cnt', '-epo.fif'), overwrite=True)

if do_plot_channels:
    # # DEBUG
    # file_path = r"E:\COGA_eec\eeg_pipe\eec_4_b1_10003053_32-epo.fif"
    # epochs_clean = mne.read_epochs(file_path, preload=True)
    
    # 1. Interactive Time-Series Epoch Browser
    # Allows scrolling across epochs/channels, scaling amplitude with +/- keys
    # epochs_clean.plot(scalings=dict(eeg=40e-6), n_epochs=5, n_channels=30)
    
    # 2. Power Spectral Density (PSD)
    # Inspect the frequency spectrum to verify 1 Hz drift removal, 60 Hz notch filtering, and alpha peaks
    epochs_clean.compute_psd(fmin=1, fmax=100).plot()
    
    # 3. Sensor Heatmap / Epochs Image
    # Shows amplitude across all trials for specific regions (e.g., occipital alpha or frontal)
    epochs_clean.plot_image(picks=['OZ', 'O1', 'O2'])
    
    # 4. Global Field Power (GFP) & Evoked Butterfly Plot
    # epochs_clean.average().plot(spatial_colors=True)
    plt.show()
    
    plt.close('all') # Prevent memory leaks from open figures

    

