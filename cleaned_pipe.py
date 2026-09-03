# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:32:04 2026
@author: lifep
"""

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels
from autoreject import AutoReject
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = 'E:\\COGA_eec\\data\\'
WRITE_PATH = 'E:\\COGA_eec\\eeg_pipe\\'

# Load and sort your metadata dataframe
meta_df = pd.read_pickle(r'C:\Users\lifep\Documents\Data\pacdat_MASTER.pkl')
meta_df = meta_df[pd.notna(meta_df.eeg_file_name)]
meta_df = meta_df.sort_values(['ID', 'age_this_visit'], ascending=[True, True]).reset_index(drop=True)

notch_freq = 60.0       # FREQUENCY (Hz) TO REMOVE LINE NOISE FROM SIGNAL 
lowfrq = 1              # LOW PASS FREQUENCY, RECOMMENDED SETTING TO 1 HZ IF USING mne-icalabel
hifrq = None             # HIGH PASS FREQUENCY
do_plot_channels = False # SET TO FALSE FOR THE FULL BATCH LOOP TO PREVENT MEMORY CRASHES
eye_blink_chans = ['X', 'Y'] # NAMES OF CHANNELS CONTAINING EOG

def extract_cnt_name(csv_name):
    if pd.isna(csv_name):
        return csv_name
    base_name = csv_name.split('_cnt_', 1)[0]
    return ''.join([base_name,'.cnt'])

# PRELIMINARY PROCESSING OF METADATA
meta_df['cnt_file_name'] = meta_df['eeg_file_name'].apply(extract_cnt_name)
meta_df['sample_freq'] = meta_df['eeg_file_name'].str.split('_cnt_').str[1]
meta_df['duration_seconds'] = 0
meta_df['right_handed'] = 999
meta_df['data_format'] = ''


total_good = 0

# BATCH LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Iterate through every row in your metadata dataframe
for i in range(len(meta_df)):
# for i in range(11):

    # 1. Pre-Flight Parameter Extraction
    # Query your pandas dataframe to extract parameters before touching the file[cite: 5]
    cnt_file_name = meta_df.iloc[i].cnt_file_name
    sample_freq = meta_df.iloc[i].sample_freq
    site_name = meta_df.iloc[i].site.lower()
    cnt_file_path = DATA_PATH + site_name + '\\' + cnt_file_name
    
    print(f"\n[{i+1}/{len(meta_df)}] Attempting to process: {meta_df.iloc[i].eeg_file_name}")


    if sample_freq == '256':
        this_data_format = 'int16'
    else:
        this_data_format = 'int32'
        
    print(f"Expected Data format = {this_data_format}, sample rate = {sample_freq}")

    # 2. Safely Open the .cnt File
    # TRY 'auto' FOR DATA FORMAT FIRST THEN TRY OTHER OPTION IF IT DOESN'T WORK
    try:
        raw_header = mne.io.read_raw_cnt(cnt_file_path, data_format='auto', preload=False, verbose=False)
        actual_sample_rate = raw_header.info['sfreq']
        channel_count = raw_header.info['nchan']
        print(f"Success: File contains {channel_count} channels sampled at {actual_sample_rate} Hz.")
        
        data = raw_header.load_data()
        # raw = data.copy()
        # LET'S GET THE DURATION OF THIS SIGNAL, RIGHT HANDEDNESS, AND DATA FORMAT TO OPEN
        meta_df.loc[i,'duration_seconds'] = len(data.get_data(['CZ'])[0])/data.info['sfreq']
        if 'hand' in data.info['subject_info'].keys():
            meta_df.loc[i,'right_handed'] = data.info['subject_info']['hand']==1
            
        meta_df.loc[i,'data_format'] = 'auto'
        total_good+=1
        
    except (RuntimeError, MemoryError, ValueError) as e:
        # Catch corrupted files, log them, and skip to the next iteration
        print(f"Skipping {cnt_file_name}: Header may be corrupted - trying to open with {this_data_format}.")
    
        with open(WRITE_PATH + 'errors_cnt_files.txt', 'a') as bf:
            bf.write(str(cnt_file_name) + '\n')
            # bf.write(str(cnt_file_name) + '\t ' + str(e) + '\n')
        meta_df.loc[i,'data_format'] = 'unknown'
        continue # Immediately move to the next file in the loop

            
        # try:
        #     raw_header = mne.io.read_raw_cnt(cnt_file_path, data_format=this_data_format, preload=False, verbose=False)
        #     actual_sample_rate = raw_header.info['sfreq']
        #     channel_count = raw_header.info['nchan']
        #     print(f"Success: File contains {channel_count} channels sampled at {actual_sample_rate} Hz.")
        #     data = raw_header.load_data()
        #     # raw = data.copy()
        #     # LET'S GET THE DURATION OF THIS SIGNAL, RIGHT HANDEDNESS, AND DATA FORMAT TO OPEN
        #     meta_df.iloc[i].duration_seconds = len(data.get_data(['CZ'])[0])/data.info['sfreq']
        #     if 'hand' in data.info['subject_info'].keys():
        #         meta_df.iloc[i].right_handed = data.info['subject_info']['hand']==1
        #     meta_df.iloc[i].data_format = this_data_format
        #     total_good+=1
        # except (RuntimeError, MemoryError, ValueError) as e:
        #     # Catch corrupted files, log them, and skip to the next iteration[cite: 5]
        #     print(f"Skipping {cnt_file_name}: Header is mathematically corrupted.")
        #     with open(WRITE_PATH + 'errors_cnt_files.txt', 'a') as bf:
        #         bf.write(str(cnt_file_name) + '\n')
        #         # bf.write(str(cnt_file_name) + '\t ' + str(e) + '\n')
        #     meta_df.iloc[i].data_format = 'unknown'
        #     continue # Immediately move to the next file in the loop
            
            
            
print(f"\nTotal good CNT files = {total_good} out of {len(meta_df)}")

meta_df.to_pickle(WRITE_PATH + 'meta_data.pkl')

    # # 3. Montage Setup
    # for ch in eye_blink_chans:
    #     if ch in raw.ch_names:
    #         raw.set_channel_types({ch: 'eog'})
    # raw.drop_channels(['BLANK'], on_missing='warn')
    # montage = mne.channels.make_standard_montage('standard_1005')
    # raw.set_montage(montage, match_case=False)
    
    # # 4. Import & Downsample
    # if raw.info['sfreq'] > 256.0:
    #     raw.resample(256.0)
        
    # # 5. Filtering
    # raw = raw.filter(lowfrq, hifrq)
    # raw.notch_filter(notch_freq, filter_length='auto', phase='zero', verbose=False)

    # # 6. Signal Quality Checks (PyPREP)
    # nd = NoisyChannels(raw, random_state=42)
    # nd.find_all_bads()
    # bad_channels = nd.get_bads()
    # print(f"PyPREP identified {len(bad_channels)} bad channels: {bad_channels}")
    # raw.info['bads'].extend(bad_channels)

    # if do_plot_channels:
    #     raw.compute_psd().plot()
    #     plt.show()

    # # 7. Re-Referencing
    # raw = raw.set_eeg_reference("average")

    # # 8. Artifact Removal (ICA)
    # ica = ICA(n_components=15, max_iter="auto", random_state=42, method="infomax", fit_params=dict(extended=True), verbose=False)
    # ica.fit(raw, picks='eeg')
    
    # eog_indices, eog_scores = ica.find_bads_eog(raw)
    # if eog_indices:
    #     print(f"Ground-truth validation flagged eye components: {eog_indices}")
    #     ica.exclude.extend(eog_indices)
    #     if do_plot_channels:
    #         ica.plot_scores(eog_scores)
    #         ica.plot_components(picks=eog_indices, sphere='eeglab')
    #         ica.plot_properties(raw, picks=eog_indices)
    #         plt.show()
    # else:
    #     print("No strong EOG correlations found. The subject likely did not blink.")

    # ic_labels = label_components(raw, ica, method="iclabel")
    # labels = ic_labels["labels"]
    
    # exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    # ica.exclude.extend(exclude_idx)
    # ica.exclude = list(set(ica.exclude)) # Remove duplicates
    
    # raw_clean = ica.apply(raw)

    # # 9. Pseudo-Epoching & AutoReject
    # epochs = mne.make_fixed_length_epochs(raw_clean, duration=4.0, preload=True)
    # ar = AutoReject(random_state=42, picks='eeg', verbose=False)
    # epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    # print(f"Dropped {reject_log.bad_epochs.sum()} bad epochs out of {len(epochs)}.")
    
    # # 10. Save your cleaned epochs here before the loop restarts
    # epochs_clean.save(WRITE_PATH + cnt_file_name.replace('.cnt', '-epo.fif'), overwrite=True)
    
    # if do_plot_channels:
    #     plt.close('all') # Prevent memory leaks from open figures