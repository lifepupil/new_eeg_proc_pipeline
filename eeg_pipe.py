# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:32:04 2026

@author: lifep
"""

import pandas as pd
import os
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import pyprep as pp

DATA_PATH = 'E:\\COGA_eec\\data\\'
# Load your metadata dataframe
# meta_df = pd.read_pickle(r"E:\COGA_eec\pacdat_MASTER_fz.pkl")
meta_df = pd.read_pickle(r'C:\Users\lifep\Documents\Data\pacdat_MASTER.pkl')

meta_df = meta_df.sort_values(['ID', 'age_this_visit'], ascending=[True, True]).reset_index(drop=True)
meta_df = meta_df[pd.notna(meta_df.eeg_file_name)]

notch_freq = 60.0       # FREQUENCY (Hz) TO REMOVE LINE NOISE FROM SIGNAL 
lowfrq = 1              # LOW PASS FREQUENCY, RECOMMENDED SETTING TO 1 HZ IF USING mne-icalabel
hifrq = 100             # HIGH PASS FREQUENCY
maxZeroPerc = 0.5       # PERCENTAGE OF ZEROS IN SIGNAL ABOVE WHICH CHANNEL IS LABELED 'BADS'
do_plot_channels = True # TO GENERATE PLOTS OF THE CLEANED EEG SIGNAL
# mpl.rcParams['figure.dpi'] = 300 # DETERMINES THE RESOLUTION OF THE EEG PLOTS
eye_blink_chans = ['X', 'Y'] # NAMES OF CHANNELS CONTAINING EOG
institutionDir = 'uconn' # suny, indiana, iowa, uconn, ucsd, washu
    
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
    
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# DO SOME PRELIMINARY PROCESSING OF METADATA TO EXTRACT CORRECT .CNT FILENAMES
# Create a new column with the clean .cnt filenames
meta_df['cnt_filename'] = meta_df['eeg_file_name'].apply(extract_cnt_name)
print(meta_df[['eeg_file_name', 'cnt_filename']].head())


# OPEN CNT FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
i = 1
cnt_file_name = meta_df.iloc[i].cnt_filename
site_name = meta_df.iloc[i].site.lower()
cnt_file_path = DATA_PATH + site_name + '\\' + cnt_file_name
data = mne.io.read_raw_cnt(cnt_file_path, data_format='int16', preload=True)

raw = data.copy()

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
rename_channels_eeglab_standard(raw)
# raw.plot_sensors(kind='3d')

# IMPORT AND DOWNSAMPLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# downsampling to reduce processing time and for consistency
if raw.info['sfreq'] > 256.0:
    raw.resample(256.0)
    # INITIAL FILTERING
    # NOW WE PERFORM PREPROCESSING STEPS ON THE (COMPLETELY) RAW DATA FROM THE .CNT FILES
    # LOW AND HIGH PASS FILTERING THAT SATISFIES ZERO-PHASE DESIGN
    raw = raw.filter(lowfrq, hifrq)
    # REMOVE 60 HZ LINE NOISE FROM SIGNAL WITH NOTCH FILTER
    raw.notch_filter(60, filter_length='auto', phase='zero', verbose=False)


# SIGNAL QUALITY CHECKS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~







# RE-REFERENCING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # WE NEED TO APPLY A COMMON AVERAGE REFERENCE TO USE MNE-ICALabel         
raw = raw.set_eeg_reference("average")

# ARTIFACT REMOVAL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # 1. Initialize the ICA object 
# ica = ICA(
#     n_components=15,
#     max_iter="auto",
#     random_state=42,
#     method="infomax",
#     fit_params=dict(extended=True),
#     verbose=False
#     )
# # 2. Fit the ICA STRICTLY on the scalp EEG channels to prevent EOG variance distortion
# ica.fit(raw, picks='eeg')
# # 3. Ground-Truth Validation: Correlate components with physical EOG
# # MNE automatically looks for channels with the 'eog' type to perform this math
# eog_indices, eog_scores = ica.find_bads_eog(raw)
# # 4. Automatically add the highly correlated components to the exclusion list
# ica.exclude.extend(eog_indices)
# print(f"Ground-truth validation flagged the following eye components: {eog_indices}")
# # 5. Run mne-icalabel to catch remaining non-eye artifacts (muscle, heartbeat)
# ic_labels = label_components(raw, ica, method='iclabel')
# # Extract components labeled as 'muscle' or 'heart' and add them to the exclusion list
# probabilities = ic_labels['y_pred_proba']
# labels = ic_labels['labels']
# for idx, label in enumerate(labels):
#     if label in ['muscle artifact', 'heart beat'] and idx not in ica.exclude:
#         ica.exclude.append(idx)
# # 6. Apply the ICA to subtract all flagged components from the continuous data
# # This leaves the underlying brain activity intact
# raw_clean = ica.apply(raw.copy())

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
ica.exclude.extend(eog_indices)
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
rawclean = ica.apply(raw)

ica.plot_components(sphere='eeglab')



# PSEUDO-EPOCHING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


