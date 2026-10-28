# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:32:04 2026

@author: lifep
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

from pyprep.find_noisy_channels import NoisyChannels
from autoreject import AutoReject
import traceback
from tensorpac import Pac
from scipy.signal import welch
from fooof import FOOOF
# from fooof.analysis import get_band_peak_fm
from scipy.ndimage import maximum_filter

# from bycycle.cyclepoints import _infer_zerocrossings
from bycycle.features import compute_features
from scipy.signal import hilbert
from bycycle import Bycycle
from bycycle.plts import plot_cyclepoints_array
from bycycle.cyclepoints import find_extrema, find_zerox
from neurodsp.filt import filter_signal


# TO DEAL WITH DEPRECATION WARNINGS
import warnings
import scipy.fftpack
import scipy.fftpack.helper
# Alias the deprecated helper attribute to the modern implementation
scipy.fftpack.helper.next_fast_len = scipy.fftpack.next_fast_len
# Suppress DeprecationWarnings from tensorpac internals
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorpac")
# Suppress scipy-specific helper deprecation
warnings.filterwarnings("ignore", category=DeprecationWarning, module="scipy")

# DEBUG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
i = 0
# True False
do_preproc = True
do_quality_checks = True
do_artifact_removal = True
do_AutoReject = False
do_pac = True
do_bycycle = False
do_MWT_phase_plane = False
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DEBUG

epoch_dur = 10
epochs_per_block = 1

DATA_PATH = 'E:\\COGA_eec\\data\\'
WRITE_PATH = 'E:\\COGA_eec\\eeg_pipe\\'

# Choose channels to inspect (e.g., 'CZ', 'FZ', or 'OZ') for PAC estimates
PHI_channel = 'F3'
AMP_channel = 'O1'   
pac_method, pac_surrogate, pac_correction = 6, 2, 4
n_perm = 500
mcp = 'fdr' # maxstat bonferroni fdr

phi_start = 7
phi_stop = 14
phi_width = 1
phi_step = 0.5

amp_start = 24
amp_stop = 52
amp_width = 2
amp_step = 1

vmax_pac = 4

notch_freqs = [60.0, 80.0]      # FREQUENCY (Hz) TO REMOVE LINE NOISE AND UNKNOWN NOISE FROM SIGNAL  
lowfrq = 1                      # LOW PASS FREQUENCY, RECOMMENDED SETTING TO 1 HZ IF USING mne-icalabel
hifrq = 100                     # HIGH PASS FREQUENCY
do_plot_channels = False  # TO GENERATE PLOTS OF THE CLEANED EEG SIGNAL
eye_blink_chans = ['X', 'Y'] # NAMES OF CHANNELS CONTAINING EOG

# PAC METHODS
# 1 : Mean Vector Length (MVL) 
# 2 : Modulation Index (MI) 
# 3 : Heights Ratio (HR) 
# 4 : ndPAC (see tensorpac.methods.norm_direct_pac())
# 5 : Phase-Locking Value (PLV) 
# 6 : Gaussian Copula PAC (GCPAC)
# PAC SURROGATE DISTRIBUTIONS
# 0 : No surrogates
# 1 : Swap phase / amplitude across trials
# 2 : Swap amplitude time blocks
# 3 : Time lag
# PAC CORRECTION METHODS
# 0 : No normalization
# 1 : Substract the mean of surrogates
# 2 : Divide by the mean of surrogates
# 3 : Substract then divide by the mean of surrogates
# 4 : Z-score


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



def validate_oscillation_peaks(
    data, 
    sfreq, 
    f_pha_range=(3, 12), 
    min_peak_height=0.15, 
    aperiodic_mode='fixed'
    ):
    """
    Automates oscillation presence checks across epochs using specparam.
    
    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_times)
        Accepts:
      - A single 1D epoch: shape (n_times,) -> returns (bool, float)
      - A 2D batch of epochs: shape (n_epochs, n_times) -> returns (ndarray, ndarray)
        Raw electrophysiological time series.
    sfreq : float
        Sampling rate in Hz.
    f_pha_range : tuple (f_min, f_max)
        Target phase frequency range to validate.
    min_peak_height : float
        Minimum peak amplitude above 1/f floor (in log10 units).
    aperiodic_mode : str
        'fixed' (standard) or 'knee' (if low frequencies flatten).
        
    Returns
    -------
    valid_mask : ndarray of bool, shape (n_epochs,)
        Boolean array where True indicates a genuine spectral peak exists.
    peak_cf : ndarray of float, shape (n_epochs,)
        Center frequency of detected peak (NaN if no peak).
    """
    # Detect if data is a single 1D epoch
    is_single_epoch = (data.ndim == 1)
    epochs = np.atleast_2d(data)
    n_epochs = epochs.shape[0]    
    valid_mask = np.zeros(n_epochs, dtype=bool)
    peak_cf = np.full(n_epochs, np.nan)
    
    # Initialize spectral parameterizer
    fm = FOOOF(
        peak_width_limits=(1.0, 4.0),
        min_peak_height=min_peak_height,
        max_n_peaks=4,
        aperiodic_mode=aperiodic_mode,
        verbose=False
    )
    
    for i in range(n_epochs):
        # # 1. Compute Welch PSD (ensure window preserves low-frequency resolution)
        # freqs, psd = welch(epochs[i], fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq))
        
        # data shape: (n_epochs, n_channels, n_times) or (n_times,)
        # 10s window at 256 Hz with bandwidth = 0.5 Hz (NW ≈ 2.5, 4 tapers)
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            data, 
            sfreq=sfreq, 
            fmin=2, 
            fmax=35, 
            bandwidth=1,  # spectral smoothing width in Hz
            adaptive=True, 
            normalization='full',
            verbose=False
        )
        
        # 2. Fit periodic + aperiodic model over a relevant window (e.g., 2 to 45 Hz)
        try:
            fm.fit(freqs, psd, freq_range=[2, 35])
            fm.plot()
        except Exception:
            continue
            
        # 3. Query peaks within the candidate phase band
        # fm.peak_params_ returns array of [center_frequency, power_over_baseline, bandwidth]
        peaks = fm.peak_params_
        if peaks is None or len(peaks) == 0:
            continue
            
        pha_peaks = [
            p for p in peaks 
            if f_pha_range[0] <= p[0] <= f_pha_range[1]
        ]
        
        # 4. Check if any peak passes the height threshold
        if len(pha_peaks) > 0:
            # Select peak with highest power over baseline
            best_peak = max(pha_peaks, key=lambda x: x[1])
            if best_peak[1] >= min_peak_height:
                valid_mask[i] = True
                peak_cf[i] = best_peak[0]
                
    # Return scalar results if the input was a single 1D epoch
    if is_single_epoch:
        return bool(valid_mask[0]), float(peak_cf[0])
                
    return valid_mask, peak_cf



def detect_harmonic_pac_peaks(
    pac_matrix, 
    f_pha_vec, 
    f_amp_vec, 
    psd_freqs, 
    psd_power, 
    pac_threshold=0.2, 
    harmonic_tolerance=0.08
):
    """
    Identifies comodulogram peaks and classifies them as candidate harmonics.
    
    Parameters
    ----------
    pac_matrix : ndarray, shape (n_amp, n_pha)
        2D comodulogram matrix (e.g., from Tensorpac).
    f_pha_vec : ndarray, shape (n_pha,)
        Center frequencies for phase bins.
    f_amp_vec : ndarray, shape (n_amp,)
        Center frequencies for amplitude bins.
    psd_freqs : ndarray
        Frequencies from power spectral density.
    psd_power : ndarray
        Power spectral density values (linear or log).
    pac_threshold : float
        Minimum PAC value to qualify as a peak.
    harmonic_tolerance : float
        Allowed fractional deviation from exact integer multiples.
        
    Returns
    -------
    peaks : list of dict
        Detected local maxima with harmonic classification.
    """
    # 1. Detect 2D local maxima in the comodulogram
    neighborhood_size = 3
    local_max = (maximum_filter(pac_matrix, size=neighborhood_size) == pac_matrix)
    above_thresh = (pac_matrix >= pac_threshold)
    peak_coords = np.argwhere(local_max & above_thresh)
    
    results = []
    
    for r, c in peak_coords:
        f_pha = f_pha_vec[c]
        f_amp = f_amp_vec[r]
        val = pac_matrix[r, c]
        
        # 2. Check harmonic ratio
        ratio = f_amp / f_pha
        nearest_int = int(np.round(ratio))
        ratio_dev = np.abs(ratio - nearest_int) / nearest_int
        
        is_integer_multiple = (nearest_int >= 2) and (ratio_dev <= harmonic_tolerance)
        
        # 3. Check if raw PSD has an independent peak at f_amp
        # If the PSD at f_amp is just decaying 1/f or absent, harmonic artifact likelihood spikes
        amp_idx = np.argmin(np.abs(psd_freqs - f_amp))
        # Look in a +/- 2 Hz window around f_amp in PSD
        window = (psd_freqs >= f_amp - 2.0) & (psd_freqs <= f_amp + 2.0)
        has_psd_local_bump = (
            len(psd_power[window]) > 0 and 
            psd_power[amp_idx] == np.max(psd_power[window])
        )
        
        # Classification heuristic
        if is_integer_multiple and not has_psd_local_bump:
            classification = "Likely Harmonic Artifact (Waveform Asymmetry)"
        elif is_integer_multiple and has_psd_local_bump:
            classification = "Ambiguous (Integer Multiple with PSD Peak - Run Bicoherence)"
        else:
            classification = "Candidate Genuine Coupling"
            
        results.append({
            "f_pha": f_pha,
            "f_amp": f_amp,
            "pac_value": val,
            "harmonic_ratio": ratio,
            "nearest_multiple": nearest_int,
            "is_harmonic_candidate": is_integer_multiple,
            "classification": classification
        })
        
    return results



# ~~~~~~~~~~~~~~~~~~~ START ~~~~~~~~~~~~~~~~~~~~~~~~
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
age = meta_df.iloc[i]['age_this_visit']
sex = meta_df.iloc[i]['sex']
diag = meta_df.iloc[i]['AUD_this_visit']

cnt_file_name = meta_df.iloc[i].cnt_file_name
sample_freq = meta_df.iloc[i].sample_freq
site_name = meta_df.iloc[i].site.lower()
cnt_file_path = DATA_PATH + site_name + '\\' + cnt_file_name
print(f"Attempting to process: {meta_df.iloc[i].eeg_file_name}")



if do_preproc:

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
    
if do_quality_checks:
    
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
    

if do_artifact_removal:
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
    raw = ica.apply(raw)
    
    
    # PSEUDO-EPOCHING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Pseudo-Epoching: Slice the continuous clean data into fixed-length epochs
    # We are using 4.0 seconds, but 2 to 4 seconds is standard for resting state
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_dur, preload=True)

    
if do_AutoReject:
    
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
    

# # 1. Load the cleaned n-second epoched FIF file
# fif_path = r"E:\COGA_eec\eeg_pipe\eec_1_a1_10003051-epo.fif"
fif_path = WRITE_PATH + cnt_file_name.replace('.cnt', '-epo.fif')
if 'epochs' in locals():
    epochs_clean = epochs
else:
    epochs_clean = mne.read_epochs(fif_path, preload=True)

# Extract data: shape is (n_epochs, n_channels, n_times)
epoch_data = epochs_clean.get_data()


ch1_idx = epochs_clean.ch_names.index(PHI_channel)
ch2_idx = epochs_clean.ch_names.index(AMP_channel)


if do_pac:
    
    # Extract sampling frequency and data: shape -> (n_epochs, n_channels, n_times)
    sfreq = epochs_clean.info['sfreq']
    ch_names = epochs_clean.ch_names
    
    # 2. Group into 30-second intervals
    # Since epochs are 10s each, 3 consecutive epochs equal 30s
    n_blocks = len(epochs_clean) // epochs_per_block
    
    
    # 3. Initialize Tensorpac PAC object
    # idpac=(2, 0, 0): Modulation Index (Tort et al.), no surrogate, no normalization
    p = Pac(idpac=(pac_method, pac_surrogate, pac_correction), 
            f_pha=(phi_start, phi_stop, phi_width, phi_step), 
            f_amp=(amp_start, amp_stop, amp_width, amp_step), 
            dcomplex='wavelet', 
            width=7
            )
    

    
    # 4. Compute and plot comodulograms per 30-second successive interval
    for block_idx in range(n_blocks):
        start_ep = block_idx * epochs_per_block
        # start_ep = block_idx 
        end_ep = start_ep + epochs_per_block
        
        # Slice the 3 epochs for this 30s block
        # Shape needed for tensorpac: (n_trials, n_times)
        block_data_PHI = epochs_clean.get_data() [start_ep:end_ep, ch1_idx, :][0]
        block_data_AMP = epochs_clean.get_data() [start_ep:end_ep, ch2_idx, :][0]
        
        # fq, psd = welch(block_data,
        #                 fs=sfreq,
        #                 nperseg=int(sfreq*2),
        #                 noverlap=int(sfreq)
        #                 )
        # plt.plot(fq,psd)
        # plt.xlim(0,50)
        # plt.ylim(0,3e-11)
        # plt.title(f"{age} {sex} AUD={diag} - {PHI_channel} - Block {block_idx+1} (30s)")
        
        
        # valid_mask, peak_cf = validate_oscillation_peaks(
        #     data=block_data[0],
        #     sfreq=raw.info['sfreq'],
        #     f_pha_range=(4,8),
        #     min_peak_height=0.3
        # )
        # print(f"Single Valid Epoch: is_valid={valid_mask}, center_freq={peak_cf:.2f} Hz")
        
    
        # Compute the comodulogram across the pooled trials of this block
        pac_matrix12 = p.filterfit(sfreq, block_data_PHI, block_data_AMP, n_perm=n_perm, n_jobs=-1,random_state=42)
        pval12 = p.infer_pvalues(p=1.0, mcp=mcp)
        print(f"Minimum p-value {pval12.min()}")
        xpac12 = pac_matrix12.mean(axis=-1)
        xpac_smean12 = xpac12[pval12<=0.05].mean()

        surro12 = p.surrogates.squeeze()
        print(f"Surrogates shape (n_perm, n_amp, n_pha) : {surro12.shape}")
        surro_max12 = surro12.max(axis=(1, 2))
        # plt.hist(surro_max)
        print(f"Surrogates min, max : {surro_max12.min()}, {surro_max12.max()}")
        surro_max_z12 = (surro_max12 - np.mean(surro_max12)) / np.std(surro_max12)

        
        pac_matrix21 = p.filterfit(sfreq, block_data_AMP, block_data_PHI, n_perm=n_perm, n_jobs=-1,random_state=42)
        pval21 = p.infer_pvalues(p=1.0, mcp=mcp)
        print(f"Minimum p-value {pval21.min()}")
        xpac21 = pac_matrix21.mean(axis=-1)
        xpac_smean21 = xpac21[pval21<=0.05].mean()

        surro21 = p.surrogates.squeeze()
        print(f"Surrogates shape (n_perm, n_amp, n_pha) : {surro21.shape}")
        surro_max21 = surro21.max(axis=(1, 2))
        # plt.hist(surro_max)
        print(f"Surrogates min, max : {surro_max21.min()}, {surro_max21.max()}")
        surro_max_z21 = (surro_max21 - np.mean(surro_max21)) / np.std(surro_max21)

        
        # # -------------------------------------------------------------------------
        # # 4. Run the Harmonic Detector Function
        # # -------------------------------------------------------------------------
        # # Extract center frequencies
        # f_pha_vec = p.xvec
        # f_amp_vec = p.yvec
        # detected_peaks = detect_harmonic_pac_peaks(
        #     pac_matrix=pac_matrix,
        #     f_pha_vec=f_pha_vec,
        #     f_amp_vec=f_amp_vec,
        #     psd_freqs=fq,
        #     psd_power=psd,
        #     pac_threshold=2.0,       # Significant z-score threshold
        #     harmonic_tolerance=0.08  # Within 8% of an exact integer multiple
        # )
        
        
        # # Print tabular report
        # print(f"{'f_pha (Hz)':>10} | {'f_amp (Hz)':>10} | {'Z-PAC':>7} | {'Ratio':>6} | {'Classification'}")
        # print("-" * 75)
        # for res in detected_peaks:
        #     print(
        #         f"{res['f_pha']:10.1f} | "
        #         f"{res['f_amp']:10.1f} | "
        #         f"{res['pac_value']:7.2f} | "
        #         f"{res['harmonic_ratio']:6.2f} | "
        #         f"{res['classification']}"
        #     )
        
        # # -------------------------------------------------------------------------
        # # 5. Visual Inspection Plot
        # # -------------------------------------------------------------------------
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        
        # # Plot PSD
        # ax1.semilogy(fq, psd, color='black', lw=1.5)
        # ax1.set_xlim(2, 60)
        # ax1.set_title("Power Spectral Density")
        # ax1.set_xlabel("Frequency (Hz)")
        # ax1.set_ylabel("Power (linear/Welch)")
        # ax1.axvline(10.0, color='crimson', linestyle='--', label='Alpha (10 Hz)')
        # ax1.axvline(20.0, color='orange', linestyle=':', label='2nd Harmonic (20 Hz)')
        # ax1.axvline(30.0, color='orange', linestyle=':', label='3rd Harmonic (30 Hz)')
        # ax1.legend(loc='upper right')
        
        # # Plot Comodulogram
        # im = ax2.pcolormesh(f_pha_vec, f_amp_vec, pac_matrix, cmap='viridis', shading='auto')
        # fig.colorbar(im, ax=ax2, label='Z-scored PAC')
        # ax2.set_title("Comodulogram (Spurious Harmonics)")
        # ax2.set_xlabel("Phase Frequency (Hz)")
        # ax2.set_ylabel("Amplitude Frequency (Hz)")
        
        # # Mark detected harmonic peaks on the comodulogram
        # for res in detected_peaks:
        #     if res['is_harmonic_candidate']:
        #         ax2.scatter(res['f_pha'], res['f_amp'], facecolors='none', edgecolors='red', s=120, lw=2)
        
        # plt.tight_layout()
        # plt.show()
        
        
        
        # # Plot comodulogram
        # fig, ax = plt.subplots(figsize=(6, 5))
        # p.comodulogram(pac_matrix.mean(axis=-1), cmap='viridis', vmin=0, title=f"{PHI_channel} - Block {block_idx+1} (30s)", ax=ax)
        # plt.show()
        # plt.title(f"{age} {sex} AUD={diag} - {PHI_channel} - Block {block_idx+1} (30s)")
    
        # Plot comodulogram using tensorpac's internal axis routing
        plt.figure(figsize=(10, 6))
        plt.subplot(2,2,1)
        p.comodulogram(
            pac_matrix12.mean(axis=-1),
            cmap='viridis',
            vmin=0,
            vmax=4,
            title=f"{age} {sex} AUD={diag} Ph: {PHI_channel} Amp: {AMP_channel} - Block {block_idx+1} ({epoch_dur}s)",
            subplot=111
        )
        
        plt.subplot(2,2,2)
        # plt.plot(block_data_PHI)
        # plt.title(f"phase channel {PHI_channel}")
        
        # p.comodulogram(
        #     c3,
        #     cmap='seismic',
        #     vmin=-1,
        #     vmax=1,
        #     title=f"{PHI_channel} - {AMP_channel}",
        #     subplot=111
        # )
        p.comodulogram(
            pval12,
            cmap='seismic',
            vmin=0,
            vmax=1,
            title=f"p-values (min={pval12.min()})",
            subplot=111
        )
        
        plt.subplot(2,2,3)
        # pac_matrix21 = p.filterfit(sfreq, block_data_PHI, block_data_AMP, n_perm=n_perm, p=0.05, mcp=mcp)
        # pval21 = p.infer_pvalues(p=0.05, mcp=mcp)
        # pac_s21 = pac_matrix21.copy().mean(axis=-1)
        # pac_s21[pval21>0.05] = np.nan
        
        
        # c1 = pac_matrix12.mean(axis=-1)
        # c2 = pac_matrix21.mean(axis=-1)
        # c3 = c1 - c2
        
        p.comodulogram(
            pac_matrix21.mean(axis=-1),
            cmap='viridis',
            vmin=0,
            vmax=vmax_pac,
            title=f"phase {AMP_channel} and amplitude {PHI_channel} - Block {block_idx+1} ({epoch_dur}s)",
            subplot=111
        )
        
        plt.subplot(2,2,4)
        # plt.plot(block_data_AMP)
        # plt.title(f"amplitude channel {AMP_channel}")
        p.comodulogram(
            pval21,
            cmap='seismic',
            vmin=0,
            vmax=1,
            title=f"p-values (min={pval21.min()})",
            subplot=111
        )    
        
        plt.hist(surro_max_z12, bins=20, alpha=0.7, color="steelblue", density=True)
        # # xpac_smean is already in z-score units
        # plt.axvline(
        #     xpac_smean12, lw=2, color="red", label=f"Mean Z-PAC ({xpac_smean12:.2f} $\sigma$)"
        # )
        # plt.title("Z-Score Null vs. Observed Z-PAC")
        # plt.xlabel("Standard Deviations ($\sigma$)")
        # plt.ylabel("Density")
        # plt.legend()
        
        # plt.hist(surro_max_z21, bins=20, alpha=0.7, color="steelblue", density=True)
        # # xpac_smean is already in z-score units
        # plt.axvline(
        #     xpac_smean21, lw=2, color="red", label=f"Mean Z-PAC ({xpac_smean21:.2f} $\sigma$)"
        # )
        # plt.title("Z-Score Null vs. Observed Z-PAC")
        # plt.xlabel("Standard Deviations ($\sigma$)")
        # plt.ylabel("Density")
        # plt.legend()
        
        plt.show()
        plt.close()


if do_bycycle:
    sfreq = epochs_clean.info["sfreq"]
    ch_names = epochs_clean.ch_names
    ch1_idx = ch_names.index(PHI_channel)
    ch2_idx = ch_names.index(AMP_channel)
    phi_start, phi_stop = 8,12
    amp_start, amp_stop = 40,50
    
    # 1. Parameterize candidate cycle definitions
    f_range = (phi_start, phi_stop)  # Phase carrier band (e.g., alpha: 7-14 Hz)
    amp_band = (
        amp_start,
        amp_stop,
    )  # Amplitude band (e.g., gamma: 24-52 Hz)

    # Burst detection thresholds to restrict PAC to genuine oscillatory bursts
    threshold_kwargs = {
        "amp_fraction_threshold": 0.3,
        "amp_consistency_threshold": 0.4,
        "period_consistency_threshold": 0.5,
        "monotonicity_threshold": 0.8,
        "min_n_cycles": 3,
    }

    # # Extract clean continuous data across epochs for both channels
    # # Shape: (n_samples,)
    # sig_phase = epochs_clean.get_data()[:, ch1_idx, :].reshape(-1)
    # sig_amp_target = epochs_clean.get_data()[:, ch2_idx, :].reshape(-1)
    
    # sig_phase = np.concatenate([epoch_data[0,ch1_idx,:], epoch_data[1,ch1_idx,:]])
    # sig_amp_target = np.concatenate([epoch_data[0,ch2_idx,:], epoch_data[1,ch2_idx,:]])
    ep_i = 0
    sig_phase = epoch_data[ep_i,ch1_idx,:]
    sig_amp_target = epoch_data[ep_i,ch2_idx,:]
    
    # Apply a lowpass filter at 25Hz
    fc = 25
    filter_seconds = 0.5
    
    # sig_phase_low = filter_signal(sig_phase, sfreq, 'lowpass', fc, n_seconds=filter_seconds, remove_edges=False)
    # sig_amp_target_low = filter_signal(sig_amp_target, sfreq, 'lowpass', fc, n_seconds=filter_seconds, remove_edges=False)

    # f_alpha = (8,12)
    # f_theta = (4,8)
    # thresholds = {
    #     'amp_fraction': .3,
    #     'amp_consistency': .4,
    #     'period_consistency': .5,
    #     'monotonicity': .8,
    #     'min_n_cycles': 3
    # }
    
    # bm = Bycycle(thresholds=thresholds)
    # bm.fit(sig, sfreq, f_alpha)
    
    # peaks = bm.df_features['sample_peak'].values
    # troughs =bm.df_features['sample_trough'].values
    
    # # 3. Find zerocrossings/flanks to complete the required inputs
    # rises, decays = find_zerox(sig, peaks, troughs)
    
    # fig, ax = plt.subplots(figsize=(15, 3))
    # plot_cyclepoints_array(bm.sig, bm.fs, bm.df_features, ax=ax)
    

    
    
    # 2. Extract Cycle Features on the Phase channel
    df_features = compute_features(
        sig_phase,
        sfreq,
        f_range=f_range,
        threshold_kwargs=threshold_kwargs,
        center_extrema="peak",
    )

    # 3. Derive instantaneous phase (0 to 2*pi) from cycle points
    # Assign cycle extrema to phase values: trough=0, rise_zc=pi/2, peak=pi, decay_zc=3pi/2
    phase_ts = np.full(len(sig_phase), np.nan)
    for _, cycle in df_features.iterrows():
        # Only retain cycles meeting the burst detection criteria
        if not cycle["is_burst"]:
            continue

        c_start = int(cycle["sample_last_trough"])
        c_peak = int(cycle["sample_peak"])
        c_end = int(cycle["sample_next_trough"])
        c_zerox_rise = int(cycle["sample_zerox_rise"])
        c_zerox_decay = int(cycle["sample_zerox_decay"])

        # Interpolate phase segmentally to preserve nonsinusoidal shape
        phase_ts[c_start:c_zerox_rise] = np.linspace(
            0, np.pi / 2, c_zerox_rise - c_start, endpoint=False
        )
        phase_ts[c_zerox_rise:c_peak] = np.linspace(
            np.pi / 2, np.pi, c_peak - c_zerox_rise, endpoint=False
        )
        phase_ts[c_peak:c_zerox_decay] = np.linspace(
            np.pi, 3 * np.pi / 2, c_zerox_decay - c_peak, endpoint=False
        )
        phase_ts[c_zerox_decay:c_end] = np.linspace(
            3 * np.pi / 2, 2 * np.pi, c_end - c_zerox_decay, endpoint=False
        )

    # 4. Extract Amplitude Envelope from the Target Channel
    # Bandpass filter the target channel and apply Hilbert transform
    raw_amp = mne.filter.filter_data(
        sig_amp_target,
        sfreq,
        l_freq=amp_band[0],
        h_freq=amp_band[1],
        verbose=False,
    )
    amp_env = np.abs(hilbert(raw_amp))

    # 5. Compute Modulation Index (Tort et al., 2010) during detected bursts
    valid_idx = ~np.isnan(phase_ts)
    valid_phase = phase_ts[valid_idx]
    valid_amp = amp_env[valid_idx]

    n_bins = 18
    phase_bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    mean_amp_per_bin = np.zeros(n_bins)

    bin_assignments = np.digitize(valid_phase, phase_bins) - 1
    for b in range(n_bins):
        amps_in_bin = valid_amp[bin_assignments == b]
        mean_amp_per_bin[b] = (
            np.mean(amps_in_bin) if len(amps_in_bin) > 0 else 0
        )

    # Normalize amplitude distribution across bins to sum to 1
    P = mean_amp_per_bin / np.sum(mean_amp_per_bin)
    # Shannon entropy and KL-divergence against uniform distribution Q
    shannon_entropy = -np.sum(P * np.log(P + 1e-12))
    log_N = np.log(n_bins)
    mi = (log_N - shannon_entropy) / log_N

    print(
        f"ByCycle Cross-PAC ({PHI_channel} phase -> {AMP_channel} amp): MI = {mi:.5f}"
    )

    # 6. Plot Phase-Amplitude Distribution
    fig, ax = plt.subplots(
        subplot_kw={"projection": "polar"}, figsize=(6, 6)
    )
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    ax.bar(
        bin_centers,
        P,
        width=(2 * np.pi / n_bins),
        bottom=0.0,
        edgecolor="k",
        alpha=0.7,
    )
    ax.set_title(
        f"ByCycle PAC: {PHI_channel} (Phase) vs {AMP_channel} (Amp)\nMI = {mi:.5f}",
        va="bottom",
    )
    plt.show()


if do_MWT_phase_plane:
    import pywt
    
    # # --- 1. Generate Synthetic EEG Signal ---
    # sfreq = 250  # Sampling rate in Hz
    # duration = 4.0  # Duration in seconds
    t = np.arange(0, len(sig_phase), sfreq)
    dt = 1 / sfreq
    
    # # Simulated EEG segment: broadband noise + transient oscillatory bursts (alpha & beta)
    # rng = np.random.default_rng(42)
    # noise = rng.normal(0, 0.5, size=t.shape)
    # alpha_burst = 1.5 * np.sin(2 * np.pi * 10 * t) * np.exp(-((t - 1.5) ** 2) / 0.1)
    # beta_burst = 0.8 * np.sin(2 * np.pi * 22 * t) * np.exp(-((t - 2.8) ** 2) / 0.15)
    # eeg_segment = alpha_burst + beta_burst + noise
    
    
    eeg_segment = sig_phase
    
    # --- 2. Time-Frequency Analysis via PyWavelets ---
    wavelet_name = 'cmor1.5-1.0'  # Complex Morlet with B=1.5 and C=1.0
    target_freqs = np.linspace(2, 40, 100)  # Desired frequency grid in Hz
    
    # Map desired physical frequencies (Hz) to wavelet scales: scale = fc / (f * dt)
    fc = pywt.central_frequency(wavelet_name)
    scales = fc / (target_freqs * dt)
    
    # Compute complex CWT (returns coefficients and effective frequencies in Hz)
    cwt_matrix, freqs = pywt.cwt(
        eeg_segment, scales, wavelet_name, sampling_period=dt
    )
    
    # Extract the imaginary component
    cwt_imaginary = np.imag(cwt_matrix)
    
    # --- 3. Multi-Panel Visualization ---
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]}
    )
    
    # Top Panel: Raw EEG Segment
    ax_top.plot(t, eeg_segment, color='#2c3e50', lw=1.2)
    ax_top.set_ylabel(r'Amplitude ($\mu$V)')
    ax_top.set_title('Raw EEG Segment', fontweight='bold', loc='left')
    ax_top.grid(True, linestyle='--', alpha=0.5)
    
    # Bottom Panel: Time-Frequency Imaginary Plane
    vbound = np.percentile(np.abs(cwt_imaginary), 98)  # Symmetric color limits
    im = ax_bottom.pcolormesh(
        t, freqs, cwt_imaginary,
        shading='gouraud',
        cmap='coolwarm',
        vmin=-vbound,
        vmax=vbound
    )
    ax_bottom.set_ylabel('Frequency (Hz)')
    ax_bottom.set_xlabel('Time (s)')
    ax_bottom.set_title(f'Imaginary Component Time-Frequency Plane ({wavelet_name})', fontweight='bold', loc='left')
    
    # Colorbar for quadrature/phase representation
    cbar = fig.colorbar(im, ax=ax_bottom, orientation='horizontal', pad=0.2, aspect=30)
    cbar.set_label(r'$\Im\{\text{CWT}(t, f)\}$')
    
    plt.tight_layout()
    plt.show()