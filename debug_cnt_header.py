# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:11:49 2026

@author: lifep
"""

import mne
import numpy as np
import os

cnt_file_path = r'E:\COGA_eec\data\uconn\eec_1_a1_10003051.cnt'

if 0:
    # 1. Grab valid metadata from the broken header
    raw_header = mne.io.read_raw_cnt(cnt_file_path, data_format='int32', preload=False)
    sfreq = raw_header.info['sfreq']
    ch_names = raw_header.info['ch_names']
    n_chan = raw_header.info['nchan']
    
    # 2. Calculate true sample count from the 5.3 MB physical file
    header_bytes = 900
    bytes_per_sample = 2  # int16
    file_size_bytes = os.path.getsize(cnt_file_path)
    
    # Determine exact number of valid 32-channel multiplexed frames
    usable_bytes = file_size_bytes - header_bytes
    total_samples = usable_bytes // bytes_per_sample
    frames = total_samples // n_chan
    
    # 3. Extract raw binary, skipping the corrupted header
    with open(cnt_file_path, 'rb') as f:
        f.seek(header_bytes)
        # Read only the valid frames to avoid tripping on EOF event tables
        raw_bytes = np.fromfile(f, dtype=np.int16, count=frames * n_chan)
    
    # 4. Reshape and transpose for MNE (channels, samples)
    data_array = raw_bytes.reshape(frames, n_chan).T
    
    # 5. Build a fresh, uncorrupted Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data_array, info)
    
    print(f"Successfully loaded {frames} frames ({frames/sfreq:.2f} seconds) of continuous data.")
    
    
    events_from_mne, event_dict = mne.events_from_annotations(raw_header)
    print(events_from_mne)
    
if 0:
    import os
    import struct
    import numpy as np
    
    cnt_file_path = r'E:\COGA_eec\data\uconn\eec_1_a1_10003051.cnt'
    file_size = os.path.getsize(cnt_file_path)
    
    # Known parameters from your data block
    header_bytes = 900
    bytes_per_sample = 2  # int16
    n_chan = 32           # Extracted from your previous array shape error
    
    events_list = []
    
    with open(cnt_file_path, 'rb') as f:
        # 1. Read the event table pointer at byte offset 886
        f.seek(886)
        event_table_pos = struct.unpack('<i', f.read(4))[0]
    
        if event_table_pos < header_bytes or event_table_pos >= file_size:
            print(f"Error: Invalid event table position ({event_table_pos}).")
        else:
            f.seek(event_table_pos)
            event_table_bytes = file_size - event_table_pos
            
            # 2. Determine event struct size. Neuroscan uses 8 bytes (Type 1) or 19 bytes (Type 2/3).
            event_size = 19 if event_table_bytes % 19 == 0 else 8
            num_events = event_table_bytes // event_size
            
            print(f"Found event table at byte {event_table_pos}. Extracting {num_events} events...")
    
            for i in range(num_events):
                raw_event = f.read(event_size)
                if len(raw_event) < 8:
                    break
                    
                # 3. Unpack the first 8 bytes (stim_type: uint16, keyboard: uint8, keypad: uint8, offset: int32)
                stim_type, keyboard, keypad_accept, byte_offset = struct.unpack('<H B B i', raw_event[:8])
                
                # 4. Neuroscan stores the absolute byte position of the trigger. 
                # Convert to an MNE sample index based on the continuous data block.
                sample_index = (byte_offset - header_bytes) // (n_chan * bytes_per_sample)
                
                if sample_index >= 0:
                    # MNE format: [sample_index, previous_value (usually 0), trigger_code]
                    events_list.append([sample_index, 0, stim_type])
    
    # 5. Convert to MNE standard N x 3 integer array
    events_array = np.array(events_list, dtype=int)
    print(f"Successfully extracted {len(events_array)} triggers.")
    
    
if 0:

    import os
    import mne
    import numpy as np
    
    cnt_file_path = r'E:\COGA_eec\data\uconn\eec_1_a1_10003051.cnt'
    
    # 1. Parse the physical file to bypass the corrupted n_samples header
    header_bytes = 900
    bytes_per_sample = 2  # int16
    n_chan = 32
    
    file_size_bytes = os.path.getsize(cnt_file_path)
    usable_bytes = file_size_bytes - header_bytes
    frames = (usable_bytes // bytes_per_sample) // n_chan
    
    with open(cnt_file_path, 'rb') as f:
        f.seek(header_bytes)
        raw_bytes = np.fromfile(f, dtype=np.int16, count=frames * n_chan)
    
    # Reshape into (channels, samples)
    data_array = raw_bytes.reshape(frames, n_chan).T
    
    # 2. Extract triggers from the stimulus channel (Index 31 for a 32-channel system)
    stim_channel = data_array[31, :]
    events_list = []
    
    # Scan for pulses (a transition from 0 to a non-zero trigger code)
    for i in range(1, frames):
        current_val = stim_channel[i]
        prev_val = stim_channel[i-1]
        
        # When the channel jumps to a new positive integer, record the onset
        if current_val > 0 and current_val != prev_val:
            events_list.append([i, 0, current_val])
    
    events_array = np.array(events_list, dtype=int)
    
    print(f"Extracted {len(events_array)} triggers from the stimulus channel.")
    if len(events_array) > 0:
        print(f"Unique trigger codes found: {np.unique(events_array[:, 2])}")
    else:
        print("No non-zero trigger codes found on channel 32.")
        
if 1:
    import numpy as np
    import matplotlib.pyplot as plt

    
    cnt_file_path = r'E:\COGA_eec\data\uconn\eec_1_a1_10003051.cnt'
    cnt_file_path = r'E:\COGA_eec\data\uconn\eec_1_a1_10006015.cnt'
    cnt_file_path = r'E:\COGA_eec\data\uconn\eec_1_a1_10006127.cnt'
    cnt_file_path = r'E:\COGA_eec\data\uconn\eec_1_a1_10007003.cnt'
    # cnt_file_path = r'E:\COGA_eec\data\uconn\xxxxxxxxxxxxx.cnt'
    # cnt_file_path = r'E:\COGA_eec\data\uconn\xxxxxxxxxxxxx.cnt'

    header_bytes = 900
    # frames_to_test = 2000  # roughly 2 seconds depending on sfreq
    frames_to_test = 100  # roughly 2 seconds depending on sfreq
    n_chan = 32
    
    with open(cnt_file_path, 'rb') as f:
        f.seek(header_bytes)
        raw_bytes = f.read(frames_to_test * n_chan * 2)
    
    # Read as Little-Endian (Standard Windows/Neuroscan)
    le_data = np.frombuffer(raw_bytes, dtype=np.dtype('<i2'))
    
    # Read as Big-Endian (Sun SPARC/UNIX)
    be_data = np.frombuffer(raw_bytes, dtype=np.dtype('>i2'))
    
    print("--- Maximum Amplitudes (Raw Integers) ---")
    print(f"Little-Endian Max Value: {np.abs(le_data).max()}")
    print(f"Big-Endian Max Value:    {np.abs(be_data).max()}")
    
    

    
    with open(cnt_file_path, 'rb') as f:
        f.seek(header_bytes)
        raw_bytes = f.read(frames_to_test * n_chan * 2)
    
    le_data = np.frombuffer(raw_bytes, dtype=np.dtype('<i2')).reshape(frames_to_test, n_chan).T
    be_data = np.frombuffer(raw_bytes, dtype=np.dtype('>i2')).reshape(frames_to_test, n_chan).T
    
    # Plot Channel 1 (Index 0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    ax1.plot(le_data[0, 33:], color='blue')
    ax1.set_title('Little-Endian (Biological Drift) ' + cnt_file_path.split('\\')[-1])
    ax1.set_ylabel('Raw Amplitude')
    
    ax2.plot(be_data[0, 33:], color='red')
    ax2.set_title('Big-Endian (Byte-Swapped White Noise)')
    ax2.set_ylabel('Raw Amplitude')
    
    plt.tight_layout()
    plt.show()