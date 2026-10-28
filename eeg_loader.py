import os
import logging
from pathlib import Path
from typing import Union

import numpy as np
import mne



# -----------------------------------------------------------------------------
# Logger Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger("EEGLoader")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Core Loader Function
# -----------------------------------------------------------------------------
def load_eeg(file_path: Union[str, Path], montage_name: str = 'standard_1020') -> mne.io.Raw:
    """
    Loads an EEG file into an MNE Raw object.
    
    Automatically handles standard formats (.edf, .vhdr, .fif) and routes 
    corrupted .cnt files to a safe binary brute-force extractor to bypass 
    Windows 32-bit Overflow errors.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    logger.info(f"Initiating load for: {file_path.name}")

    try:
        # Route by file extension
        if ext == '.cnt':
            raw = _load_neuroscan_cnt(file_path)
        elif ext == '.edf':
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif ext == '.vhdr':
            raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        elif ext == '.fif':
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        else:
            logger.error(f"Unsupported file extension: {ext}")
            raise ValueError(f"Unsupported file extension: {ext}")

        # Apply standard electrode geometry (fails silently if channel names don't match 10-20)
        try:
            raw.set_montage(montage_name, match_case=False)
            logger.info(f"Successfully applied '{montage_name}' montage.")
        except Exception as e:
            logger.warning(f"Could not apply montage '{montage_name}': {e}")

        return raw

    except Exception as e:
        logger.error(f"Fatal error loading {file_path.name}: {str(e)}")
        raise


# # -----------------------------------------------------------------------------
# # Private Helper: Corrupted CNT Handler
# # -----------------------------------------------------------------------------
# def _load_neuroscan_cnt(file_path: Path) -> mne.io.Raw:
#     """
#     Attempts standard MNE CNT load. If a 1990s header corruption triggers a 
#     Windows OverflowError or RuntimeError, it falls back to binary extraction.
#     """
#     try:
#         # Attempt standard load first (works for uncorrupted files)
#         return mne.io.read_raw_cnt(file_path, preload=True, verbose=False)
    
#     except (OverflowError, RuntimeError) as e:
#         logger.warning(f"Standard parser failed ({type(e).__name__}). Engaging binary fallback.")
        
#         # 1. Force int16 to bypass Windows math check and extract valid metadata
#         raw_header = mne.io.read_raw_cnt(file_path, data_format='int16', preload=False, verbose=False)
#         info = raw_header.info
        
#         header_bytes = 900
#         bytes_per_sample = 2
#         n_chan = info['nchan']
        
#         # 2. Calculate true geometry from physical file size
#         usable_bytes = os.path.getsize(file_path) - header_bytes
#         frames = (usable_bytes // bytes_per_sample) // n_chan
        
#         logger.info(f"Physical file calculation: {frames} frames across {n_chan} channels.")
        
#         # 3. Read binary payload, skipping the broken header
#         with open(file_path, 'rb') as f:
#             f.seek(header_bytes)
#             raw_bytes = np.fromfile(f, dtype=np.int16, count=frames * n_chan)
            
#         data_array = raw_bytes.reshape(frames, n_chan).T
#         data_array_volts = np.zeros_like(data_array, dtype=np.float64)
        
#         # 4. Apply channel-specific calibration factors to convert to Volts
#         for i, ch in enumerate(info['chs']):
#             cal = ch.get('cal', 1e-6)  # Default to standard microvolt scalar if missing
#             data_array_volts[i, :] = data_array[i, :] * cal
            
#         # 5. Build and return the clean Raw object
#         raw = mne.io.RawArray(data_array_volts, info, verbose=False)
#         logger.info("Binary fallback extraction successful.")
#         return raw
    


def _detect_byte_depth(file_path: Path, n_chan: int, data_start_offset: int) -> str:
    """
    Dynamically determines if a Neuroscan CNT file is 16-bit or 32-bit 
    by checking for bit-shift amplitude explosion.
    """
    # Read a small ~2 second chunk
    frames_to_test = 1000
    bytes_to_read = frames_to_test * n_chan * 4 
    
    # Protect against unusually small files
    max_bytes = os.path.getsize(file_path) - data_start_offset
    bytes_to_read = min(bytes_to_read, max_bytes)
    bytes_to_read -= (bytes_to_read % 4)  # Ensure exact 32-bit alignment
    
    with open(file_path, 'rb') as f:
        f.seek(data_start_offset)
        raw_bytes = f.read(bytes_to_read)
        
    # Interpret the bytes as 32-bit integers
    test_32 = np.frombuffer(raw_bytes, dtype=np.int32)
    max_val = np.max(np.abs(test_32))
    
    # 24-bit ADCs max at ~8.3 million. Bit-shifted 16-bit spikes > 50 million.
    if max_val > 20_000_000:
        return 'int16'
    else:
        return 'int32'


def _load_neuroscan_cnt(file_path: Path) -> mne.io.Raw:
    """
    Attempts standard MNE load. If corrupted, auto-detects the byte depth 
    and brute-forces the binary extraction.
    """
    try:
        return mne.io.read_raw_cnt(file_path, preload=True, verbose=False)
    
    except (OverflowError, RuntimeError) as e:
        logger.warning(f"Standard parser failed ({type(e).__name__}). Automating byte-depth detection.")
        
        # 1. Force int16 purely to extract valid channel metadata
        raw_header = mne.io.read_raw_cnt(file_path, data_format='int16', preload=False, verbose=False)
        info = raw_header.info
        n_chan = info['nchan']
        
        # 2. Calculate true start of biological data
        setup_bytes = 900
        electrode_bytes = n_chan * 75
        data_start_offset = setup_bytes + electrode_bytes
        
        # 3. Auto-detect data format
        fmt = _detect_byte_depth(file_path, n_chan, data_start_offset)
        logger.info(f"Auto-detected data format: {fmt}")
        
        bytes_per_sample = 4 if fmt == 'int32' else 2
        dtype = np.int32 if fmt == 'int32' else np.int16
        
        # 4. Calculate true geometry from physical file size
        usable_bytes = os.path.getsize(file_path) - data_start_offset
        frames = (usable_bytes // bytes_per_sample) // n_chan
        
        # 5. Read binary payload, skipping both header blocks
        with open(file_path, 'rb') as f:
            f.seek(data_start_offset)
            raw_bytes = np.fromfile(f, dtype=dtype, count=frames * n_chan)
            
        data_array = raw_bytes.reshape(frames, n_chan).T
        data_array_volts = np.zeros_like(data_array, dtype=np.float64)
        
        # 6. Apply channel-specific calibration factors
        for i, ch in enumerate(info['chs']):
            cal = ch.get('cal', 1e-6)
            data_array_volts[i, :] = data_array[i, :] * cal
            
        raw = mne.io.RawArray(data_array_volts, info, verbose=False)
        logger.info(f"Binary extraction successful using {fmt}.")
        
        return raw