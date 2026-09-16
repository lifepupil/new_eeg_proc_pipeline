# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:54:20 2026

@author: lifep
"""

import numpy as np

# Ensure arrays match trial dimensions (Tensorpac averages across trials)
# raw pac shape: (n_amp, n_pha), surrogates shape: (n_perm, n_amp, n_pha)
m_pac = p.pac.mean(-1) if p.pac.ndim == 3 else p.pac
m_surro = p.surrogates.mean(-1) if p.surrogates.ndim == 4 else p.surrogates

# Proportion of surrogates >= observed raw PAC
# Add +1 to numerator and denominator to avoid p-values of absolute zero
n_perm = m_surro.shape[0]
pval_uncorrected = (np.sum(m_surro >= m_pac[np.newaxis, ...], axis=0) + 1) / (
    n_perm + 1
)

print(
    f"Min uncorrected p-value: {pval_uncorrected.min():.6f} (at alpha=0.05 threshold)"
)