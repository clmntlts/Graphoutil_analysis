"""
Pause detection utilities for handwriting analysis.
Based on Critten (2023) methodology.
"""

import numpy as np
from itertools import groupby


def min_duration_mask(mask, min_samples):
    """
    Filtre les segments True dans mask pour ne garder que ceux >= min_samples
    
    Args:
        mask: Boolean array
        min_samples: Minimum number of consecutive True values to keep
        
    Returns:
        Filtered boolean array
    """
    result = np.zeros_like(mask, dtype=bool)
    for val, grp in groupby(enumerate(mask), key=lambda x: x[1]):
        if val:
            indices = [i for i, _ in grp]
            if len(indices) >= min_samples:
                result[indices] = True
    return result


def detect_pauses(t, speed, pressure, speed_threshold, min_samples):
    """
    Détecte les pauses selon critères de Critten (2023):
    - Durée >= 20ms
    - Vitesse faible (< speed_threshold) OU pression = 0
    
    Args:
        t: Time series (pandas Series)
        speed: Speed array
        pressure: Pressure series
        speed_threshold: Threshold for low speed (px/s)
        min_samples: Minimum number of samples for a pause
        
    Returns:
        pause_mask_filtered: Boolean mask of pauses
        pauses: List of pause dictionaries with details
    """
    # Deux types de pauses
    zero_pressure = pressure == 0
    low_speed = speed < speed_threshold
    
    # Pause = vitesse faible OU pression nulle
    pause_mask = zero_pressure | low_speed
    
    # Filtrer par durée minimale
    pause_mask_filtered = min_duration_mask(pause_mask, min_samples)
    
    # Extraire les segments de pause
    pauses = []
    diff_mask = np.diff(np.concatenate([[False], pause_mask_filtered, [False]]).astype(int))
    starts = np.where(diff_mask == 1)[0]
    ends = np.where(diff_mask == -1)[0]
    
    for start_idx, end_idx in zip(starts, ends):
        duration_s = t.iloc[end_idx - 1] - t.iloc[start_idx]
        duration_ms = duration_s * 1000
        
        # Déterminer le type de pause
        segment_pressure = pressure.iloc[start_idx:end_idx]
        segment_speed = speed[start_idx:end_idx]
        
        if (segment_pressure == 0).all():
            pause_type = "pen_lift"
        elif (segment_pressure > 0).all() and (segment_speed < speed_threshold).all():
            pause_type = "low_speed"
        else:
            pause_type = "mixed"
        
        pauses.append({
            'start_time': t.iloc[start_idx],
            'end_time': t.iloc[end_idx - 1],
            'start_idx': start_idx,
            'end_idx': end_idx - 1,
            'duration_ms': duration_ms,
            'type': pause_type,
            'mean_pressure': segment_pressure.mean(),
            'mean_speed': segment_speed.mean()
        })
    
    return pause_mask_filtered, pauses
