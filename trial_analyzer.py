"""
Trial-level analysis module.
Computes comprehensive statistics for each trial.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging

from pause_detection import detect_pauses

logger = logging.getLogger(__name__)


class TrialAnalyzer:
    """
    Analyzes individual trials and computes comprehensive statistics
    """
    
    def __init__(self, config):
        """
        Initialize TrialAnalyzer
        
        Args:
            config: AnalysisConfig object
        """
        self.config = config
    
    def analyze_all_trials(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Analyze all trials in the dataset
        
        Args:
            df: DataFrame with Trial column
            
        Returns:
            Tuple of (summary_df, all_pauses_list)
        """
        trials_summary = []
        all_pauses = []
        
        for trial_id, trial in df.groupby("Trial"):
            try:
                summary, pauses = self.analyze_trial(trial, trial_id)
                trials_summary.append(summary)
                all_pauses.extend(pauses)
            except Exception as e:
                logger.error(f"Error analyzing trial {trial_id}: {e}")
                continue
        
        summary_df = pd.DataFrame(trials_summary)
        logger.info(f"Analyzed {len(trials_summary)} trials")
        
        return summary_df, all_pauses
    
    def analyze_trial(self, trial: pd.DataFrame, trial_id: int) -> Tuple[Dict, List[Dict]]:
        """
        Analyze a single trial
        
        Args:
            trial: Trial data
            trial_id: Trial identifier
            
        Returns:
            Tuple of (summary_dict, pauses_list)
        """
        trial = trial.copy().reset_index(drop=True)
        t = (trial["PacketTime"] - trial["PacketTime"].iloc[0]) / 1000  # seconds
        p = trial["NormalPressure"].to_numpy()
        
        # Check for empty trial
        nonzero_idx = np.where(p > 0)[0]
        if len(nonzero_idx) == 0:
            logger.warning(f"Trial {trial_id} has no pressure data")
            return self._empty_summary(trial_id), []
        
        # Temporal features
        onset = nonzero_idx[0]
        offset = nonzero_idx[-1]
        rt = t.iloc[onset]
        writing_duration = t.iloc[offset] - t.iloc[onset]
        total_duration = t.iloc[-1]
        
        # Kinematic features
        vx = np.gradient(trial["X"], t)
        vy = np.gradient(trial["Y"], t)
        speed = np.sqrt(vx**2 + vy**2)
        
        # Detect pauses
        pause_mask, pauses = detect_pauses(
            t, speed, trial["NormalPressure"],
            self.config.speed_threshold_px_per_s,
            self.config.min_pause_samples
        )
        
        # Add trial_id to pauses
        for pause in pauses:
            pause['trial'] = trial_id
        
        # Pause statistics
        total_pause_duration = sum([p['duration_ms'] for p in pauses]) / 1000
        num_pauses = len(pauses)
        pause_rate = num_pauses / writing_duration if writing_duration > 0 else 0
        
        # Spatial features
        path_length = np.sum(np.sqrt(np.diff(trial["X"])**2 + np.diff(trial["Y"])**2))
        straight_line = np.sqrt(
            (trial["X"].iloc[-1] - trial["X"].iloc[0])**2 +
            (trial["Y"].iloc[-1] - trial["Y"].iloc[0])**2
        )
        linearity = straight_line / path_length if path_length > 0 else np.nan
        
        width = trial["X"].max() - trial["X"].min()
        height = trial["Y"].max() - trial["Y"].min()
        area = width * height
        
        # Acceleration features
        ax = np.gradient(vx, t)
        ay = np.gradient(vy, t)
        acceleration = np.sqrt(ax**2 + ay**2)
        
        # Jerk features
        jx = np.gradient(ax, t)
        jy = np.gradient(ay, t)
        jerk = np.sqrt(jx**2 + jy**2)
        
        # Pressure features
        mean_pressure = p[p > 0].mean() if len(p[p > 0]) > 0 else 0
        pressure_variability = p[p > 0].std() if len(p[p > 0]) > 1 else 0
        
        summary = {
            "Trial": trial_id,
            
            # Temporal
            "RT_s": round(rt, 3),
            "WritingDuration_s": round(writing_duration, 3),
            "TotalDuration_s": round(total_duration, 3),
            
            # Pauses
            "NumPauses": num_pauses,
            "TotalPauseDuration_s": round(total_pause_duration, 3),
            "PauseRate_per_s": round(pause_rate, 3),
            "MeanPauseDuration_ms": round(np.mean([p['duration_ms'] for p in pauses]), 2) if pauses else 0,
            
            # Kinematics
            "MeanSpeed": round(np.nanmean(speed), 2),
            "MaxSpeed": round(np.nanmax(speed), 2),
            "SpeedSD": round(np.nanstd(speed), 2),
            "MeanAcceleration": round(np.nanmean(acceleration), 2),
            "MeanJerk": round(np.nanmean(jerk), 2),
            
            # Spatial
            "PathLength": round(path_length, 2),
            "StraightLine": round(straight_line, 2),
            "Linearity": round(linearity, 3),
            "Width": round(width, 1),
            "Height": round(height, 1),
            "Area": round(area, 1),
            
            # Pressure
            "MeanPressure": round(mean_pressure, 3),
            "PressureSD": round(pressure_variability, 3),
            
            # Efficiency metrics
            "WritingEfficiency": round((writing_duration - total_pause_duration) / writing_duration, 3) if writing_duration > 0 else 0,
            "SpatialEfficiency": round(linearity, 3),
        }
        
        return summary, pauses
    
    def _empty_summary(self, trial_id: int) -> Dict:
        """Return empty summary for invalid trials"""
        return {
            "Trial": trial_id,
            "RT_s": np.nan,
            "WritingDuration_s": np.nan,
            "TotalDuration_s": np.nan,
            "NumPauses": 0,
            "TotalPauseDuration_s": np.nan,
            "PauseRate_per_s": np.nan,
            "MeanPauseDuration_ms": np.nan,
            "MeanSpeed": np.nan,
            "MaxSpeed": np.nan,
            "SpeedSD": np.nan,
            "MeanAcceleration": np.nan,
            "MeanJerk": np.nan,
            "PathLength": np.nan,
            "StraightLine": np.nan,
            "Linearity": np.nan,
            "Width": np.nan,
            "Height": np.nan,
            "Area": np.nan,
            "MeanPressure": np.nan,
            "PressureSD": np.nan,
            "WritingEfficiency": np.nan,
            "SpatialEfficiency": np.nan,
        }


def analyze_trials(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Convenience function for trial analysis
    
    Args:
        df: DataFrame with Trial column
        config: AnalysisConfig object
        
    Returns:
        Tuple of (summary_df, all_pauses_list)
    """
    analyzer = TrialAnalyzer(config)
    return analyzer.analyze_all_trials(df)
