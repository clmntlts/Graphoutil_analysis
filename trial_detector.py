"""
Trial detection using multi-objective optimization.
Detects trial boundaries in continuous handwriting data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def robust_z(x: pd.Series) -> pd.Series:
    """
    Calculate robust z-score using median absolute deviation
    
    Args:
        x: Input series
        
    Returns:
        Robust z-scores
    """
    med = x.median()
    mad = np.median(np.abs(x - med))
    return (x - med) / (mad + 1e-6)


class TrialDetector:
    """
    Detects trial boundaries using multi-objective optimization
    """
    
    def __init__(self, config):
        """
        Initialize TrialDetector
        
        Args:
            config: AnalysisConfig object
        """
        self.config = config
        
        if config.n_trials is None:
            raise ValueError("n_trials must be specified in config")
    
    def detect_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect trial boundaries and add Trial column to dataframe
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Trial column added
        """
        logger.info(f"Detecting {self.config.n_trials} trials...")
        
        # Add temporal and spatial features
        df = self._add_features(df)
        
        # Find candidate boundaries
        candidates = self._find_candidates(df)
        
        # Optimize boundary selection
        boundary_idx = self._optimize_boundaries(candidates)
        
        # Add trial labels
        df["NewTrial"] = False
        df.loc[boundary_idx, "NewTrial"] = True
        df["Trial"] = df["NewTrial"].cumsum()
        
        n_detected = df["Trial"].max() + 1
        logger.info(f"Successfully detected {n_detected} trials")
        
        if n_detected != self.config.n_trials:
            logger.warning(f"Expected {self.config.n_trials} trials but found {n_detected}")
        
        return df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and spatial features for boundary detection"""
        # Temporal gap
        df["DeltaT"] = df["PacketTime"].diff().fillna(self.config.expected_interval)
        
        # Spatial jump
        dx = df["X"].diff().fillna(0)
        dy = df["Y"].diff().fillna(0)
        df["DistJump"] = np.sqrt(dx**2 + dy**2)
        
        # Previous pressure
        df["PressurePrev"] = df["NormalPressure"].shift(1).fillna(0)
        
        return df
    
    def _find_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find candidate trial boundaries
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame of candidates with scores
        """
        # Select points with temporal gaps
        candidates = df[df["DeltaT"] > self.config.min_gap_ms].copy()
        
        # Calculate multi-objective score
        candidates["Score"] = (
            robust_z(candidates["DeltaT"]) * 2.0 +           # Temporal gap
            robust_z(candidates["DistJump"]) * 1.0 +         # Spatial jump
            robust_z(-candidates["PressurePrev"]) * 1.5      # Pressure drop
        )
        
        # Sort by score
        candidates = candidates.sort_values("Score", ascending=False)
        
        logger.info(f"Found {len(candidates)} candidate boundaries")
        return candidates
    
    def _optimize_boundaries(self, candidates: pd.DataFrame) -> List[int]:
        """
        Optimize boundary selection using multi-objective cost function
        
        Args:
            candidates: DataFrame of candidate boundaries
            
        Returns:
            List of selected boundary indices
        """
        def evaluate(k: int) -> Tuple[float, List[int], int]:
            """Evaluate k top candidates"""
            selected = candidates.head(k).index.sort_values()
            
            filtered = []
            last = -np.inf
            violations = 0
            
            for idx in selected:
                if idx - last >= self.config.min_sep_samples:
                    filtered.append(idx)
                    last = idx
                else:
                    violations += 1
            
            n_trials_found = len(filtered) + 1
            score_mean = candidates.loc[selected, "Score"].mean()
            
            # Multi-objective cost
            cost = (
                1000 * abs(n_trials_found - self.config.n_trials) +  # Hard constraint
                -10 * score_mean +                                    # Quality
                50 * violations                                       # Separation
            )
            
            return cost, filtered, n_trials_found
        
        # Search for optimal k
        best_solution = None
        best_cost = np.inf
        max_k = min(len(candidates), self.config.n_trials * 3)
        
        for k in range(self.config.n_trials - 1, max_k):
            cost, idxs, n_found = evaluate(k)
            
            if cost < best_cost:
                best_cost = cost
                best_solution = idxs
            
            # Early stopping if perfect solution found
            if n_found == self.config.n_trials and cost < 0:
                break
        
        if best_solution is None or len(best_solution) + 1 != self.config.n_trials:
            raise RuntimeError(
                f"Could not find exactly {self.config.n_trials} trials. "
                f"Best solution found {len(best_solution) + 1 if best_solution else 0} trials."
            )
        
        return best_solution
    
    def visualize_detection(self, df: pd.DataFrame, save_path: str = None):
        """
        Visualize detected trial boundaries
        
        Args:
            df: DataFrame with Trial column
            save_path: Optional path to save figure
        """
        boundary_idx = df[df["NewTrial"]].index
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot temporal gaps
        ax1.plot(df["PacketTime"], df["DeltaT"], alpha=0.7)
        ax1.scatter(df.loc[boundary_idx, "PacketTime"],
                   df.loc[boundary_idx, "DeltaT"], 
                   c="red", s=100, zorder=5, label="Detected boundaries")
        ax1.set_yscale("log")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Time Gap (ms)")
        ax1.set_title("Trial Boundary Detection - Temporal Gaps")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot spatial trajectory with trial colors
        colors = plt.cm.tab20.colors
        for trial_id in df["Trial"].unique():
            trial_data = df[df["Trial"] == trial_id]
            ax2.plot(trial_data["X"], trial_data["Y"],
                    color=colors[int(trial_id) % len(colors)],
                    linewidth=1.5, alpha=0.8)
        
        ax2.set_aspect(self.config.trajectory_aspect_ratio)
        ax2.set_xlabel("X (px)")
        ax2.set_ylabel("Y (px)")
        ax2.set_title(f"Detected Trials (n={df['Trial'].max() + 1})")
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved detection visualization to {save_path}")
        
        plt.show()


def detect_trials_auto(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Convenience function for trial detection
    
    Args:
        df: Input DataFrame
        config: AnalysisConfig object
        
    Returns:
        DataFrame with Trial column
    """
    detector = TrialDetector(config)
    return detector.detect_trials(df)
