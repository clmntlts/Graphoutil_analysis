"""
Trial detection using multi-objective optimization with interactive validation.
Detects trial boundaries in continuous handwriting data from PsychoPy experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
from typing import List, Tuple, Dict
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
    Detects trial boundaries using timestamp jumps as primary criterion.
    PsychoPy start/stop commands create clear temporal gaps between trials.
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
        
        # Store detection results for interactive review
        self.candidates_info = None
        self.selected_boundaries = None
    
    def detect_trials(self, df: pd.DataFrame, interactive: bool = False, 
                     csv_save_path: str = None) -> pd.DataFrame:
        """
        Detect trial boundaries and add Trial column to dataframe
        
        Args:
            df: Input DataFrame
            interactive: If True, launch interactive validation interface
            csv_save_path: Optional path to save CSV file with boundary data
            
        Returns:
            DataFrame with Trial column added
        """
        logger.info(f"Detecting {self.config.n_trials} trials...")
        
        # Add temporal and spatial features
        df = self._add_features(df)
        
        # Find candidate boundaries with confidence scores
        candidates = self._find_candidates(df)
        
        # Optimize boundary selection
        boundary_idx = self._optimize_boundaries(candidates, df)

        # Store for interactive review  ✅ FIXED ORDER
        self.selected_boundaries = boundary_idx
        self.candidates_info = self._prepare_candidates_info(candidates, df)

        # Interactive validation if requested
        if interactive:
            boundary_idx = self._interactive_validation(df, candidates, boundary_idx, csv_save_path)
            self.selected_boundaries = boundary_idx

        # Add trial labels - ensure sequential numbering without gaps
        df["Trial"] = 0
        if len(boundary_idx) > 0:
            # Sort boundaries to ensure proper ordering
            boundary_idx_sorted = sorted(boundary_idx)
            
            # Assign trial numbers sequentially
            for trial_num, boundary in enumerate(boundary_idx_sorted, start=1):
                # All rows from this boundary onward belong to this trial (until next boundary)
                df.loc[boundary:, "Trial"] = trial_num
        
        # Verify sequential numbering
        unique_trials = sorted(df["Trial"].unique())
        expected_trials = list(range(len(boundary_idx) + 1))
        
        if unique_trials != expected_trials:
            logger.warning(f"Trial numbering issue detected. Expected {expected_trials}, got {unique_trials}")
            # Force sequential renumbering
            trial_mapping = {old: new for new, old in enumerate(unique_trials)}
            df["Trial"] = df["Trial"].map(trial_mapping)
        
        n_detected = len(df["Trial"].unique())
        logger.info(f"Successfully detected {n_detected} trials (Trial IDs: {sorted(df['Trial'].unique())})")
        
        if n_detected != self.config.n_trials:
            logger.warning(f"Expected {self.config.n_trials} trials but found {n_detected}")
        
        return df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and spatial features for boundary detection"""
        # Temporal gap (PRIMARY FEATURE for PsychoPy start/stop)
        df["DeltaT"] = df["PacketTime"].diff().fillna(self.config.expected_interval)
        
        # Spatial jump
        dx = df["X"].diff().fillna(0)
        dy = df["Y"].diff().fillna(0)
        df["DistJump"] = np.sqrt(dx**2 + dy**2)
        
        # Velocity (NEW FEATURE - distance / time)
        # Handle division by zero by setting a minimum time delta
        time_delta_sec = df["DeltaT"] / 1000.0  # Convert ms to seconds
        time_delta_sec = time_delta_sec.replace(0, np.nan)  # Avoid division by zero
        df["Velocity"] = df["DistJump"] / time_delta_sec  # pixels per second
        df["Velocity"] = df["Velocity"].fillna(0)  # Fill NaN with 0
        
        # Previous pressure
        df["PressurePrev"] = df["NormalPressure"].shift(1).fillna(0)
        
        # Current pressure (for detecting pen lifts)
        df["PressureCurr"] = df["NormalPressure"]
        
        return df
    
    def _find_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find candidate trial boundaries with confidence scores
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame of candidates with scores and confidence metrics
        """
        # PRIMARY: Select points with significant temporal gaps
        # PsychoPy stop/start creates gaps >> normal sampling interval
        typical_gap = self.config.expected_interval
        
        # Candidates must have gaps significantly larger than sampling interval
        min_trial_gap = max(self.config.min_gap_ms, typical_gap * 5)
        candidates = df[df["DeltaT"] > min_trial_gap].copy()
        
        if len(candidates) == 0:
            logger.warning(f"No candidates found with DeltaT > {min_trial_gap}ms")
            # Fallback to smaller gap
            min_trial_gap = self.config.min_gap_ms
            candidates = df[df["DeltaT"] > min_trial_gap].copy()
        
        # Calculate confidence scores for each candidate
        candidates = self._calculate_confidence_scores(candidates, df)
        
        # Sort by score
        candidates = candidates.sort_values("Score", ascending=False)
        
        logger.info(f"Found {len(candidates)} candidate boundaries (gap > {min_trial_gap:.1f}ms)")
        return candidates
    
    def _calculate_confidence_scores(self, candidates: pd.DataFrame, 
                                     df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate multi-component confidence score for each candidate
        
        Components:
        1. Temporal gap (PRIMARY - most important for PsychoPy)
        2. Velocity spike (SECONDARY - should be extreme between trials)
        3. Spatial jump (TERTIARY - supports temporal)
        4. Pressure change (QUATERNARY - additional evidence)
        """
        # Temporal score (most important - weighted 3x)
        temporal_z = robust_z(candidates["DeltaT"])
        
        # Velocity score (very important - weighted 2x)
        # Between trials, velocity should spike dramatically
        velocity_z = robust_z(candidates["Velocity"])
        
        # Spatial score (important - weighted 1x)
        spatial_z = robust_z(candidates["DistJump"])
        
        # Pressure drop score (supplementary - weighted 0.5x)
        pressure_z = robust_z(-candidates["PressurePrev"])
        
        # Combined score with temporal and velocity dominance
        candidates["Score"] = (
            temporal_z * 3.0 +      # Temporal gap (MOST IMPORTANT)
            velocity_z * 2.0 +      # Velocity spike (VERY IMPORTANT)
            spatial_z * 1.0 +       # Spatial jump
            pressure_z * 0.5        # Pressure drop
        )
        
        # Calculate individual confidence components (0-100 scale)
        candidates["TemporalConfidence"] = np.clip(
            (temporal_z / temporal_z.max() * 100) if temporal_z.max() > 0 else 0, 
            0, 100
        )
        candidates["VelocityConfidence"] = np.clip(
            (velocity_z / velocity_z.max() * 100) if velocity_z.max() > 0 else 0,
            0, 100
        )
        candidates["SpatialConfidence"] = np.clip(
            (spatial_z / spatial_z.max() * 100) if spatial_z.max() > 0 else 0,
            0, 100
        )
        candidates["PressureConfidence"] = np.clip(
            (pressure_z / pressure_z.max() * 100) if pressure_z.max() > 0 else 0,
            0, 100
        )
        
        # Overall confidence (normalized to 0-100)
        max_score = candidates["Score"].max()
        candidates["Confidence"] = np.clip(
            (candidates["Score"] / max_score * 100) if max_score > 0 else 0,
            0, 100
        )
        
        return candidates
    
    def _optimize_boundaries(self, candidates: pd.DataFrame, 
                            df: pd.DataFrame) -> List[int]:
        """
        Optimize boundary selection with adaptive trial size validation
        
        Strategy:
        1. Calculate expected trial sizes from ALL candidates
        2. Use median trial size as adaptive baseline
        3. Iteratively select boundaries that:
           - Have high confidence scores
           - Maintain minimum separation
           - Create trials of reasonable size (not outliers)
        4. Continue until we have exactly n_trials
        
        Args:
            candidates: DataFrame of candidate boundaries
            df: Full dataframe
            
        Returns:
            List of selected boundary indices
        """
        if len(candidates) < self.config.n_trials - 1:
            raise RuntimeError(
                f"Not enough candidates ({len(candidates)}) to create {self.config.n_trials} trials. "
                f"Need at least {self.config.n_trials - 1} boundaries."
            )
        
        # ADAPTIVE: Calculate trial sizes from ALL candidates to get distribution
        all_trial_sizes = self._calculate_trial_sizes_from_candidates(candidates, df)
        
        # Use median and percentiles for adaptive thresholding
        median_trial_size = np.median(all_trial_sizes)
        percentile_25 = np.percentile(all_trial_sizes, 25)
        percentile_10 = np.percentile(all_trial_sizes, 10)
        
        # Set minimum as 30% of median (adaptive to actual data)
        min_acceptable_size = median_trial_size * 0.30
        
        logger.info(f"ADAPTIVE TRIAL SIZE THRESHOLDS:")
        logger.info(f"  Median trial size from candidates: {median_trial_size:.0f} points")
        logger.info(f"  25th percentile: {percentile_25:.0f} points")
        logger.info(f"  10th percentile: {percentile_10:.0f} points")
        logger.info(f"  Minimum acceptable (30% of median): {min_acceptable_size:.0f} points")
        
        # Sort candidates by score (best first)
        candidates_sorted = candidates.sort_values("Score", ascending=False)
        
        # Greedy selection with separation AND trial size constraints
        selected = []
        skipped_count = 0
        
        for idx in candidates_sorted.index:
            # Check separation constraint
            if not all(abs(idx - s) >= self.config.min_sep_samples for s in selected):
                continue
            
            # Check trial size constraint (adaptive)
            test_boundaries = sorted(selected + [idx])
            if self._creates_small_trial(test_boundaries, df, min_acceptable_size):
                skipped_count += 1
                logger.debug(f"Skipping candidate at index {idx} - would create small trial (< {min_acceptable_size:.0f})")
                continue
            
            # This boundary passes all checks
            selected.append(idx)
            
            # Stop when we have enough boundaries
            if len(selected) == self.config.n_trials - 1:
                break
        
        logger.info(f"Skipped {skipped_count} candidates due to small trial size constraint")
        
        # Sort selected boundaries by position
        selected = sorted(selected)
        
        # If we don't have enough, try with relaxed constraints
        if len(selected) < self.config.n_trials - 1:
            logger.warning(
                f"Only found {len(selected)} boundaries with strict constraints. "
                f"Trying with relaxed trial size constraint (20% of median)..."
            )
            relaxed_min_size = median_trial_size * 0.20
            selected = self._relaxed_selection_with_size_check(
                candidates_sorted, df, relaxed_min_size
            )
        
        if len(selected) != self.config.n_trials - 1:
            logger.error(f"Could not find exactly {self.config.n_trials - 1} boundaries. Found {len(selected)}.")
            # Log trial sizes with current selection
            self._log_trial_sizes(selected, df)
            raise RuntimeError(
                f"Could not find exactly {self.config.n_trials} trials. "
                f"Found {len(selected) + 1} trials. Check your data or adjust n_trials parameter."
            )
        
        # Final validation
        logger.info(f"Selected {len(selected)} boundaries")
        self._log_trial_sizes(selected, df)
        
        return selected
    
    def _calculate_trial_sizes_from_candidates(self, candidates: pd.DataFrame, 
                                               df: pd.DataFrame) -> List[float]:
        """
        Calculate what trial sizes would be if we used different subsets of candidates.
        This gives us the distribution of expected trial sizes.
        
        Args:
            candidates: DataFrame of candidate boundaries
            df: Full dataframe
            
        Returns:
            List of trial sizes from sampling different candidate combinations
        """
        all_sizes = []
        
        # Method 1: Use all candidates sorted by position
        all_candidate_indices = sorted(candidates.index.tolist())
        trial_starts = [df.index[0]] + all_candidate_indices
        trial_ends = all_candidate_indices + [df.index[-1]]
        
        for start, end in zip(trial_starts, trial_ends):
            start_pos = df.index.get_loc(start) if start in df.index else 0
            end_pos = df.index.get_loc(end) if end in df.index else len(df) - 1
            size = end_pos - start_pos + 1
            all_sizes.append(size)
        
        # Method 2: Sample top N candidates (most likely to be selected)
        n_needed = self.config.n_trials - 1
        if len(candidates) >= n_needed:
            top_candidates = candidates.nlargest(n_needed, 'Score').index.tolist()
            top_sorted = sorted(top_candidates)
            trial_starts = [df.index[0]] + top_sorted
            trial_ends = top_sorted + [df.index[-1]]
            
            for start, end in zip(trial_starts, trial_ends):
                start_pos = df.index.get_loc(start) if start in df.index else 0
                end_pos = df.index.get_loc(end) if end in df.index else len(df) - 1
                size = end_pos - start_pos + 1
                all_sizes.append(size)
        
        # Method 3: Simple expected size
        expected_size = len(df) / self.config.n_trials
        all_sizes.append(expected_size)
        
        return all_sizes
    
    def _creates_small_trial(self, boundaries: List[int], df: pd.DataFrame, 
                            min_size: float) -> bool:
        """
        Check if a set of boundaries would create any trials that are too small
        
        Args:
            boundaries: Sorted list of boundary indices
            df: Full dataframe
            min_size: Minimum acceptable trial size
            
        Returns:
            True if any trial would be smaller than min_size
        """
        if len(boundaries) == 0:
            return False
        
        # Calculate trial sizes
        trial_starts = [df.index[0]] + boundaries
        trial_ends = boundaries + [df.index[-1]]
        
        for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
            # Get positions in dataframe
            start_pos = df.index.get_loc(start) if start in df.index else 0
            end_pos = df.index.get_loc(end) if end in df.index else len(df) - 1
            
            trial_size = end_pos - start_pos + 1
            
            if trial_size < min_size:
                return True
        
        return False
    
    def _log_trial_sizes(self, boundaries: List[int], df: pd.DataFrame):
        """Log the sizes of trials created by given boundaries"""
        if len(boundaries) == 0:
            logger.info(f"No boundaries - single trial of {len(df)} points")
            return
        
        trial_starts = [df.index[0]] + boundaries
        trial_ends = boundaries + [df.index[-1]]
        
        logger.info("Trial sizes with selected boundaries:")
        for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
            start_pos = df.index.get_loc(start) if start in df.index else 0
            end_pos = df.index.get_loc(end) if end in df.index else len(df) - 1
            trial_size = end_pos - start_pos + 1
            logger.info(f"  Trial {i}: {trial_size} points")
    
    def _relaxed_selection_with_size_check(self, candidates_sorted: pd.DataFrame, 
                                          df: pd.DataFrame, min_size: float) -> List[int]:
        """
        Relaxed boundary selection with size checking
        Uses reduced minimum separation and relaxed size constraint
        """
        relaxed_sep = self.config.min_sep_samples // 2
        selected = []
        
        for idx in candidates_sorted.index:
            # Check relaxed separation
            if not all(abs(idx - s) >= relaxed_sep for s in selected):
                continue
            
            # Check relaxed trial size constraint
            test_boundaries = sorted(selected + [idx])
            if self._creates_small_trial(test_boundaries, df, min_size):
                continue
            
            selected.append(idx)
            if len(selected) == self.config.n_trials - 1:
                break
        
        return sorted(selected)
    
    def _prepare_candidates_info(self, candidates: pd.DataFrame, 
                                 df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare detailed information about all candidates for visualization
        """
        info = candidates.copy()
        
        # Add context information
        info["Time_s"] = df.loc[info.index, "PacketTime"] / 1000
        info["X"] = df.loc[info.index, "X"]
        info["Y"] = df.loc[info.index, "Y"]
        info["IsSelected"] = info.index.isin(self.selected_boundaries)
        
        # Add rank
        info["Rank"] = range(1, len(info) + 1)
        
        return info
    
    def _interactive_validation(self, df: pd.DataFrame, 
                               candidates: pd.DataFrame,
                               initial_boundaries: List[int],
                               csv_save_path: str = None) -> List[int]:
        """
        Launch interactive interface for boundary validation and correction
        
        Args:
            df: DataFrame with features
            candidates: Candidate boundaries
            initial_boundaries: Initial boundary indices
            csv_save_path: Optional path for CSV export
            
        Returns:
            Updated list of boundary indices
        """
        validator = InteractiveTrialValidator(
            df, candidates, initial_boundaries, 
            self.config.n_trials, self.config, csv_save_path
        )
        
        return validator.run()
    
    def visualize_detection(self, df: pd.DataFrame, save_path: str = None):
        """
        Visualize detected trial boundaries
        
        Args:
            df: DataFrame with Trial column
            save_path: Optional path to save figure
        """
        # Get boundary indices from selected_boundaries
        boundary_idx = self.selected_boundaries if self.selected_boundaries else []
        
        # Diagnostic: Check for trial number gaps
        unique_trials = sorted(df["Trial"].unique())
        expected_trials = list(range(len(unique_trials)))
        if unique_trials != expected_trials:
            logger.warning(f"⚠ Trial numbering has gaps! Found trials: {unique_trials}, Expected: {expected_trials}")
            print(f"\n⚠ WARNING: Trial numbering has gaps!")
            print(f"   Found trials: {unique_trials}")
            print(f"   Expected: {expected_trials}")
            print(f"   Missing trials: {set(expected_trials) - set(unique_trials)}\n")
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 14))
        
        # Plot 1: Temporal gaps (log scale)
        ax1 = axes[0]
        ax1.plot(df["PacketTime"], df["DeltaT"], alpha=0.5, linewidth=0.5)
        
        if len(boundary_idx) > 0:
            ax1.scatter(df.loc[boundary_idx, "PacketTime"],
                       df.loc[boundary_idx, "DeltaT"], 
                       c="red", s=100, zorder=5, label="Selected boundaries", marker='o')
        
        # Show all candidates if available
        if self.candidates_info is not None:
            other_candidates = self.candidates_info[~self.candidates_info["IsSelected"]]
            ax1.scatter(other_candidates["Time_s"] * 1000,
                       df.loc[other_candidates.index, "DeltaT"],
                       c="orange", s=50, alpha=0.5, zorder=4, 
                       label="Other candidates", marker='x')
        
        ax1.axhline(self.config.min_gap_ms, color='green', 
                   linestyle='--', alpha=0.5, label=f"Min gap ({self.config.min_gap_ms}ms)")
        ax1.set_yscale("log")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Time Gap (ms)")
        ax1.set_title("Trial Boundary Detection - Temporal Gaps")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Velocity spikes
        ax2 = axes[1]
        ax2.plot(df["PacketTime"], df["Velocity"], alpha=0.5, linewidth=0.5, color='purple')
        if len(boundary_idx) > 0:
            ax2.scatter(df.loc[boundary_idx, "PacketTime"],
                       df.loc[boundary_idx, "Velocity"], 
                       c="red", s=100, zorder=5, label="Selected boundaries", marker='o')
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Velocity (px/s)")
        ax2.set_title("Velocity Spikes at Boundaries")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Spatial jumps
        ax3 = axes[2]
        ax3.plot(df["PacketTime"], df["DistJump"], alpha=0.5, linewidth=0.5, color='blue')
        if len(boundary_idx) > 0:
            ax3.scatter(df.loc[boundary_idx, "PacketTime"],
                       df.loc[boundary_idx, "DistJump"], 
                       c="red", s=100, zorder=5, label="Selected boundaries", marker='o')
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Spatial Jump (px)")
        ax3.set_title("Spatial Jumps at Boundaries")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Spatial trajectory with trial colors - only where pressure > 0
        ax4 = axes[3]
        colors = plt.cm.tab20.colors
        for trial_id in df["Trial"].unique():
            trial_data = df[df["Trial"] == trial_id]
            
            # Only plot segments where pressure > 0
            pressure_mask = trial_data["NormalPressure"] > 0
            
            # Find continuous segments with pressure
            pressure_changes = pressure_mask.astype(int).diff().fillna(0)
            segment_starts = trial_data.index[pressure_changes == 1].tolist()
            segment_ends = trial_data.index[pressure_changes == -1].tolist()
            
            # Handle edge cases
            if pressure_mask.iloc[0]:
                segment_starts.insert(0, trial_data.index[0])
            if pressure_mask.iloc[-1]:
                segment_ends.append(trial_data.index[-1])
            
            # Plot each continuous segment
            for start_idx, end_idx in zip(segment_starts, segment_ends):
                segment = trial_data.loc[start_idx:end_idx]
                ax4.plot(segment["X"], segment["Y"],
                        color=colors[int(trial_id) % len(colors)],
                        linewidth=1.5, alpha=0.8, label=f"Trial {trial_id}" if start_idx == segment_starts[0] else "")
        
        ax4.set_aspect(self.config.trajectory_aspect_ratio)
        ax4.set_xlabel("X (px)")
        ax4.set_ylabel("Y (px)")
        n_trials_actual = len(df['Trial'].unique())
        ax4.set_title(f"Detected Trials (n={n_trials_actual})")
        ax4.grid(True, alpha=0.3)
        
        # Only show legend if not too many trials
        if n_trials_actual < 15:
            ax4.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved detection visualization to {save_path}")
        
        plt.show()
    
    def print_candidate_report(self):
        """Print detailed report of all candidates with confidence scores"""
        if self.candidates_info is None:
            print("No candidate information available. Run detect_trials() first.")
            return
        
        print("\n" + "="*80)
        print("TRIAL BOUNDARY CANDIDATES REPORT")
        print("="*80)
        print(f"Total candidates found: {len(self.candidates_info)}")
        print(f"Required boundaries: {self.config.n_trials - 1}")
        print(f"Selected boundaries: {self.candidates_info['IsSelected'].sum()}")
        print("="*80)
        
        # Show top candidates
        display_cols = ['Rank', 'Time_s', 'DeltaT', 'Velocity', 'DistJump', 
                       'Confidence', 'TemporalConfidence', 'VelocityConfidence',
                       'SpatialConfidence', 'IsSelected']
        
        print("\nTop 20 Candidates (sorted by confidence):")
        print("-"*80)
        report = self.candidates_info[display_cols].head(20).copy()
        report['Time_s'] = report['Time_s'].round(2)
        report['DeltaT'] = report['DeltaT'].round(1)
        report['Velocity'] = report['Velocity'].round(1)
        report['DistJump'] = report['DistJump'].round(1)
        report['Confidence'] = report['Confidence'].round(1)
        report['TemporalConfidence'] = report['TemporalConfidence'].round(1)
        report['VelocityConfidence'] = report['VelocityConfidence'].round(1)
        report['SpatialConfidence'] = report['SpatialConfidence'].round(1)
        
        print(report.to_string(index=False))
        print("="*80 + "\n")
    
    def diagnose_trial_numbering(self, df: pd.DataFrame):
        """
        Diagnose trial numbering issues
        
        Args:
            df: DataFrame with Trial column
        """
        print("\n" + "="*80)
        print("TRIAL NUMBERING DIAGNOSTICS")
        print("="*80)
        
        unique_trials = sorted(df["Trial"].unique())
        n_unique = len(unique_trials)
        expected_trials = list(range(n_unique))
        
        print(f"Expected number of trials: {self.config.n_trials}")
        print(f"Detected unique trials: {n_unique}")
        print(f"Trial IDs found: {unique_trials}")
        print(f"Expected sequential IDs: {expected_trials}")
        
        # Check for gaps
        gaps = set(expected_trials) - set(unique_trials)
        if gaps:
            print(f"\n⚠ GAPS DETECTED: Missing trial IDs: {sorted(gaps)}")
        else:
            print(f"\n✓ No gaps in trial numbering")
        
        # Check trial sizes
        print("\nTrial sizes (number of data points per trial):")
        trial_counts = df.groupby("Trial").size().sort_index()
        for trial_id, count in trial_counts.items():
            print(f"  Trial {trial_id}: {count} points")
        
        # Check boundary indices
        if self.selected_boundaries:
            print(f"\nSelected boundary indices: {self.selected_boundaries}")
            print(f"Number of boundaries: {len(self.selected_boundaries)}")
        
        print("="*80 + "\n")


class InteractiveTrialValidator:
    """
    Interactive interface for validating and correcting trial boundaries
    """
    
    def __init__(self, df: pd.DataFrame, candidates: pd.DataFrame,
                 initial_boundaries: List[int], n_trials: int, config, 
                 csv_save_path: str = None):
        self.df = df
        self.candidates = candidates.sort_values("Score", ascending=False)
        self.boundaries = sorted(initial_boundaries)
        self.initial_boundaries = sorted(initial_boundaries)  # Store initial for reset
        self.n_trials = n_trials
        self.config = config
        
        # For manual point selection
        self.manual_points = []
        
        # Scrolling state - show 1/10 of time range at once
        self.time_min = self.df["PacketTime"].min()
        self.time_max = self.df["PacketTime"].max()
        self.time_range = self.time_max - self.time_min
        self.window_size = self.time_range / 10  # Show 1/10 of data
        self.view_start = self.time_min
        self.view_end = self.view_start + self.window_size
        
        # For CSV export
        self.csv_save_path = csv_save_path
        
    def run(self) -> List[int]:
        """Launch interactive interface and return validated boundaries"""
        # Create figure with 4 rows and 5 columns
        self.fig = plt.figure(figsize=(18, 12))
        gs = self.fig.add_gridspec(5, 5, 
                                  height_ratios=[2, 2, 0.5, 0.5, 0.5],  # Upper 2 rows, then 3 aligned lower rows
                                  width_ratios=[1, 1, 1, 1, 0.9],  # 4 cols for plots, 1 for list
                                  hspace=0.08, wspace=0.3)
        
        # Rows 1-2, Columns 1-4: Spatial trajectory (upper 2/3)
        self.ax_spatial = self.fig.add_subplot(gs[0:2, 0:4])
        
        # Row 3, Columns 1-4: Temporal gaps (lower 1/3, top)
        self.ax_temporal = self.fig.add_subplot(gs[2, 0:4])
        
        # Row 4, Columns 1-4: Velocity (lower 1/3, middle, shared x-axis)
        self.ax_velocity = self.fig.add_subplot(gs[3, 0:4], sharex=self.ax_temporal)
        
        # Row 5, Columns 1-4: Spatial jumps (lower 1/3, bottom, shared x-axis)
        self.ax_jumps = self.fig.add_subplot(gs[4, 0:4], sharex=self.ax_temporal)
        
        # Rows 1-5, Column 5: Candidate list (all rows on right side)
        self.ax_candidates = self.fig.add_subplot(gs[0:5, 4])
        self.ax_candidates.axis('off')
        
        # Initial plots
        self._plot_all()
        
        # Connect events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.cid_close = self.fig.canvas.mpl_connect('close_event', self._on_close)
        
        plt.suptitle(f"Interactive Trial Boundary Validation - Need {self.n_trials-1} boundaries for {self.n_trials} trials\n"
                    f"[Scroll wheel or ← → to navigate | Click temporal plot to toggle boundaries | 'h' for help]",
                    fontsize=11, weight='bold', y=0.995)
        
        print("\n" + "="*70)
        print("INTERACTIVE VALIDATION MODE")
        print("="*70)
        print("NAVIGATION:")
        print("  • Scroll wheel or arrow keys (← →) to pan timeline")
        print("  • Click on the temporal gap plot to toggle boundaries")
        print("\nCOMMANDS:")
        print("  • Press 'a' to auto-select top N candidates")
        print("  • Press 'r' to reset to initial selection") 
        print("  • Press 'c' to clear all boundaries")
        print("  • Press 'h' to view full timeline (reset zoom)")
        print("  • Press 's' to save CSV now (or close window to auto-save)")
        print("  • Close the window when satisfied")
        print("\nNOTE: Raw data with boundaries will be saved to CSV on close")
        print("="*70)
        print(f"Viewing window: {self.window_size/1000:.1f}s ({100/10:.0f}% of total)")
        print("="*70 + "\n")
        
        plt.show()
        
        return sorted(self.boundaries)
    
    def _plot_all(self):
        """Refresh all plots"""
        self._plot_spatial()
        self._plot_temporal()
        self._plot_velocity()
        self._plot_jumps()
        self._plot_candidates_list()
        self._update_status()
        self.fig.canvas.draw_idle()  # Use draw_idle for better performance
    
    def _plot_spatial(self):
        """Plot spatial trajectory colored by trial - only when pressure > 0"""
        self.ax_spatial.clear()
        
        # Create temporary trial labels based on current boundaries
        temp_df = self.df.copy()
        temp_df["TempTrial"] = 0
        for i, boundary in enumerate(sorted(self.boundaries)):
            temp_df.loc[boundary:, "TempTrial"] = i + 1
        
        # Plot each trial
        colors = plt.cm.tab20.colors
        for trial_id in temp_df["TempTrial"].unique():
            trial_data = temp_df[temp_df["TempTrial"] == trial_id]
            
            # Only plot segments where pressure > 0
            # Find continuous segments with pressure
            pressure_mask = trial_data["NormalPressure"] > 0
            
            # Find transitions to identify continuous segments
            pressure_changes = pressure_mask.astype(int).diff().fillna(0)
            segment_starts = trial_data.index[pressure_changes == 1].tolist()
            segment_ends = trial_data.index[pressure_changes == -1].tolist()
            
            # Handle edge cases
            if pressure_mask.iloc[0]:
                segment_starts.insert(0, trial_data.index[0])
            if pressure_mask.iloc[-1]:
                segment_ends.append(trial_data.index[-1])
            
            # Plot each continuous segment
            for start_idx, end_idx in zip(segment_starts, segment_ends):
                segment = trial_data.loc[start_idx:end_idx]
                self.ax_spatial.plot(segment["X"], segment["Y"],
                                    color=colors[int(trial_id) % len(colors)],
                                    linewidth=1.5, alpha=0.8)
        
        # Respect aspect ratio from config
        self.ax_spatial.set_aspect(self.config.trajectory_aspect_ratio)
        self.ax_spatial.set_xlabel('X (px)', fontsize=10)
        self.ax_spatial.set_ylabel('Y (px)', fontsize=10)
        
        # Status indicator
        n_current = len(self.boundaries) + 1
        n_needed = self.n_trials
        status = "✓ CORRECT" if n_current == n_needed else f"⚠ NEED {n_needed}"
        color = 'green' if n_current == n_needed else 'red'
        
        self.ax_spatial.set_title(
            f'Spatial Trajectory: {n_current} trials detected {status}',
            fontsize=11, weight='bold', color=color
        )
        self.ax_spatial.grid(True, alpha=0.3)
    
    def _plot_temporal(self):
        """Plot temporal gaps with selected boundaries - scrollable view"""
        self.ax_temporal.clear()
        
        # Filter data to current view window
        mask = (self.df["PacketTime"] >= self.view_start) & (self.df["PacketTime"] <= self.view_end)
        view_df = self.df[mask]
        
        if len(view_df) == 0:
            self.ax_temporal.text(0.5, 0.5, 'No data in view', 
                                 transform=self.ax_temporal.transAxes,
                                 ha='center', va='center')
            return
        
        # Plot all gaps in view
        self.ax_temporal.plot(view_df["PacketTime"], view_df["DeltaT"], 
                             alpha=0.6, linewidth=1, color='gray', label='Temporal gaps')
        
        # Mark all candidates in view
        candidates_in_view = self.candidates[
            (self.df.loc[self.candidates.index, "PacketTime"] >= self.view_start) &
            (self.df.loc[self.candidates.index, "PacketTime"] <= self.view_end)
        ]
        
        if len(candidates_in_view) > 0:
            self.ax_temporal.scatter(
                self.df.loc[candidates_in_view.index, "PacketTime"],
                self.df.loc[candidates_in_view.index, "DeltaT"],
                c='orange', s=50, alpha=0.6, label='Candidates', marker='x', linewidths=2
            )
        
        # Mark selected boundaries in view
        boundaries_in_view = [b for b in self.boundaries 
                             if self.view_start <= self.df.loc[b, "PacketTime"] <= self.view_end]
        
        if boundaries_in_view:
            self.ax_temporal.scatter(
                self.df.loc[boundaries_in_view, "PacketTime"],
                self.df.loc[boundaries_in_view, "DeltaT"],
                c='red', s=120, zorder=5, label='Selected', marker='o', 
                edgecolors='black', linewidths=2.5
            )
        
        # Show threshold
        self.ax_temporal.axhline(self.config.min_gap_ms, color='green',
                                linestyle='--', alpha=0.5, linewidth=1.5,
                                label=f'Min gap ({self.config.min_gap_ms}ms)')
        
        self.ax_temporal.set_yscale('log')
        self.ax_temporal.set_xlim(self.view_start, self.view_end)
        self.ax_temporal.set_ylabel('Time Gap (ms)', fontsize=9)
        self.ax_temporal.set_title('Temporal Gaps (Click to toggle)', fontsize=10)
        self.ax_temporal.legend(loc='upper right', fontsize=8)
        self.ax_temporal.grid(True, alpha=0.3)
        
        # Remove x-axis labels (shared with velocity plot below)
        self.ax_temporal.tick_params(labelbottom=False)
    
    def _plot_velocity(self):
        """Plot velocity spikes - aligned with temporal plot"""
        self.ax_velocity.clear()
        
        # Filter data to current view window
        mask = (self.df["PacketTime"] >= self.view_start) & (self.df["PacketTime"] <= self.view_end)
        view_df = self.df[mask]
        
        if len(view_df) == 0:
            return
        
        # Plot velocity in view
        self.ax_velocity.plot(view_df["PacketTime"], view_df["Velocity"],
                             alpha=0.6, linewidth=1, color='purple', label='Velocity')
        
        # Mark selected boundaries in view
        boundaries_in_view = [b for b in self.boundaries 
                             if self.view_start <= self.df.loc[b, "PacketTime"] <= self.view_end]
        
        if boundaries_in_view:
            self.ax_velocity.scatter(
                self.df.loc[boundaries_in_view, "PacketTime"],
                self.df.loc[boundaries_in_view, "Velocity"],
                c='red', s=120, zorder=5, marker='o',
                edgecolors='black', linewidths=2.5, label='Selected'
            )
        
        self.ax_velocity.set_xlim(self.view_start, self.view_end)
        self.ax_velocity.set_ylabel('Velocity (px/s)', fontsize=9)
        self.ax_velocity.set_title('Velocity Spikes', fontsize=10)
        self.ax_velocity.legend(loc='upper right', fontsize=8)
        self.ax_velocity.grid(True, alpha=0.3)
        
        # Remove x-axis labels (shared with jumps plot below)
        self.ax_velocity.tick_params(labelbottom=False)
    
    def _plot_jumps(self):
        """Plot spatial jumps - aligned with temporal plot"""
        self.ax_jumps.clear()
        
        # Filter data to current view window
        mask = (self.df["PacketTime"] >= self.view_start) & (self.df["PacketTime"] <= self.view_end)
        view_df = self.df[mask]
        
        if len(view_df) == 0:
            return
        
        # Plot spatial jumps in view
        self.ax_jumps.plot(view_df["PacketTime"], view_df["DistJump"],
                          alpha=0.6, linewidth=1, color='blue', label='Spatial jumps')
        
        # Mark selected boundaries in view
        boundaries_in_view = [b for b in self.boundaries 
                             if self.view_start <= self.df.loc[b, "PacketTime"] <= self.view_end]
        
        if boundaries_in_view:
            self.ax_jumps.scatter(
                self.df.loc[boundaries_in_view, "PacketTime"],
                self.df.loc[boundaries_in_view, "DistJump"],
                c='red', s=120, zorder=5, marker='o',
                edgecolors='black', linewidths=2.5, label='Selected'
            )
        
        self.ax_jumps.set_xlim(self.view_start, self.view_end)
        self.ax_jumps.set_xlabel('Time (ms)', fontsize=9)
        self.ax_jumps.set_ylabel('Spatial Jump (px)', fontsize=9)
        self.ax_jumps.set_title('Spatial Jumps', fontsize=10)
        self.ax_jumps.legend(loc='upper right', fontsize=8)
        self.ax_jumps.grid(True, alpha=0.3)
    
    def _plot_candidates_list(self):
        """Display list of candidates with confidence scores"""
        self.ax_candidates.clear()
        self.ax_candidates.axis('off')
        
        text = "CANDIDATES & TRIALS\n" + "="*40 + "\n\n"
        
        # Status summary
        n_current = len(self.boundaries) + 1
        n_needed = self.n_trials
        status_symbol = "✓" if n_current == n_needed else "⚠"
        text += f"{status_symbol} Trials: {n_current}/{n_needed}\n"
        text += f"   Boundaries: {len(self.boundaries)}/{n_needed-1}\n\n"
        text += "="*40 + "\n\n"
        
        # Show top candidates
        text += "TOP CANDIDATES:\n" + "-"*40 + "\n"
        for i, (idx, row) in enumerate(self.candidates.head(20).iterrows()):
            is_selected = idx in self.boundaries
            marker = "✓" if is_selected else "○"
            
            time_val = row.get('Time_s', self.df.loc[idx, 'PacketTime']/1000)
            text += f"{marker} #{i+1}: t={time_val:.1f}s\n"
            text += f"    ΔT={row['DeltaT']:.0f}ms"
            text += f" Conf={row.get('Confidence', 0):.0f}%\n"
            text += f"    V={row['Velocity']:.0f}px/s"
            text += f" Jump={row['DistJump']:.0f}px\n"
            
            if i < 19:  # Add spacing except for last
                text += "\n"
        
        bgcolor = 'lightgreen' if n_current == n_needed else 'lightyellow'
        
        self.ax_candidates.text(0.05, 0.98, text,
                               transform=self.ax_candidates.transAxes,
                               fontsize=7, verticalalignment='top',
                               family='monospace',
                               bbox=dict(boxstyle='round', facecolor=bgcolor, 
                                       alpha=0.8, edgecolor='gray', linewidth=1))
    
    def _update_status(self):
        """Update status in title"""
        n_current = len(self.boundaries) + 1
        n_needed = self.n_trials
        
        # Update figure title with scroll position
        progress = (self.view_start - self.time_min) / self.time_range * 100
        
        status = f"Need {self.n_trials-1} boundaries for {self.n_trials} trials | "
        status += f"View: {progress:.0f}%-{min(progress+10, 100):.0f}% of timeline"
        
        self.fig.suptitle(
            f"Interactive Trial Boundary Validation\n{status}",
            fontsize=11, weight='bold', y=0.995
        )
    
    def _on_click(self, event):
        """Handle click events to toggle boundaries"""
        if event.inaxes != self.ax_temporal:
            return
        
        # Find nearest candidate to click
        click_time = event.xdata
        if click_time is None:
            return
        
        # Only consider candidates in current view
        candidates_in_view = self.candidates[
            (self.df.loc[self.candidates.index, "PacketTime"] >= self.view_start) &
            (self.df.loc[self.candidates.index, "PacketTime"] <= self.view_end)
        ]
        
        if len(candidates_in_view) == 0:
            print("No candidates in current view")
            return
        
        # Get candidate indices and times
        candidate_times = self.df.loc[candidates_in_view.index, "PacketTime"]
        distances = np.abs(candidate_times - click_time)
        nearest_idx = candidates_in_view.index[distances.argmin()]
        
        # Only toggle if click is reasonably close (within 5% of window)
        if distances.min() > self.window_size * 0.05:
            return
        
        # Toggle selection
        if nearest_idx in self.boundaries:
            self.boundaries.remove(nearest_idx)
            print(f"✗ Removed boundary at t={self.df.loc[nearest_idx, 'PacketTime']/1000:.1f}s "
                  f"({len(self.boundaries)+1} trials)")
        else:
            self.boundaries.append(nearest_idx)
            self.boundaries = sorted(self.boundaries)
            print(f"✓ Added boundary at t={self.df.loc[nearest_idx, 'PacketTime']/1000:.1f}s "
                  f"({len(self.boundaries)+1} trials)")
        
        self._plot_all()
    
    def _on_scroll(self, event):
        """Handle scroll events to pan the timeline"""
        if event.inaxes not in [self.ax_temporal, self.ax_velocity, self.ax_jumps]:
            return
        
        # Scroll direction: up = forward in time, down = backward
        direction = 1 if event.button == 'up' else -1
        
        # Scroll by 10% of window
        scroll_amount = self.window_size * 0.1 * direction
        
        self._pan_view(scroll_amount)
    
    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'left':
            # Pan backward in time
            self._pan_view(-self.window_size * 0.2)
            
        elif event.key == 'right':
            # Pan forward in time
            self._pan_view(self.window_size * 0.2)
            
        elif event.key == 'h':
            # Reset to full view
            self.view_start = self.time_min
            self.view_end = self.time_max
            self.window_size = self.time_range
            print("↻ Reset to full timeline view")
            self._plot_all()
            
        elif event.key == 'a':
            # Auto-select top N candidates
            n_needed = self.n_trials - 1
            self.boundaries = []
            
            for idx in self.candidates.index:
                if all(abs(idx - b) >= self.config.min_sep_samples for b in self.boundaries):
                    self.boundaries.append(idx)
                    if len(self.boundaries) == n_needed:
                        break
            
            self.boundaries = sorted(self.boundaries)
            print(f"✓ Auto-selected top {len(self.boundaries)} candidates → {len(self.boundaries)+1} trials")
            self._plot_all()
        
        elif event.key == 'r':
            # Reset to initial boundaries
            self.boundaries = sorted(self.initial_boundaries)
            print(f"↻ Reset to initial selection → {len(self.boundaries)+1} trials")
            self._plot_all()
            
        elif event.key == 'c':
            # Clear all
            self.boundaries = []
            print("✗ Cleared all boundaries → 1 trial")
            self._plot_all()
        
        elif event.key == 's':
            # Save CSV manually
            self._save_csv()
            print("💾 CSV saved manually")
    
    def _pan_view(self, amount):
        """Pan the view by the given amount"""
        new_start = self.view_start + amount
        new_end = self.view_end + amount
        
        # Keep within bounds
        if new_start < self.time_min:
            new_start = self.time_min
            new_end = new_start + self.window_size
        elif new_end > self.time_max:
            new_end = self.time_max
            new_start = new_end - self.window_size
        
        self.view_start = new_start
        self.view_end = new_end
        
        # Update only the time-series plots (faster)
        self._plot_temporal()
        self._plot_velocity()
        self._plot_jumps()
        self._update_status()
        self.fig.canvas.draw_idle()
    
    def _save_csv(self):
        """Save raw data with boundary markers to CSV"""
        import os
        from datetime import datetime
        
        # Create a copy of the dataframe with boundary information
        export_df = self.df.copy()
        
        # Add boundary marker column
        export_df["IsBoundary"] = False
        if len(self.boundaries) > 0:
            export_df.loc[self.boundaries, "IsBoundary"] = True
        
        # Add trial numbers based on current boundaries
        export_df["Trial"] = 0
        if len(self.boundaries) > 0:
            boundary_idx_sorted = sorted(self.boundaries)
            for trial_num, boundary in enumerate(boundary_idx_sorted, start=1):
                export_df.loc[boundary:, "Trial"] = trial_num
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trial_detection_data_{timestamp}.csv"
        
        # Save to current directory or user-specified path
        if self.csv_save_path:
            filepath = self.csv_save_path
        else:
            filepath = filename
        
        # Save to CSV
        export_df.to_csv(filepath, index=True)
        
        self.csv_save_path = filepath
        
        return filepath
    
    def _on_close(self, event):
        """Handle window close event - save CSV automatically"""
        try:
            filepath = self._save_csv()
            print("\n" + "="*70)
            print("INTERACTIVE SESSION CLOSED")
            print("="*70)
            print(f"✓ Raw data with boundaries saved to: {filepath}")
            print(f"  Total boundaries: {len(self.boundaries)}")
            print(f"  Total trials: {len(self.boundaries) + 1}")
            print(f"  Boundary indices: {self.boundaries}")
            print("="*70 + "\n")
        except Exception as e:
            print(f"\n⚠ Error saving CSV: {e}\n")


def detect_trials_auto(df: pd.DataFrame, config, interactive: bool = False,
                      csv_save_path: str = None) -> pd.DataFrame:
    """
    Convenience function for trial detection
    
    Args:
        df: Input DataFrame
        config: AnalysisConfig object
        interactive: If True, launch interactive validation
        csv_save_path: Optional path to save CSV with boundary data (for interactive mode)
        
    Returns:
        DataFrame with Trial column
    """
    detector = TrialDetector(config)
    df = detector.detect_trials(df, interactive=interactive, csv_save_path=csv_save_path)
    
    # Print candidate report
    detector.print_candidate_report()
    
    # Run diagnostics
    detector.diagnose_trial_numbering(df)
    
    return df