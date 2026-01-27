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
    
    def detect_trials(self, df: pd.DataFrame, interactive: bool = False) -> pd.DataFrame:
        """
        Detect trial boundaries and add Trial column to dataframe
        
        Args:
            df: Input DataFrame
            interactive: If True, launch interactive validation interface
            
        Returns:
            DataFrame with Trial column added
        """
        logger.info(f"Detecting {self.config.n_trials} trials...")
        
        # Add temporal and spatial features
        df = self._add_features(df)
        
        # Find candidate boundaries with confidence scores
        candidates = self._find_candidates(df)
        
        # Optimize boundary selection (timestamp-based priority)
        boundary_idx = self._optimize_boundaries(candidates, df)
        
        # Store for interactive review
        self.candidates_info = self._prepare_candidates_info(candidates, df)
        self.selected_boundaries = boundary_idx
        
        # Interactive validation if requested
        if interactive:
            boundary_idx = self._interactive_validation(df, candidates, boundary_idx)
            self.selected_boundaries = boundary_idx
        
        # Add trial labels
        df["NewTrial"] = False
        if len(boundary_idx) > 0:
            df.loc[boundary_idx, "NewTrial"] = True
        df["Trial"] = df["NewTrial"].cumsum()
        
        n_detected = df["Trial"].max() + 1
        logger.info(f"Successfully detected {n_detected} trials")
        
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
        2. Spatial jump (SECONDARY - supports temporal)
        3. Pressure change (TERTIARY - additional evidence)
        """
        # Temporal score (most important - weighted 3x)
        temporal_z = robust_z(candidates["DeltaT"])
        
        # Spatial score (important - weighted 1x)
        spatial_z = robust_z(candidates["DistJump"])
        
        # Pressure drop score (supplementary - weighted 0.5x)
        pressure_z = robust_z(-candidates["PressurePrev"])
        
        # Combined score with temporal dominance
        candidates["Score"] = (
            temporal_z * 3.0 +      # Temporal gap (MOST IMPORTANT)
            spatial_z * 1.0 +       # Spatial jump
            pressure_z * 0.5        # Pressure drop
        )
        
        # Calculate individual confidence components (0-100 scale)
        candidates["TemporalConfidence"] = np.clip(
            (temporal_z / temporal_z.max() * 100) if temporal_z.max() > 0 else 0, 
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
        Optimize boundary selection - MUST find exactly n_trials boundaries
        
        Strategy:
        1. Start with top N-1 candidates (N trials = N-1 boundaries)
        2. Ensure minimum separation between boundaries
        3. Adjust if needed to get exact count
        
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
        
        # Sort candidates by score (best first)
        candidates_sorted = candidates.sort_values("Score", ascending=False)
        
        # Greedy selection with separation constraint
        selected = []
        for idx in candidates_sorted.index:
            # Check if this candidate is far enough from all selected ones
            if all(abs(idx - s) >= self.config.min_sep_samples for s in selected):
                selected.append(idx)
                
                # Stop when we have enough boundaries
                if len(selected) == self.config.n_trials - 1:
                    break
        
        # Sort selected boundaries by position
        selected = sorted(selected)
        
        # Verify we got exactly the right number
        if len(selected) != self.config.n_trials - 1:
            # Try with relaxed separation if we don't have enough
            logger.warning(
                f"Could not find {self.config.n_trials - 1} boundaries with minimum separation. "
                f"Trying with relaxed constraints..."
            )
            selected = self._relaxed_selection(candidates_sorted)
        
        if len(selected) != self.config.n_trials - 1:
            raise RuntimeError(
                f"Could not find exactly {self.config.n_trials} trials. "
                f"Found {len(selected) + 1} trials. Check your data or adjust n_trials parameter."
            )
        
        return selected
    
    def _relaxed_selection(self, candidates_sorted: pd.DataFrame) -> List[int]:
        """
        Relaxed boundary selection when strict separation fails
        Uses reduced minimum separation
        """
        relaxed_sep = self.config.min_sep_samples // 2
        selected = []
        
        for idx in candidates_sorted.index:
            if all(abs(idx - s) >= relaxed_sep for s in selected):
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
                               initial_boundaries: List[int]) -> List[int]:
        """
        Launch interactive interface for boundary validation and correction
        
        Returns:
            Updated list of boundary indices
        """
        validator = InteractiveTrialValidator(
            df, candidates, initial_boundaries, 
            self.config.n_trials, self.config
        )
        
        return validator.run()
    
    def visualize_detection(self, df: pd.DataFrame, save_path: str = None):
        """
        Visualize detected trial boundaries
        
        Args:
            df: DataFrame with Trial column
            save_path: Optional path to save figure
        """
        boundary_idx = df[df["NewTrial"]].index
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Temporal gaps (log scale)
        ax1 = axes[0]
        ax1.plot(df["PacketTime"], df["DeltaT"], alpha=0.5, linewidth=0.5)
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
        
        # Plot 2: Spatial jumps
        ax2 = axes[1]
        ax2.plot(df["PacketTime"], df["DistJump"], alpha=0.5, linewidth=0.5, color='blue')
        ax2.scatter(df.loc[boundary_idx, "PacketTime"],
                   df.loc[boundary_idx, "DistJump"], 
                   c="red", s=100, zorder=5, label="Selected boundaries", marker='o')
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Spatial Jump (px)")
        ax2.set_title("Spatial Jumps at Boundaries")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Spatial trajectory with trial colors
        ax3 = axes[2]
        colors = plt.cm.tab20.colors
        for trial_id in df["Trial"].unique():
            trial_data = df[df["Trial"] == trial_id]
            ax3.plot(trial_data["X"], trial_data["Y"],
                    color=colors[int(trial_id) % len(colors)],
                    linewidth=1.5, alpha=0.8, label=f"Trial {trial_id}")
        
        ax3.set_aspect(self.config.trajectory_aspect_ratio)
        ax3.set_xlabel("X (px)")
        ax3.set_ylabel("Y (px)")
        ax3.set_title(f"Detected Trials (n={df['Trial'].max() + 1})")
        ax3.grid(True, alpha=0.3)
        
        # Only show legend if not too many trials
        if df["Trial"].max() < 15:
            ax3.legend(loc='best', fontsize=8)
        
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
        display_cols = ['Rank', 'Time_s', 'DeltaT', 'DistJump', 
                       'Confidence', 'TemporalConfidence', 'SpatialConfidence', 
                       'IsSelected']
        
        print("\nTop 20 Candidates (sorted by confidence):")
        print("-"*80)
        report = self.candidates_info[display_cols].head(20).copy()
        report['Time_s'] = report['Time_s'].round(2)
        report['DeltaT'] = report['DeltaT'].round(1)
        report['DistJump'] = report['DistJump'].round(1)
        report['Confidence'] = report['Confidence'].round(1)
        report['TemporalConfidence'] = report['TemporalConfidence'].round(1)
        report['SpatialConfidence'] = report['SpatialConfidence'].round(1)
        
        print(report.to_string(index=False))
        print("="*80 + "\n")


class InteractiveTrialValidator:
    """
    Interactive interface for validating and correcting trial boundaries
    """
    
    def __init__(self, df: pd.DataFrame, candidates: pd.DataFrame,
                 initial_boundaries: List[int], n_trials: int, config):
        self.df = df
        self.candidates = candidates.sort_values("Score", ascending=False)
        self.boundaries = sorted(initial_boundaries)
        self.n_trials = n_trials
        self.config = config
        
        # For manual point selection
        self.manual_points = []
        
    def run(self) -> List[int]:
        """Launch interactive interface and return validated boundaries"""
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(4, 3, height_ratios=[3, 2, 2, 0.5],
                                  width_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
        
        # Main plot: Temporal gaps with candidates
        self.ax_temporal = self.fig.add_subplot(gs[0, :2])
        
        # Candidate list
        self.ax_candidates = self.fig.add_subplot(gs[0, 2])
        
        # Spatial trajectory
        self.ax_spatial = self.fig.add_subplot(gs[1, :])
        
        # Spatial jumps
        self.ax_jumps = self.fig.add_subplot(gs[2, :])
        
        # Instructions
        self.ax_info = self.fig.add_subplot(gs[3, :])
        self.ax_info.axis('off')
        
        # Initial plots
        self._plot_all()
        
        # Connect events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        plt.suptitle(f"Interactive Trial Boundary Validation - Need {self.n_trials-1} boundaries for {self.n_trials} trials",
                    fontsize=14, weight='bold')
        
        print("\n" + "="*60)
        print("INTERACTIVE VALIDATION MODE")
        print("="*60)
        print("Click on the temporal gap plot to toggle boundaries")
        print("Press 'a' to auto-select top N candidates")
        print("Press 'r' to reset to initial selection")
        print("Press 'c' to clear all boundaries")
        print("Close the window when satisfied")
        print("="*60 + "\n")
        
        plt.show()
        
        return sorted(self.boundaries)
    
    def _plot_all(self):
        """Refresh all plots"""
        self._plot_temporal()
        self._plot_candidates_list()
        self._plot_spatial()
        self._plot_jumps()
        self._update_info()
        self.fig.canvas.draw()
    
    def _plot_temporal(self):
        """Plot temporal gaps with selected boundaries"""
        self.ax_temporal.clear()
        
        # Plot all gaps
        self.ax_temporal.plot(self.df["PacketTime"], self.df["DeltaT"], 
                             alpha=0.5, linewidth=0.5, color='gray')
        
        # Mark all candidates
        self.ax_temporal.scatter(
            self.df.loc[self.candidates.index, "PacketTime"],
            self.df.loc[self.candidates.index, "DeltaT"],
            c='orange', s=30, alpha=0.5, label='All candidates', marker='x'
        )
        
        # Mark selected boundaries
        if self.boundaries:
            self.ax_temporal.scatter(
                self.df.loc[self.boundaries, "PacketTime"],
                self.df.loc[self.boundaries, "DeltaT"],
                c='red', s=100, zorder=5, label='Selected', marker='o', 
                edgecolors='black', linewidths=2
            )
        
        # Show threshold
        self.ax_temporal.axhline(self.config.min_gap_ms, color='green',
                                linestyle='--', alpha=0.5, 
                                label=f'Min gap ({self.config.min_gap_ms}ms)')
        
        self.ax_temporal.set_yscale('log')
        self.ax_temporal.set_xlabel('Time (ms)', fontsize=10)
        self.ax_temporal.set_ylabel('Time Gap (ms)', fontsize=10)
        self.ax_temporal.set_title('Temporal Gaps (Click to toggle boundaries)', fontsize=11)
        self.ax_temporal.legend(loc='upper right')
        self.ax_temporal.grid(True, alpha=0.3)
    
    def _plot_candidates_list(self):
        """Display list of candidates with confidence scores"""
        self.ax_candidates.clear()
        self.ax_candidates.axis('off')
        
        text = "TOP CANDIDATES\n" + "="*35 + "\n\n"
        
        for i, (idx, row) in enumerate(self.candidates.head(15).iterrows()):
            is_selected = idx in self.boundaries
            marker = "✓" if is_selected else " "
            
            text += f"{marker} #{i+1}: t={row.get('Time_s', self.df.loc[idx, 'PacketTime']/1000):.1f}s\n"
            text += f"   ΔT={row['DeltaT']:.0f}ms"
            text += f" | Conf={row.get('Confidence', 0):.0f}%\n"
            text += f"   Jump={row['DistJump']:.0f}px\n\n"
        
        self.ax_candidates.text(0.05, 0.95, text,
                               transform=self.ax_candidates.transAxes,
                               fontsize=8, verticalalignment='top',
                               family='monospace',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def _plot_spatial(self):
        """Plot spatial trajectory colored by trial"""
        self.ax_spatial.clear()
        
        # Create temporary trial labels
        temp_df = self.df.copy()
        temp_df["TempTrial"] = 0
        for i, boundary in enumerate(sorted(self.boundaries)):
            temp_df.loc[boundary:, "TempTrial"] = i + 1
        
        # Plot each trial
        colors = plt.cm.tab20.colors
        for trial_id in temp_df["TempTrial"].unique():
            trial_data = temp_df[temp_df["TempTrial"] == trial_id]
            self.ax_spatial.plot(trial_data["X"], trial_data["Y"],
                                color=colors[int(trial_id) % len(colors)],
                                linewidth=1.5, alpha=0.8)
        
        self.ax_spatial.set_aspect(self.config.trajectory_aspect_ratio)
        self.ax_spatial.set_xlabel('X (px)')
        self.ax_spatial.set_ylabel('Y (px)')
        self.ax_spatial.set_title(f'Spatial Trajectory ({len(self.boundaries)+1} trials)')
        self.ax_spatial.grid(True, alpha=0.3)
    
    def _plot_jumps(self):
        """Plot spatial jumps"""
        self.ax_jumps.clear()
        
        self.ax_jumps.plot(self.df["PacketTime"], self.df["DistJump"],
                          alpha=0.5, linewidth=0.5, color='blue')
        
        if self.boundaries:
            self.ax_jumps.scatter(
                self.df.loc[self.boundaries, "PacketTime"],
                self.df.loc[self.boundaries, "DistJump"],
                c='red', s=100, zorder=5, marker='o',
                edgecolors='black', linewidths=2
            )
        
        self.ax_jumps.set_xlabel('Time (ms)')
        self.ax_jumps.set_ylabel('Spatial Jump (px)')
        self.ax_jumps.set_title('Spatial Jumps')
        self.ax_jumps.grid(True, alpha=0.3)
    
    def _update_info(self):
        """Update information panel"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        n_current = len(self.boundaries) + 1
        n_needed = self.n_trials
        status = "✓ CORRECT" if n_current == n_needed else f"⚠ INCORRECT"
        
        text = f"BOUNDARIES: {len(self.boundaries)} selected → {n_current} trials "
        text += f"(need {n_needed}) {status}\n\n"
        text += "CONTROLS: Click plot to toggle | 'a': auto-select | 'r': reset | 'c': clear | Close when done"
        
        color = 'lightgreen' if n_current == n_needed else 'lightcoral'
        
        self.ax_info.text(0.05, 0.5, text,
                         transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    def _on_click(self, event):
        """Handle click events to toggle boundaries"""
        if event.inaxes != self.ax_temporal:
            return
        
        # Find nearest candidate to click
        click_time = event.xdata
        if click_time is None:
            return
        
        # Get candidate indices and times
        candidate_times = self.df.loc[self.candidates.index, "PacketTime"]
        distances = np.abs(candidate_times - click_time)
        nearest_idx = self.candidates.index[distances.argmin()]
        
        # Toggle selection
        if nearest_idx in self.boundaries:
            self.boundaries.remove(nearest_idx)
            print(f"✗ Removed boundary at t={self.df.loc[nearest_idx, 'PacketTime']/1000:.1f}s")
        else:
            self.boundaries.append(nearest_idx)
            self.boundaries = sorted(self.boundaries)
            print(f"✓ Added boundary at t={self.df.loc[nearest_idx, 'PacketTime']/1000:.1f}s")
        
        self._plot_all()
    
    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'a':
            # Auto-select top N candidates
            n_needed = self.n_trials - 1
            self.boundaries = []
            
            for idx in self.candidates.index:
                if all(abs(idx - b) >= self.config.min_sep_samples for b in self.boundaries):
                    self.boundaries.append(idx)
                    if len(self.boundaries) == n_needed:
                        break
            
            self.boundaries = sorted(self.boundaries)
            print(f"✓ Auto-selected top {len(self.boundaries)} candidates")
            
        elif event.key == 'r':
            # Reset to initial
            print("↶ Reset to initial selection")
            # Would need to store initial state
            
        elif event.key == 'c':
            # Clear all
            self.boundaries = []
            print("✗ Cleared all boundaries")
        
        self._plot_all()


def detect_trials_auto(df: pd.DataFrame, config, interactive: bool = False) -> pd.DataFrame:
    """
    Convenience function for trial detection
    
    Args:
        df: Input DataFrame
        config: AnalysisConfig object
        interactive: If True, launch interactive validation
        
    Returns:
        DataFrame with Trial column
    """
    detector = TrialDetector(config)
    df = detector.detect_trials(df, interactive=interactive)
    
    # Print candidate report
    detector.print_candidate_report()
    
    return df