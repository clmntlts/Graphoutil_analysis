"""
Trial detection using multi-objective optimization with interactive validation.
Detects trial boundaries in continuous handwriting data from PsychoPy experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def robust_z(x: pd.Series) -> pd.Series:
    """
    Calculate robust z-score using median absolute deviation.

    Args:
        x: Input series

    Returns:
        Robust z-scores
    """
    med = x.median()
    mad = np.median(np.abs(x - med))
    return (x - med) / (mad + 1e-6)


# ---------------------------------------------------------------------------
# Colour palette – one colour per trial (cycles for > 20 trials)
# ---------------------------------------------------------------------------
TAB20_COLORS = plt.cm.tab20.colors


def trial_color(trial_id: int) -> tuple:
    return TAB20_COLORS[int(trial_id) % len(TAB20_COLORS)]


# ---------------------------------------------------------------------------
# TrialDetector
# ---------------------------------------------------------------------------

class TrialDetector:
    """
    Detects trial boundaries using timestamp jumps as primary criterion.
    PsychoPy start/stop commands create clear temporal gaps between trials.
    """

    def __init__(self, config):
        """
        Initialize TrialDetector.

        Args:
            config: AnalysisConfig object (must have n_trials set).
        """
        self.config = config

        if config.n_trials is None:
            raise ValueError("n_trials must be specified in config")

        # Store detection results for interactive review
        self.candidates_info = None
        self.selected_boundaries = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_trials(self, df: pd.DataFrame, interactive: bool = False,
                      csv_save_path: str = None) -> pd.DataFrame:
        """
        Detect trial boundaries and add Trial column to dataframe.

        Args:
            df:            Input DataFrame (PacketTime, X, Y, NormalPressure).
            interactive:   If True, launch interactive validation interface.
            csv_save_path: Optional path to save CSV file with boundary data.

        Returns:
            DataFrame with Trial column added (0 = data before first boundary).
        """
        logger.info(f"Detecting {self.config.n_trials} trials...")

        # Add temporal and spatial features
        df = self._add_features(df)

        # Find candidate boundaries with confidence scores
        candidates = self._find_candidates(df)

        # Optimize boundary selection
        boundary_idx = self._optimize_boundaries(candidates, df)

        # Store for interactive review
        self.selected_boundaries = boundary_idx
        self.candidates_info = self._prepare_candidates_info(candidates, df)

        # Interactive validation if requested
        if interactive:
            boundary_idx = self._interactive_validation(
                df, candidates, boundary_idx, csv_save_path
            )
            self.selected_boundaries = boundary_idx

        # Add trial labels - ensure sequential numbering without gaps
        df["Trial"] = 0
        if len(boundary_idx) > 0:
            boundary_idx_sorted = sorted(boundary_idx)
            for trial_num, boundary in enumerate(boundary_idx_sorted, start=1):
                df.loc[boundary:, "Trial"] = trial_num

        # Verify sequential numbering and fix if needed
        unique_trials = sorted(df["Trial"].unique())
        expected_trials = list(range(len(boundary_idx) + 1))

        if unique_trials != expected_trials:
            logger.warning(
                f"Trial numbering issue detected. "
                f"Expected {expected_trials}, got {unique_trials}"
            )
            trial_mapping = {old: new for new, old in enumerate(unique_trials)}
            df["Trial"] = df["Trial"].map(trial_mapping)

        n_detected = len(df["Trial"].unique())
        logger.info(
            f"Successfully detected {n_detected} trials "
            f"(Trial IDs: {sorted(df['Trial'].unique())})"
        )

        if n_detected != self.config.n_trials:
            logger.warning(
                f"Expected {self.config.n_trials} trials but found {n_detected}"
            )

        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and spatial features for boundary detection."""
        # Temporal gap (PRIMARY - PsychoPy start/stop)
        df["DeltaT"] = df["PacketTime"].diff().fillna(self.config.expected_interval)

        # Spatial jump
        dx = df["X"].diff().fillna(0)
        dy = df["Y"].diff().fillna(0)
        df["DistJump"] = np.sqrt(dx ** 2 + dy ** 2)

        # Velocity (pixels per second)
        time_delta_sec = df["DeltaT"] / 1000.0
        time_delta_sec = time_delta_sec.replace(0, np.nan)
        df["Velocity"] = df["DistJump"] / time_delta_sec
        df["Velocity"] = df["Velocity"].fillna(0)

        # Previous pressure
        df["PressurePrev"] = df["NormalPressure"].shift(1).fillna(0)

        # Current pressure (for detecting pen lifts)
        df["PressureCurr"] = df["NormalPressure"]

        return df

    # ------------------------------------------------------------------
    # Candidate finding & scoring
    # ------------------------------------------------------------------

    def _find_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find candidate trial boundaries with confidence scores.

        Args:
            df: DataFrame with features.

        Returns:
            DataFrame of candidates with scores and confidence metrics.
        """
        typical_gap   = self.config.expected_interval
        min_trial_gap = max(self.config.min_gap_ms, typical_gap * 5)
        candidates    = df[df["DeltaT"] > min_trial_gap].copy()

        if len(candidates) == 0:
            logger.warning(f"No candidates found with DeltaT > {min_trial_gap}ms")
            min_trial_gap = self.config.min_gap_ms
            candidates    = df[df["DeltaT"] > min_trial_gap].copy()

        candidates = self._calculate_confidence_scores(candidates, df)
        candidates = candidates.sort_values("Score", ascending=False)

        logger.info(
            f"Found {len(candidates)} candidate boundaries "
            f"(gap > {min_trial_gap:.1f}ms)"
        )
        return candidates

    def _calculate_confidence_scores(self, candidates: pd.DataFrame,
                                     df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate multi-component confidence score for each candidate.

        Components:
            1. Temporal gap   (weight 3.0 – most important for PsychoPy)
            2. Velocity spike (weight 2.0)
            3. Spatial jump   (weight 1.0)
            4. Pressure drop  (weight 0.5)
        """
        temporal_z = robust_z(candidates["DeltaT"])
        velocity_z = robust_z(candidates["Velocity"])
        spatial_z  = robust_z(candidates["DistJump"])
        pressure_z = robust_z(-candidates["PressurePrev"])

        candidates["Score"] = (
            temporal_z * 3.0
            + velocity_z * 2.0
            + spatial_z  * 1.0
            + pressure_z * 0.5
        )

        def _norm(z):
            m = z.max()
            return np.clip((z / m * 100) if m > 0 else 0, 0, 100)

        candidates["TemporalConfidence"] = _norm(temporal_z)
        candidates["VelocityConfidence"] = _norm(velocity_z)
        candidates["SpatialConfidence"]  = _norm(spatial_z)
        candidates["PressureConfidence"] = _norm(pressure_z)

        max_score = candidates["Score"].max()
        candidates["Confidence"] = np.clip(
            (candidates["Score"] / max_score * 100) if max_score > 0 else 0,
            0, 100
        )

        return candidates

    # ------------------------------------------------------------------
    # Boundary optimisation
    # ------------------------------------------------------------------

    def _optimize_boundaries(self, candidates: pd.DataFrame,
                             df: pd.DataFrame) -> List[int]:
        """
        Greedy boundary selection with adaptive trial-size validation.

        Strategy:
            1. Compute expected trial sizes from all candidates.
            2. Use median trial size as adaptive baseline.
            3. Select candidates greedily (highest score first) subject to:
               - minimum separation between boundaries
               - no resulting trial smaller than 30% of median
            4. Relax constraints if not enough boundaries found.
        """
        if len(candidates) < self.config.n_trials - 1:
            raise RuntimeError(
                f"Not enough candidates ({len(candidates)}) to create "
                f"{self.config.n_trials} trials. "
                f"Need at least {self.config.n_trials - 1} boundaries."
            )

        all_trial_sizes     = self._calculate_trial_sizes_from_candidates(candidates, df)
        median_trial_size   = np.median(all_trial_sizes)
        percentile_25       = np.percentile(all_trial_sizes, 25)
        percentile_10       = np.percentile(all_trial_sizes, 10)
        min_acceptable_size = median_trial_size * 0.30

        logger.info("ADAPTIVE TRIAL SIZE THRESHOLDS:")
        logger.info(f"  Median trial size: {median_trial_size:.0f} points")
        logger.info(f"  25th percentile:   {percentile_25:.0f} points")
        logger.info(f"  10th percentile:   {percentile_10:.0f} points")
        logger.info(f"  Minimum (30%):     {min_acceptable_size:.0f} points")

        candidates_sorted = candidates.sort_values("Score", ascending=False)
        selected      = []
        skipped_count = 0

        for idx in candidates_sorted.index:
            if not all(
                abs(idx - s) >= self.config.min_sep_samples for s in selected
            ):
                continue

            test_boundaries = sorted(selected + [idx])
            if self._creates_small_trial(test_boundaries, df, min_acceptable_size):
                skipped_count += 1
                logger.debug(
                    f"Skipping index {idx} – would create trial "
                    f"< {min_acceptable_size:.0f} points"
                )
                continue

            selected.append(idx)
            if len(selected) == self.config.n_trials - 1:
                break

        logger.info(f"Skipped {skipped_count} candidates (small trial constraint)")

        selected = sorted(selected)

        if len(selected) < self.config.n_trials - 1:
            logger.warning(
                f"Only {len(selected)} boundaries with strict constraints. "
                f"Relaxing to 20% of median..."
            )
            relaxed_min_size = median_trial_size * 0.20
            selected = self._relaxed_selection_with_size_check(
                candidates_sorted, df, relaxed_min_size
            )

        if len(selected) != self.config.n_trials - 1:
            logger.error(
                f"Could not find exactly {self.config.n_trials - 1} boundaries. "
                f"Found {len(selected)}."
            )
            self._log_trial_sizes(selected, df)
            raise RuntimeError(
                f"Could not find exactly {self.config.n_trials} trials. "
                f"Found {len(selected) + 1} trials. "
                f"Check your data or adjust n_trials parameter."
            )

        logger.info(f"Selected {len(selected)} boundaries")
        self._log_trial_sizes(selected, df)
        return selected

    def _calculate_trial_sizes_from_candidates(self, candidates: pd.DataFrame,
                                               df: pd.DataFrame) -> List[float]:
        all_sizes = []

        all_candidate_indices = sorted(candidates.index.tolist())
        trial_starts = [df.index[0]] + all_candidate_indices
        trial_ends   = all_candidate_indices + [df.index[-1]]
        for start, end in zip(trial_starts, trial_ends):
            sp = df.index.get_loc(start) if start in df.index else 0
            ep = df.index.get_loc(end)   if end   in df.index else len(df) - 1
            all_sizes.append(ep - sp + 1)

        n_needed = self.config.n_trials - 1
        if len(candidates) >= n_needed:
            top_sorted = sorted(
                candidates.nlargest(n_needed, "Score").index.tolist()
            )
            trial_starts = [df.index[0]] + top_sorted
            trial_ends   = top_sorted + [df.index[-1]]
            for start, end in zip(trial_starts, trial_ends):
                sp = df.index.get_loc(start) if start in df.index else 0
                ep = df.index.get_loc(end)   if end   in df.index else len(df) - 1
                all_sizes.append(ep - sp + 1)

        all_sizes.append(len(df) / self.config.n_trials)
        return all_sizes

    def _creates_small_trial(self, boundaries: List[int], df: pd.DataFrame,
                             min_size: float) -> bool:
        if len(boundaries) == 0:
            return False
        trial_starts = [df.index[0]] + boundaries
        trial_ends   = boundaries + [df.index[-1]]
        for start, end in zip(trial_starts, trial_ends):
            sp = df.index.get_loc(start) if start in df.index else 0
            ep = df.index.get_loc(end)   if end   in df.index else len(df) - 1
            if ep - sp + 1 < min_size:
                return True
        return False

    def _log_trial_sizes(self, boundaries: List[int], df: pd.DataFrame):
        if len(boundaries) == 0:
            logger.info(f"No boundaries – single trial of {len(df)} points")
            return
        trial_starts = [df.index[0]] + boundaries
        trial_ends   = boundaries + [df.index[-1]]
        logger.info("Trial sizes with selected boundaries:")
        for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
            sp = df.index.get_loc(start) if start in df.index else 0
            ep = df.index.get_loc(end)   if end   in df.index else len(df) - 1
            logger.info(f"  Trial {i}: {ep - sp + 1} points")

    def _relaxed_selection_with_size_check(self, candidates_sorted: pd.DataFrame,
                                           df: pd.DataFrame,
                                           min_size: float) -> List[int]:
        relaxed_sep = self.config.min_sep_samples // 2
        selected = []
        for idx in candidates_sorted.index:
            if not all(abs(idx - s) >= relaxed_sep for s in selected):
                continue
            test_boundaries = sorted(selected + [idx])
            if self._creates_small_trial(test_boundaries, df, min_size):
                continue
            selected.append(idx)
            if len(selected) == self.config.n_trials - 1:
                break
        return sorted(selected)

    # ------------------------------------------------------------------
    # Candidate info preparation
    # ------------------------------------------------------------------

    def _prepare_candidates_info(self, candidates: pd.DataFrame,
                                 df: pd.DataFrame) -> pd.DataFrame:
        info = candidates.copy()
        info["Time_s"]     = df.loc[info.index, "PacketTime"] / 1000
        info["X"]          = df.loc[info.index, "X"]
        info["Y"]          = df.loc[info.index, "Y"]
        info["IsSelected"] = info.index.isin(self.selected_boundaries)
        info["Rank"]       = range(1, len(info) + 1)
        return info

    # ------------------------------------------------------------------
    # Interactive validation
    # ------------------------------------------------------------------

    def _interactive_validation(self, df: pd.DataFrame,
                                candidates: pd.DataFrame,
                                initial_boundaries: List[int],
                                csv_save_path: str = None) -> List[int]:
        validator = InteractiveTrialValidator(
            df, candidates, initial_boundaries,
            self.config.n_trials, self.config, csv_save_path
        )
        return validator.run()

    # ------------------------------------------------------------------
    # Static visualisation (post-detection summary)
    # ------------------------------------------------------------------

    def visualize_detection(self, df: pd.DataFrame, save_path: str = None):
        """
        Non-interactive summary figure of detected trial boundaries.

        Args:
            df:        DataFrame with Trial column already set.
            save_path: Optional path to save figure.
        """
        boundary_idx = self.selected_boundaries if self.selected_boundaries else []

        unique_trials   = sorted(df["Trial"].unique())
        expected_trials = list(range(len(unique_trials)))
        if unique_trials != expected_trials:
            logger.warning(
                f"Trial numbering gaps! "
                f"Found: {unique_trials}, Expected: {expected_trials}"
            )

        fig, axes = plt.subplots(4, 1, figsize=(14, 14))

        # ── Plot 1: Temporal gaps ──────────────────────────────────────
        ax1 = axes[0]
        ax1.plot(df["PacketTime"], df["DeltaT"], alpha=0.5, linewidth=0.5)

        if len(boundary_idx) > 0:
            for i, b in enumerate(sorted(boundary_idx), start=1):
                color = trial_color(i)
                ax1.scatter(
                    df.loc[b, "PacketTime"], df.loc[b, "DeltaT"],
                    c=[color], s=120, zorder=5, marker="o",
                    edgecolors="black", linewidths=1.5
                )
                ax1.annotate(
                    f"T{i}",
                    xy=(df.loc[b, "PacketTime"], df.loc[b, "DeltaT"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=color, fontweight="bold"
                )

        if self.candidates_info is not None:
            other = self.candidates_info[~self.candidates_info["IsSelected"]]
            ax1.scatter(
                other["Time_s"] * 1000,
                df.loc[other.index, "DeltaT"],
                c="orange", s=50, alpha=0.5, zorder=4,
                label="Other candidates", marker="x"
            )

        ax1.axhline(
            self.config.min_gap_ms, color="green",
            linestyle="--", alpha=0.5,
            label=f"Min gap ({self.config.min_gap_ms}ms)"
        )
        ax1.set_yscale("log")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("ΔT (ms)")
        ax1.set_title("Trial Boundary Detection – Temporal Gaps")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ── Plot 2: Velocity ───────────────────────────────────────────
        ax2 = axes[1]
        ax2.plot(df["PacketTime"], df["Velocity"],
                 alpha=0.5, linewidth=0.5, color="purple")
        if len(boundary_idx) > 0:
            for i, b in enumerate(sorted(boundary_idx), start=1):
                color = trial_color(i)
                ax2.scatter(
                    df.loc[b, "PacketTime"], df.loc[b, "Velocity"],
                    c=[color], s=120, zorder=5, marker="o",
                    edgecolors="black", linewidths=1.5
                )
                ax2.annotate(
                    f"T{i}",
                    xy=(df.loc[b, "PacketTime"], df.loc[b, "Velocity"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=color, fontweight="bold"
                )
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Velocity (px/s)")
        ax2.set_title("Velocity Spikes at Boundaries")
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Spatial jumps ──────────────────────────────────────
        ax3 = axes[2]
        ax3.plot(df["PacketTime"], df["DistJump"],
                 alpha=0.5, linewidth=0.5, color="teal")
        if len(boundary_idx) > 0:
            for i, b in enumerate(sorted(boundary_idx), start=1):
                color = trial_color(i)
                ax3.scatter(
                    df.loc[b, "PacketTime"], df.loc[b, "DistJump"],
                    c=[color], s=120, zorder=5, marker="o",
                    edgecolors="black", linewidths=1.5
                )
                ax3.annotate(
                    f"T{i}",
                    xy=(df.loc[b, "PacketTime"], df.loc[b, "DistJump"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=color, fontweight="bold"
                )
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Spatial Jump (px)")
        ax3.set_title("Spatial Jumps at Boundaries")
        ax3.grid(True, alpha=0.3)

        # ── Plot 4: Spatial trajectory ─────────────────────────────────
        ax4 = axes[3]
        _plot_trajectory_by_trial(ax4, df, "Trial")
        ax4.set_aspect(self.config.trajectory_aspect_ratio)
        ax4.set_xlabel("X (px)")
        ax4.set_ylabel("Y (px)")
        n_trials_actual = len(df["Trial"].unique())
        ax4.set_title(f"Detected Trials (n={n_trials_actual})")
        ax4.grid(True, alpha=0.3)
        if n_trials_actual < 15:
            ax4.legend(loc="best", fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved detection visualization to {save_path}")

        plt.show()

    # ------------------------------------------------------------------
    # Reports & diagnostics
    # ------------------------------------------------------------------

    def print_candidate_report(self):
        """Print detailed report of all candidates with confidence scores."""
        if self.candidates_info is None:
            print("No candidate information available. Run detect_trials() first.")
            return

        print("\n" + "=" * 80)
        print("TRIAL BOUNDARY CANDIDATES REPORT")
        print("=" * 80)
        print(f"Total candidates found:  {len(self.candidates_info)}")
        print(f"Required boundaries:     {self.config.n_trials - 1}")
        print(f"Selected boundaries:     {self.candidates_info['IsSelected'].sum()}")
        print("=" * 80)

        display_cols = [
            "Rank", "Time_s", "DeltaT", "Velocity", "DistJump",
            "Confidence", "TemporalConfidence", "VelocityConfidence",
            "SpatialConfidence", "IsSelected"
        ]

        print("\nTop 20 Candidates (sorted by confidence):")
        print("-" * 80)
        report = self.candidates_info[display_cols].head(20).copy()
        for col in ["Time_s", "DeltaT", "Velocity", "DistJump",
                    "Confidence", "TemporalConfidence",
                    "VelocityConfidence", "SpatialConfidence"]:
            report[col] = report[col].round(1)
        print(report.to_string(index=False))
        print("=" * 80 + "\n")

    def diagnose_trial_numbering(self, df: pd.DataFrame):
        """Diagnose trial numbering issues."""
        print("\n" + "=" * 80)
        print("TRIAL NUMBERING DIAGNOSTICS")
        print("=" * 80)

        unique_trials   = sorted(df["Trial"].unique())
        n_unique        = len(unique_trials)
        expected_trials = list(range(n_unique))

        print(f"Expected number of trials:  {self.config.n_trials}")
        print(f"Detected unique trials:      {n_unique}")
        print(f"Trial IDs found:             {unique_trials}")
        print(f"Expected sequential IDs:     {expected_trials}")

        gaps = set(expected_trials) - set(unique_trials)
        if gaps:
            print(f"\n⚠ GAPS DETECTED: Missing trial IDs: {sorted(gaps)}")
        else:
            print("\n✓ No gaps in trial numbering")

        print("\nTrial sizes (number of data points per trial):")
        for trial_id, count in df.groupby("Trial").size().sort_index().items():
            print(f"  Trial {trial_id}: {count} points")

        if self.selected_boundaries:
            print(f"\nSelected boundary indices: {self.selected_boundaries}")
            print(f"Number of boundaries:      {len(self.selected_boundaries)}")

        print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Shared helper – trajectory coloured by trial with trial-number labels
# ---------------------------------------------------------------------------

def _plot_trajectory_by_trial(ax, df: pd.DataFrame, trial_col: str = "Trial"):
    """
    Draw pen trajectory on *ax* coloured by trial.
    Only strokes where NormalPressure > 0 are drawn.
    Each trial receives a bold trial-number label at the stroke centroid.
    """
    for trial_id in sorted(df[trial_col].unique()):
        trial_data = df[df[trial_col] == trial_id]
        color      = trial_color(trial_id)

        pressure_mask    = trial_data["NormalPressure"] > 0
        pressure_changes = pressure_mask.astype(int).diff().fillna(0)
        seg_starts       = trial_data.index[pressure_changes == 1].tolist()
        seg_ends         = trial_data.index[pressure_changes == -1].tolist()

        if pressure_mask.iloc[0]:
            seg_starts.insert(0, trial_data.index[0])
        if pressure_mask.iloc[-1]:
            seg_ends.append(trial_data.index[-1])

        first_seg   = True
        all_x, all_y = [], []

        for s, e in zip(seg_starts, seg_ends):
            seg = trial_data.loc[s:e]
            ax.plot(
                seg["X"], seg["Y"],
                color=color, linewidth=1.5, alpha=0.85,
                label=f"Trial {trial_id}" if first_seg else ""
            )
            all_x.extend(seg["X"].tolist())
            all_y.extend(seg["Y"].tolist())
            first_seg = False

        # Label at the median position of all drawn strokes for this trial
        if all_x:
            cx = np.median(all_x)
            cy = np.median(all_y)
            ax.text(
                cx, cy, str(trial_id),
                fontsize=9, fontweight="bold", color=color,
                ha="center", va="center",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white", edgecolor=color,
                    alpha=0.80, linewidth=1.3
                ),
                zorder=10
            )


# ---------------------------------------------------------------------------
# InteractiveTrialValidator  – redesigned GUI
# ---------------------------------------------------------------------------

class InteractiveTrialValidator:
    """
    Interactive GUI for validating and correcting trial boundaries.

    Layout
    ------
    Top half  : Spatial trajectory – each trial coloured and numbered
    Bottom half: Three time-series panels (ΔT, velocity, spatial jump)
                 sharing the same scrollable x-axis
    Footer    : Two rows of click-friendly buttons

    Every selected boundary is labelled T1, T2, … in a matching colour
    both on the trajectory and on all three time-series panels, making it
    immediately obvious which gap on the timeline corresponds to which
    word on the tablet surface.

    Controls
    --------
    Mouse : scroll inside any time-series panel to pan; click inside the
            ΔT panel to add/remove the nearest candidate boundary.
    Keys  : ← → pan  |  h full view  |  a auto  |  r reset  |
            c clear  |  s save
    Buttons: ◀◀ Start  ◀ Back  Fwd ▶  End ▶▶  (navigation)
             ↺ Reset   ⚡ Auto  ✕ Clear  💾 Save  (actions)
    """

    _BTN_H          = 0.038   # button height  (figure fraction)
    _BTN_GAP        = 0.008   # gap between the two button rows
    _TOOLBAR_BOTTOM = 0.005   # bottom of lower row

    def __init__(self, df: pd.DataFrame, candidates: pd.DataFrame,
                 initial_boundaries: List[int], n_trials: int, config,
                 csv_save_path: str = None):
        self.df                = df
        self.candidates        = candidates.sort_values("Score", ascending=False)
        self.boundaries        = sorted(initial_boundaries)
        self.initial_boundaries = sorted(initial_boundaries)
        self.n_trials          = n_trials
        self.config            = config
        self.csv_save_path     = csv_save_path

        # Timeline extents
        self.time_min   = df["PacketTime"].min()
        self.time_max   = df["PacketTime"].max()
        self.time_range = self.time_max - self.time_min

        # Initial window = 1/10 of total recording
        self.window_size = self.time_range / 10
        self.view_start  = self.time_min
        self.view_end    = self.view_start + self.window_size

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> List[int]:
        """Launch the interactive window and return the validated boundaries."""
        self._build_figure()
        self._plot_all()
        self._connect_events()
        self._print_instructions()
        plt.show()
        return sorted(self.boundaries)

    # ------------------------------------------------------------------
    # Figure / widget construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 11))

        # Space at the bottom for two button rows
        toolbar_height = (
            2 * self._BTN_H + self._BTN_GAP + self._TOOLBAR_BOTTOM
        )
        plot_bottom = toolbar_height + 0.05

        # GridSpec: spatial (large) + three signal panels
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(
            4, 1,
            figure=self.fig,
            top=0.94,
            bottom=plot_bottom,
            hspace=0.12,
            height_ratios=[3, 1, 0.75, 0.75]
        )

        self.ax_spatial  = self.fig.add_subplot(gs[0])
        self.ax_temporal = self.fig.add_subplot(gs[1])
        self.ax_velocity = self.fig.add_subplot(gs[2], sharex=self.ax_temporal)
        self.ax_jumps    = self.fig.add_subplot(gs[3], sharex=self.ax_temporal)

        # ── Button layout (two rows, centred) ─────────────────────────
        btn_w = 0.105
        gap   = 0.012

        def _row_left(n_buttons):
            total = n_buttons * btn_w + (n_buttons - 1) * gap
            return (1.0 - total) / 2

        nav_y    = self._TOOLBAR_BOTTOM + self._BTN_H + self._BTN_GAP
        action_y = self._TOOLBAR_BOTTOM

        def _make_btn(left, bottom, label, color="#d8d8d8"):
            ax  = self.fig.add_axes([left, bottom, btn_w, self._BTN_H])
            btn = Button(ax, label, color=color, hovercolor="#a8d8ea")
            btn.label.set_fontsize(9)
            return btn

        # Navigation row (4 buttons)
        x = _row_left(4)
        self.btn_start = _make_btn(x, nav_y, "◀◀ Start")
        x += btn_w + gap
        self.btn_back  = _make_btn(x, nav_y, "◀ Back")
        x += btn_w + gap
        self.btn_fwd   = _make_btn(x, nav_y, "Fwd ▶")
        x += btn_w + gap
        self.btn_end   = _make_btn(x, nav_y, "End ▶▶")

        # Action row (4 buttons)
        x = _row_left(4)
        self.btn_reset = _make_btn(x, action_y, "↺ Reset",  color="#ffe599")
        x += btn_w + gap
        self.btn_auto  = _make_btn(x, action_y, "⚡ Auto",  color="#b6d7a8")
        x += btn_w + gap
        self.btn_clear = _make_btn(x, action_y, "✕ Clear",  color="#ea9999")
        x += btn_w + gap
        self.btn_save  = _make_btn(x, action_y, "💾 Save",  color="#9fc5e8")

        # Wire buttons
        self.btn_start.on_clicked(lambda e: self._jump_to(self.time_min))
        self.btn_back.on_clicked( lambda e: self._pan_view(-self.window_size * 0.8))
        self.btn_fwd.on_clicked(  lambda e: self._pan_view( self.window_size * 0.8))
        self.btn_end.on_clicked(  lambda e: self._jump_to(
            self.time_max - self.window_size
        ))
        self.btn_reset.on_clicked(lambda e: self._do_reset())
        self.btn_auto.on_clicked( lambda e: self._do_auto())
        self.btn_clear.on_clicked(lambda e: self._do_clear())
        self.btn_save.on_clicked( lambda e: print(f"💾 Saved to: {self._save_csv()}"))

    def _connect_events(self):
        self.cid_click  = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_click
        )
        self.cid_key    = self.fig.canvas.mpl_connect(
            "key_press_event", self._on_key
        )
        self.cid_scroll = self.fig.canvas.mpl_connect(
            "scroll_event", self._on_scroll
        )
        self.cid_close  = self.fig.canvas.mpl_connect(
            "close_event", self._on_close
        )

    # ------------------------------------------------------------------
    # Full redraw
    # ------------------------------------------------------------------

    def _plot_all(self):
        self._plot_spatial()
        self._plot_temporal()
        self._plot_velocity()
        self._plot_jumps()
        self._update_suptitle()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------

    def _plot_spatial(self):
        """Spatial trajectory coloured by current trial assignment.
        Each trial's writing area is labelled with its trial number.
        """
        self.ax_spatial.clear()

        # Build temporary trial labels from current boundaries
        temp_df = self.df.copy()
        temp_df["TempTrial"] = 0
        for i, b in enumerate(sorted(self.boundaries)):
            temp_df.loc[b:, "TempTrial"] = i + 1

        _plot_trajectory_by_trial(self.ax_spatial, temp_df, "TempTrial")

        self.ax_spatial.set_aspect(self.config.trajectory_aspect_ratio)
        self.ax_spatial.set_xlabel("X (px)", fontsize=9)
        self.ax_spatial.set_ylabel("Y (px)", fontsize=9)
        self.ax_spatial.grid(True, alpha=0.3)

        n_current = len(self.boundaries) + 1
        n_needed  = self.n_trials
        ok        = n_current == n_needed
        status    = "✓ CORRECT" if ok else f"⚠ NEED {n_needed} TRIALS"
        color     = "green"     if ok else "red"

        self.ax_spatial.set_title(
            f"Spatial Trajectory — {n_current} trial(s)   {status}",
            fontsize=11, fontweight="bold", color=color
        )

    def _plot_temporal(self):
        """Temporal-gap panel (scrollable, clickable).
        Each selected boundary is drawn as a coloured vertical line
        annotated with its trial number T1, T2, …
        """
        self.ax_temporal.clear()

        mask    = (
            (self.df["PacketTime"] >= self.view_start)
            & (self.df["PacketTime"] <= self.view_end)
        )
        view_df = self.df[mask]

        if len(view_df) == 0:
            self.ax_temporal.text(
                0.5, 0.5, "No data in view",
                transform=self.ax_temporal.transAxes,
                ha="center", va="center"
            )
            return

        # Background signal
        self.ax_temporal.plot(
            view_df["PacketTime"], view_df["DeltaT"],
            alpha=0.55, linewidth=0.9, color="steelblue", label="ΔT"
        )

        # Unselected candidate markers
        cand_times  = self.df.loc[self.candidates.index, "PacketTime"]
        cand_in_view = self.candidates[
            (cand_times >= self.view_start) & (cand_times <= self.view_end)
        ]
        unselected = cand_in_view[~cand_in_view.index.isin(self.boundaries)]
        if len(unselected) > 0:
            self.ax_temporal.scatter(
                self.df.loc[unselected.index, "PacketTime"],
                self.df.loc[unselected.index, "DeltaT"],
                c="orange", s=55, alpha=0.7,
                marker="x", linewidths=2, label="Candidates", zorder=4
            )

        # Selected boundaries – coloured vertical lines + trial labels
        sorted_bnd = sorted(self.boundaries)
        for b in sorted_bnd:
            t_val     = self.df.loc[b, "PacketTime"]
            if not (self.view_start <= t_val <= self.view_end):
                continue
            trial_num = sorted_bnd.index(b) + 1
            color     = trial_color(trial_num)
            dt_val    = self.df.loc[b, "DeltaT"]

            self.ax_temporal.axvline(
                t_val, color=color, linewidth=1.8, alpha=0.7, zorder=3
            )
            self.ax_temporal.scatter(
                t_val, dt_val,
                c=[color], s=140, zorder=6,
                edgecolors="black", linewidths=1.8
            )
            self.ax_temporal.annotate(
                f"T{trial_num}",
                xy=(t_val, dt_val),
                xytext=(5, 7), textcoords="offset points",
                fontsize=9, fontweight="bold", color=color, zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, alpha=0.8, linewidth=1)
            )

        # Threshold line
        self.ax_temporal.axhline(
            self.config.min_gap_ms, color="green",
            linestyle="--", alpha=0.5, linewidth=1.2,
            label=f"min_gap ({self.config.min_gap_ms}ms)"
        )

        self.ax_temporal.set_yscale("log")
        self.ax_temporal.set_xlim(self.view_start, self.view_end)
        self.ax_temporal.set_ylabel("ΔT (ms)", fontsize=9)
        self.ax_temporal.set_title(
            "Temporal Gaps — click to add / remove a boundary", fontsize=9
        )
        self.ax_temporal.legend(loc="upper right", fontsize=7, framealpha=0.6)
        self.ax_temporal.grid(True, alpha=0.25)
        self.ax_temporal.tick_params(labelbottom=False)

    def _plot_velocity(self):
        """Velocity panel, aligned with temporal plot."""
        self.ax_velocity.clear()

        mask    = (
            (self.df["PacketTime"] >= self.view_start)
            & (self.df["PacketTime"] <= self.view_end)
        )
        view_df = self.df[mask]
        if len(view_df) == 0:
            return

        self.ax_velocity.plot(
            view_df["PacketTime"], view_df["Velocity"],
            alpha=0.6, linewidth=0.9, color="purple"
        )

        sorted_bnd = sorted(self.boundaries)
        for b in sorted_bnd:
            t_val = self.df.loc[b, "PacketTime"]
            if not (self.view_start <= t_val <= self.view_end):
                continue
            trial_num = sorted_bnd.index(b) + 1
            color     = trial_color(trial_num)
            v_val     = self.df.loc[b, "Velocity"]

            self.ax_velocity.axvline(
                t_val, color=color, linewidth=1.8, alpha=0.7, zorder=3
            )
            self.ax_velocity.scatter(
                t_val, v_val,
                c=[color], s=110, zorder=5,
                edgecolors="black", linewidths=1.5
            )
            self.ax_velocity.annotate(
                f"T{trial_num}",
                xy=(t_val, v_val),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor=color, alpha=0.75, linewidth=1)
            )

        self.ax_velocity.set_xlim(self.view_start, self.view_end)
        self.ax_velocity.set_ylabel("Velocity (px/s)", fontsize=9)
        self.ax_velocity.set_title("Velocity", fontsize=9)
        self.ax_velocity.grid(True, alpha=0.25)
        self.ax_velocity.tick_params(labelbottom=False)

    def _plot_jumps(self):
        """Spatial-jump panel, aligned with temporal and velocity panels."""
        self.ax_jumps.clear()

        mask    = (
            (self.df["PacketTime"] >= self.view_start)
            & (self.df["PacketTime"] <= self.view_end)
        )
        view_df = self.df[mask]
        if len(view_df) == 0:
            return

        self.ax_jumps.plot(
            view_df["PacketTime"], view_df["DistJump"],
            alpha=0.6, linewidth=0.9, color="teal"
        )

        sorted_bnd = sorted(self.boundaries)
        for b in sorted_bnd:
            t_val = self.df.loc[b, "PacketTime"]
            if not (self.view_start <= t_val <= self.view_end):
                continue
            trial_num = sorted_bnd.index(b) + 1
            color     = trial_color(trial_num)
            j_val     = self.df.loc[b, "DistJump"]

            self.ax_jumps.axvline(
                t_val, color=color, linewidth=1.8, alpha=0.7, zorder=3
            )
            self.ax_jumps.scatter(
                t_val, j_val,
                c=[color], s=110, zorder=5,
                edgecolors="black", linewidths=1.5
            )
            self.ax_jumps.annotate(
                f"T{trial_num}",
                xy=(t_val, j_val),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor=color, alpha=0.75, linewidth=1)
            )

        self.ax_jumps.set_xlim(self.view_start, self.view_end)
        self.ax_jumps.set_xlabel("Time (ms)", fontsize=9)
        self.ax_jumps.set_ylabel("Jump (px)", fontsize=9)
        self.ax_jumps.set_title("Spatial Jumps", fontsize=9)
        self.ax_jumps.grid(True, alpha=0.25)

    # ------------------------------------------------------------------
    # Suptitle status bar
    # ------------------------------------------------------------------

    def _update_suptitle(self):
        n_current   = len(self.boundaries) + 1
        n_needed    = self.n_trials
        ok          = n_current == n_needed
        symbol      = "✓" if ok else "⚠"

        progress_pct = (self.view_start - self.time_min) / self.time_range * 100
        window_pct   = self.window_size / self.time_range * 100

        self.fig.suptitle(
            f"Interactive Trial Boundary Validation    "
            f"{symbol}  {n_current}/{n_needed} trials detected\n"
            f"View  {progress_pct:.0f}% – {min(progress_pct + window_pct, 100):.0f}%"
            f"   |   scroll / ← → to pan"
            f"   |   click ΔT panel to toggle boundary"
            f"   |   a=auto   r=reset   c=clear   h=full   s=save",
            fontsize=9, fontweight="bold",
            color="darkgreen" if ok else "darkred",
            y=0.997
        )

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _jump_to(self, new_start: float):
        """Jump the view to *new_start*, clamped to valid range."""
        new_start = max(
            self.time_min,
            min(new_start, self.time_max - self.window_size)
        )
        self.view_start = new_start
        self.view_end   = new_start + self.window_size
        self._redraw_timeseries()

    def _pan_view(self, amount: float):
        """Shift the scrollable window by *amount* ms."""
        new_start = self.view_start + amount
        new_end   = new_start + self.window_size

        if new_start < self.time_min:
            new_start = self.time_min
            new_end   = new_start + self.window_size
        elif new_end > self.time_max:
            new_end   = self.time_max
            new_start = new_end - self.window_size

        self.view_start = new_start
        self.view_end   = new_end
        self._redraw_timeseries()

    def _redraw_timeseries(self):
        """Redraw only the three time-series panels (faster than _plot_all)."""
        self._plot_temporal()
        self._plot_velocity()
        self._plot_jumps()
        self._update_suptitle()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_click(self, event):
        """Toggle nearest candidate on click inside the temporal panel."""
        if event.inaxes != self.ax_temporal:
            return
        click_time = event.xdata
        if click_time is None:
            return

        cand_times   = self.df.loc[self.candidates.index, "PacketTime"]
        in_view      = self.candidates[
            (cand_times >= self.view_start) & (cand_times <= self.view_end)
        ]
        if len(in_view) == 0:
            print("No candidates in current view.")
            return

        distances = np.abs(
            self.df.loc[in_view.index, "PacketTime"] - click_time
        )
        nearest = in_view.index[distances.argmin()]

        # Ignore clicks more than 5% of window width from the nearest candidate
        if distances.min() > self.window_size * 0.05:
            return

        t_s = self.df.loc[nearest, "PacketTime"] / 1000
        if nearest in self.boundaries:
            self.boundaries.remove(nearest)
            print(
                f"✗ Removed boundary at t={t_s:.2f}s  "
                f"→ {len(self.boundaries) + 1} trial(s)"
            )
        else:
            self.boundaries.append(nearest)
            self.boundaries = sorted(self.boundaries)
            print(
                f"✓ Added boundary at t={t_s:.2f}s  "
                f"→ {len(self.boundaries) + 1} trial(s)"
            )

        self._plot_all()

    def _on_scroll(self, event):
        if event.inaxes not in [
            self.ax_temporal, self.ax_velocity, self.ax_jumps
        ]:
            return
        direction     = 1 if event.button == "up" else -1
        scroll_amount = self.window_size * 0.15 * direction
        self._pan_view(scroll_amount)

    def _on_key(self, event):
        if event.key == "left":
            self._pan_view(-self.window_size * 0.4)
        elif event.key == "right":
            self._pan_view( self.window_size * 0.4)
        elif event.key == "h":
            self.view_start  = self.time_min
            self.view_end    = self.time_max
            self.window_size = self.time_range
            print("↻ Full timeline view")
            self._plot_all()
        elif event.key == "a":
            self._do_auto()
        elif event.key == "r":
            self._do_reset()
        elif event.key == "c":
            self._do_clear()
        elif event.key == "s":
            print(f"💾 Saved to: {self._save_csv()}")

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _do_reset(self):
        self.boundaries = sorted(self.initial_boundaries)
        print(f"↻ Reset to initial  → {len(self.boundaries) + 1} trial(s)")
        self._plot_all()

    def _do_auto(self):
        n_needed        = self.n_trials - 1
        self.boundaries = []
        for idx in self.candidates.index:
            if all(
                abs(idx - b) >= self.config.min_sep_samples
                for b in self.boundaries
            ):
                self.boundaries.append(idx)
                if len(self.boundaries) == n_needed:
                    break
        self.boundaries = sorted(self.boundaries)
        print(
            f"⚡ Auto-selected {len(self.boundaries)} boundaries  "
            f"→ {len(self.boundaries) + 1} trial(s)"
        )
        self._plot_all()

    def _do_clear(self):
        self.boundaries = []
        print("✕ Cleared all boundaries  → 1 trial")
        self._plot_all()

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _save_csv(self) -> str:
        from datetime import datetime

        export_df = self.df.copy()
        export_df["IsBoundary"] = False
        if self.boundaries:
            export_df.loc[self.boundaries, "IsBoundary"] = True

        export_df["Trial"] = 0
        for trial_num, b in enumerate(sorted(self.boundaries), start=1):
            export_df.loc[b:, "Trial"] = trial_num

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath  = (
            self.csv_save_path
            if self.csv_save_path
            else f"trial_detection_data_{timestamp}.csv"
        )
        export_df.to_csv(filepath, index=True)
        self.csv_save_path = filepath
        return filepath

    def _on_close(self, event):
        try:
            fp = self._save_csv()
            print("\n" + "=" * 60)
            print("SESSION CLOSED")
            print("=" * 60)
            print(f"✓ Data saved to : {fp}")
            print(f"  Boundaries    : {len(self.boundaries)}")
            print(f"  Trials        : {len(self.boundaries) + 1}")
            print(f"  Indices       : {self.boundaries}")
            print("=" * 60 + "\n")
        except Exception as exc:
            print(f"\n⚠ Error saving CSV: {exc}\n")

    # ------------------------------------------------------------------
    # Console instructions
    # ------------------------------------------------------------------

    def _print_instructions(self):
        w = 64
        print("\n" + "=" * w)
        print("INTERACTIVE VALIDATION MODE")
        print("=" * w)
        print("NAVIGATION")
        print("  Scroll inside ΔT / Velocity / Jump panel   pan timeline")
        print("  ← → keys                                   pan timeline")
        print("  ◀◀ Start  ◀ Back  Fwd ▶  End ▶▶  buttons")
        print("  h key                                       full view")
        print()
        print("EDITING")
        print("  Click inside the ΔT panel  add / remove nearest boundary")
        print("  a key / ⚡ Auto             auto-select top N candidates")
        print("  r key / ↺ Reset            restore initial boundaries")
        print("  c key / ✕ Clear            remove all boundaries")
        print("  s key / 💾 Save            save CSV immediately")
        print()
        print("READING THE DISPLAY")
        print("  Each boundary is labelled T1, T2 … in a unique colour.")
        print("  The same colour and number appears on the trajectory so")
        print("  you can match each gap on the timeline to the word it")
        print("  separates.")
        print()
        total_s = self.time_range / 1000
        win_s   = self.window_size / 1000
        print(f"  Recording : {total_s:.1f}s total")
        print(f"  Window    : {win_s:.1f}s = "
              f"{100 * win_s / total_s:.0f}% of recording")
        print()
        print("  Close the window when done — CSV is auto-saved on close.")
        print("=" * w + "\n")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_trials_auto(df: pd.DataFrame, config,
                       interactive: bool = False,
                       csv_save_path: str = None) -> pd.DataFrame:
    """
    Convenience wrapper for trial detection.

    Args:
        df:            Input DataFrame (PacketTime, X, Y, NormalPressure).
        config:        AnalysisConfig object.
        interactive:   If True, launch interactive validation interface.
        csv_save_path: Optional path to save CSV with boundary data.

    Returns:
        DataFrame with ``Trial`` column added.
    """
    detector = TrialDetector(config)
    df = detector.detect_trials(
        df, interactive=interactive, csv_save_path=csv_save_path
    )
    detector.print_candidate_report()
    detector.diagnose_trial_numbering(df)
    return df